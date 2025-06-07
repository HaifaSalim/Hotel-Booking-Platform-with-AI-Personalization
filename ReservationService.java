package reservation.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import jakarta.transaction.Transactional;
import reservation.model.Availability;
import reservation.model.Hotel;
import reservation.model.Payment;
import reservation.model.Reservation;
import reservation.model.ReservationRequest;
import reservation.model.ReservationStatus;
import reservation.model.Room;
import reservation.model.User;
import reservation.repository.HotelRepository;
import reservation.repository.ReservationRepository;
import reservation.repository.RoomRepository;
import reservation.repository.UserRepository;
import reservation.repository.PaymentRepository;

import java.util.Optional;
import java.util.UUID;
import java.util.stream.Collectors;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

@Service
public class ReservationService {

	@Autowired
	private HotelRepository hotelR;

	@Autowired
	private RoomRepository roomR;

	@Autowired
	private PaymentRepository paymentR;

	@Autowired
	private UserRepository userR;

	@Autowired
	private EmailService emailService;

	@Autowired
	private ReservationRepository reservationR;

	public List<Hotel> listHotels() {
		return hotelR.findAll();
	}

	public Hotel findHotelById(Long id) {
		return hotelR.findById(id).orElse(null);
	}

	public List<Reservation> getAllReservations() {
		return reservationR.findAll();
	}

	public void saveHotel(Hotel hotel) {
		hotelR.save(hotel);
	}

	public Reservation getReservationDetails(String bookingReference) {
		Optional<Reservation> reservationOptional = reservationR.findByBookingReference(bookingReference);
		return reservationOptional.orElse(null);
	}

	public Map<Long, Integer> checkRoomAvailability(Long hotelId, LocalDate checkInDate, LocalDate checkOutDate) {
		if (checkInDate.isAfter(checkOutDate) || checkInDate.isEqual(checkOutDate)) {
			throw new IllegalArgumentException("Check-in date must be before check-out date");
		}

		List<Room> hotelRooms = roomR.findByHotelId(hotelId);

		List<Reservation> overlappingReservations = reservationR.findByHotelIdAndDatesOverlapping(hotelId, checkInDate,
				checkOutDate, ReservationStatus.CANCELLED);

		Map<Long, Integer> roomAvailability = new HashMap<>();

		for (Room room : hotelRooms) {
			roomAvailability.put(room.getRoomId(), room.getMaxQuantity());
		}

		for (Reservation reservation : overlappingReservations) {

			if (reservation.getStatus() == ReservationStatus.CANCELLED) {
				continue;
			}

			for (Map.Entry<Room, Integer> entry : reservation.getRoomQuantities().entrySet()) {
				Room room = entry.getKey();
				Integer bookedQuantity = entry.getValue();

				Long roomId = room.getRoomId();
				if (roomAvailability.containsKey(roomId)) {
					int currentAvailability = roomAvailability.get(roomId);
					roomAvailability.put(roomId, Math.max(0, currentAvailability - bookedQuantity));
				}
			}
		}

		return roomAvailability;
	}

	@Transactional
	public Reservation createReservation(ReservationRequest request) {
		if (!request.getCheckInDate().isBefore(request.getCheckOutDate())) {
			throw new IllegalArgumentException("Check-in date must be earlier than the check-out date.");
		}

		Hotel hotel = hotelR.findById(request.getHotelId()).orElseThrow(() -> new RuntimeException("Hotel not found"));

		List<Room> rooms = roomR.findAllById(request.getRoomIds());
		if (rooms.isEmpty()) {
			throw new RuntimeException("No valid rooms selected.");
		}

		Map<Long, Integer> roomAvailability = checkRoomAvailability(request.getHotelId(), request.getCheckInDate(),
				request.getCheckOutDate());

		for (Room room : rooms) {
			Long roomId = room.getRoomId();
			int requestedQuantity = request.getRoomQuantities().get(roomId);
			int availableQuantity = roomAvailability.getOrDefault(roomId, 0);

			if (requestedQuantity > availableQuantity) {
				throw new RuntimeException("Room " + room.getRoomType()
						+ " is not available in the requested quantity for the selected dates. " + "Available: "
						+ availableQuantity + ", Requested: " + requestedQuantity);
			}
		}

		Optional<User> optionalUser = request.getUserId() != null ? userR.findById(request.getUserId())
				: Optional.empty();

		String bookingReference = "BKNG-" + UUID.randomUUID().toString().substring(0, 6).toUpperCase();

		Reservation reservation = new Reservation();
		reservation.setHotel(hotel);
		reservation.setFirstName(request.getFirstName());
		reservation.setLastName(request.getLastName());
		reservation.setEmail(request.getEmail());
		reservation.setPhone(request.getPhone());
		reservation.setAdditionalRequests(request.getAdditionalRequests());
		reservation.setCheckInDate(request.getCheckInDate());
		reservation.setCheckOutDate(request.getCheckOutDate());
		reservation.setAdults(request.getAdults());
		reservation.setChildren(request.getChildren());
		reservation.setBookingReference(bookingReference);
		optionalUser.ifPresent(reservation::setUser);
		reservation.setStatus(ReservationStatus.CONFIRMED);

		Map<Room, Integer> roomQuantities = new HashMap<>();

		for (Room room : rooms) {
			int requestedQuantity = request.getRoomQuantities().get(room.getRoomId());
			roomQuantities.put(room, requestedQuantity);
		}

		reservation.setRoomQuantities(roomQuantities);
		reservation.setCheckInToken(UUID.randomUUID().toString());
		reservation = reservationR.save(reservation);

		Payment payment = new Payment();
		payment.setPaymentMethod(request.getPaymentMethod());
		payment.setCardHolderName(request.getCardHolderName());
		payment.setCardNumber(request.getCardNumber());
		payment.setExpiryMonth(request.getExpiryMonth());
		payment.setExpiryYear(request.getExpiryYear());
		payment.setCvv(request.getCvv());
		payment.setTotalPrice(request.getTotalPrice());

		reservation.setPayment(payment);
		payment.setReservation(reservation);
		paymentR.save(payment);

		try {
			emailService.ConfirmationEmail(reservation);
		} catch (Exception e) {
			System.err.println("Failed to send confirmation email: " + e.getMessage());
		}

		return reservation;
	}

	@Transactional
	public Reservation cancelReservationWithUserId(Long userId, String bookingReference) {
		Reservation reservation = reservationR.findByUserIdAndBookingReference(userId, bookingReference)
				.orElseThrow(() -> new IllegalArgumentException(
						"Reservation not found with booking reference: " + bookingReference));

		return processCancellation(reservation);
	}

	@Transactional
	public Reservation cancelReservationWithoutUserId(String bookingReference, String email) {
		Reservation reservation = reservationR.findByBookingReferenceAndEmail(bookingReference, email)
				.orElseThrow(() -> new IllegalArgumentException(
						"Reservation not found with booking reference: " + bookingReference));

		return processCancellation(reservation);
	}

	private Reservation processCancellation(Reservation reservation) {

		if (reservation.getStatus() == ReservationStatus.CANCELLED) {
			throw new IllegalStateException("Reservation is already cancelled.");
		}

		LocalDate today = LocalDate.now();
		if (today.isAfter(reservation.getCheckInDate())) {
			throw new IllegalStateException("Cannot cancel a reservation after check-in date.");
		}

		reservation.setStatus(ReservationStatus.CANCELLED);

		Reservation updatedReservation = reservationR.save(reservation);

		try {
			emailService.CancellationEmail(reservation);
		} catch (Exception e) {
			System.err.println("Failed to send cancellation email: " + e.getMessage());
		}

		return updatedReservation;
	}

	public List<Reservation> getUserReservations(Long userId) {

		List<Reservation> allReservations = reservationR.findByUserId(userId);

		List<Reservation> activeReservations = allReservations.stream()
				.filter(reservation -> reservation.getStatus() != ReservationStatus.CANCELLED)
				.collect(Collectors.toList());

		return activeReservations;
	}

	public List<Room> getAvailableRooms(Long hotelId, LocalDate checkInDate, LocalDate checkOutDate) {

		List<Room> allRooms = roomR.findByHotelId(hotelId);

		Map<Long, Integer> availabilityMap = checkRoomAvailability(hotelId, checkInDate, checkOutDate);

		List<Room> availableRooms = new ArrayList<>();

		for (Room room : allRooms) {
			int availableQuantity = availabilityMap.getOrDefault(room.getRoomId(), 0);
			if (availableQuantity > 0) {

				Room availableRoom = new Room();
				availableRoom.setRoomId(room.getRoomId());
				availableRoom.setHotel(room.getHotel());
				availableRoom.setRoomType(room.getRoomType());
				availableRoom.setOccupancy(room.getOccupancy());
				availableRoom.setPrice(room.getPrice());
				availableRoom.setImageUrl(room.getImageUrl());
				availableRoom.setAmenities(room.getAmenities());
				availableRoom.setQuantity(availableQuantity);
				availableRoom.setMaxQuantity(room.getMaxQuantity());
				availableRoom.setAvailability(Availability.Available);

				availableRooms.add(availableRoom);
			}
		}

		return availableRooms;
	}
}
