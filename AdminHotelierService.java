package reservation.service;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.fasterxml.jackson.annotation.JsonFormat;

import reservation.model.Hotel;
import reservation.model.Reservation;
import reservation.model.ReservationStatus;
import reservation.model.Room;
import reservation.model.User;
import reservation.repository.HotelRepository;
import reservation.repository.ReservationRepository;
import reservation.repository.RoomRepository;
import reservation.repository.UserRepository;

@Service
public class AdminHotelierService {

	@Autowired
	private HotelRepository hotelRepository;

	@Autowired
	private EmailService emailService;
	@Autowired
	private RoomRepository roomRepository;

	@Autowired
	private ReservationRepository reservationRepository;

	@Autowired
	private UserRepository userRepository;

	public List<Hotel> getAllHotels() {
		return hotelRepository.findAll();
	}

	public Optional<Hotel> getHotelById(Long id) {
		return hotelRepository.findById(id);
	}

	public Hotel updateHotel(Long id, Hotel updatedHotel) {
		return hotelRepository.findById(id).map(hotel -> {
			hotel.setName(updatedHotel.getName());
			hotel.setLocation(updatedHotel.getLocation());
			return hotelRepository.save(hotel);
		}).orElse(null);
	}

	public void deleteHotel(Long id) {
		hotelRepository.deleteById(id);
	}

	public List<Room> getAllRooms() {
		return roomRepository.findAll();
	}

	public Optional<Room> getRoomById(Long id) {
		return roomRepository.findById(id);
	}

	public Room updateRoom(Long id, Room updatedRoom) {
		return roomRepository.findById(id).map(room -> {
			room.setRoomType(updatedRoom.getRoomType());
			room.setPrice(updatedRoom.getPrice());
			room.setOccupancy(updatedRoom.getOccupancy());
			room.setQuantity(updatedRoom.getQuantity());
			room.setAmenities(updatedRoom.getAmenities());
			room.setAvailability(updatedRoom.getAvailability());
			return roomRepository.save(room);
		}).orElse(null);
	}

	public void deleteRoom(Long id) {
		roomRepository.deleteById(id);
	}

	public List<Reservation> getAllReservations() {
		return reservationRepository.findAll();
	}

	public Optional<Reservation> getReservationById(Long id) {
		return reservationRepository.findById(id);
	}

	public Reservation updateReservation(Long id, UpdateReservationRequest updateRequest) {
		return reservationRepository.findById(id).map(reservation -> {
			reservation.setFirstName(updateRequest.getFirstName());
			reservation.setLastName(updateRequest.getLastName());
			reservation.setCheckInDate(updateRequest.getCheckInDate());
			reservation.setCheckOutDate(updateRequest.getCheckOutDate());
			reservation.setStatus(updateRequest.getStatus());
			return reservationRepository.save(reservation);
		}).orElse(null);
	}

	public void deleteReservation(Long id) {
		Optional<Reservation> optionalReservation = reservationRepository.findById(id);

		if (optionalReservation.isPresent()) {
			Reservation reservation = optionalReservation.get();

			try {
				emailService.CancellationEmail(reservation);
			} catch (Exception e) {
				System.err.println("Failed to send cancellation email: " + e.getMessage());
			}

			reservationRepository.deleteById(id);
		} else {
			System.err.println("Reservation with ID " + id + " not found.");
		}
	}

	public List<User> getAllUsers() {
		return userRepository.findAll();
	}

	public Optional<User> getUserById(Long id) {
		return userRepository.findById(id);
	}

	public User updateUser(Long id, User updatedUser) {
		return userRepository.findById(id).map(user -> {
			user.setUsername(updatedUser.getUsername());
			user.setEmail(updatedUser.getEmail());
			return userRepository.save(user);
		}).orElse(null);
	}

	public User UpdateUser(Long id, Map<String, Object> updates) {
		Optional<User> optionalUser = userRepository.findById(id);
		if (optionalUser.isEmpty())
			return null;

		User user = optionalUser.get();

		updates.forEach((key, value) -> {
			switch (key) {
			case "username":
				user.setUsername((String) value);
				break;
			case "email":
				user.setEmail((String) value);
				break;
			}
		});

		return userRepository.save(user);
	}

	public void deleteUser(Long id) {
		userRepository.deleteById(id);
	}

	public Map<String, Object> getDashboardData() {
		Map<String, Object> dashboardData = new HashMap<>();

		LocalDate thirtyDaysAgo = LocalDate.now().minusDays(30);
		Long totalBookings = reservationRepository.countByCheckInDateAfter(thirtyDaysAgo);
		dashboardData.put("totalBookings", totalBookings);

		Double revenue = reservationRepository.sumTotalPriceByCheckInDateAfter(thirtyDaysAgo);
		dashboardData.put("revenue", revenue);

		int totalRooms = roomRepository.sumQuantity();
		int occupiedRooms = reservationRepository.countOccupiedRoomsToday();
		double occupancyRate = (double) occupiedRooms / totalRooms * 100;
		dashboardData.put("occupancyRate", Math.round(occupancyRate * 10) / 10.0);

		long checkInsToday = reservationRepository.countByCheckInDateAndStatus(LocalDate.now(),
				ReservationStatus.CONFIRMED);
		dashboardData.put("checkInsToday", checkInsToday);

		dashboardData.put("bookingTrends", getBookingTrendsData());

		dashboardData.put("roomTypeDistribution", getRoomTypeDistribution());

		List<Map<String, Object>> upcomingReservations = getUpcomingReservations();
		dashboardData.put("upcomingReservations", upcomingReservations);

		return dashboardData;
	}

	private Map<String, Object> getBookingTrendsData() {
		Map<String, Object> result = new HashMap<>();
		List<String> labels = new ArrayList<>();
		List<Long> bookingsCount = new ArrayList<>();

		LocalDate today = LocalDate.now();
		LocalDate startDate = today.minusDays(29);

		for (int i = 0; i < 30; i++) {
			LocalDate date = startDate.plusDays(i);
			labels.add(date.format(DateTimeFormatter.ofPattern("MMM dd")));

			Long count = reservationRepository.countByCheckInDate(date);
			bookingsCount.add(count);
		}

		result.put("labels", labels);
		result.put("bookings", bookingsCount);

		return result;
	}

	private Map<String, Object> getRoomTypeDistribution() {
		Map<String, Object> result = new HashMap<>();
		List<String> labels = new ArrayList<>();
		List<Integer> values = new ArrayList<>();

		List<Room> allRooms = roomRepository.findAll();

		Map<String, Integer> roomTypeMap = new HashMap<>();
		for (Room room : allRooms) {
			String roomType = room.getRoomType();
			int quantity = room.getQuantity();
			roomTypeMap.put(roomType, roomTypeMap.getOrDefault(roomType, 0) + quantity);
		}

		for (Map.Entry<String, Integer> entry : roomTypeMap.entrySet()) {
			labels.add(entry.getKey());
			values.add(entry.getValue());
		}

		result.put("labels", labels);
		result.put("values", values);

		return result;
	}

	private List<Map<String, Object>> getUpcomingReservations() {
		LocalDate today = LocalDate.now();
		LocalDate nextWeek = today.plusDays(7);

		List<Reservation> upcomingReservations = reservationRepository.findByCheckInDateBetweenAndStatus(today,
				nextWeek, ReservationStatus.CONFIRMED);

		List<Map<String, Object>> result = new ArrayList<>();

		for (Reservation res : upcomingReservations) {
			Map<String, Object> reservationData = new HashMap<>();

			reservationData.put("guestName", res.getFirstName() + " " + res.getLastName());
			reservationData.put("checkInDate",
					res.getCheckInDate().format(DateTimeFormatter.ofPattern("MMM dd, yyyy")));

			long nights = ChronoUnit.DAYS.between(res.getCheckInDate(), res.getCheckOutDate());
			reservationData.put("nights", nights);

			String roomType = res.getRoomQuantities().keySet().iterator().next().getRoomType();
			reservationData.put("roomType", roomType);

			reservationData.put("status", res.getStatus().toString());
			reservationData.put("id", res.getReservationId());

			result.add(reservationData);
		}

		return result;
	}

	public Map<String, Object> verifyCheckIn(String token) {
		Map<String, Object> response = new HashMap<>();

		Optional<Reservation> reservationOpt = reservationRepository.findByCheckInToken(token);

		if (reservationOpt.isPresent()) {
			Reservation reservation = reservationOpt.get();

			reservation.setStatus(ReservationStatus.CHECKED_IN);
			reservation.setCheckInToken(null);
			reservationRepository.save(reservation);

			int totalRoomCount = reservation.getRoomQuantities().values().stream().mapToInt(Integer::intValue).sum();

			Map<String, Integer> roomTypeCounts = new HashMap<>();
			for (Map.Entry<Room, Integer> entry : reservation.getRoomQuantities().entrySet()) {
				String roomType = entry.getKey().getRoomType();
				roomTypeCounts.put(roomType, roomTypeCounts.getOrDefault(roomType, 0) + entry.getValue());
			}

			response.put("verified", true);
			response.put("bookingReference", reservation.getBookingReference());
			response.put("guestName", reservation.getFirstName() + " " + reservation.getLastName());
			response.put("hotelName", reservation.getHotel().getName());
			response.put("checkInDate", reservation.getCheckInDate());
			response.put("checkOutDate", reservation.getCheckOutDate());
			response.put("roomCount", totalRoomCount);
			response.put("roomTypes", roomTypeCounts);
			response.put("paymentMethod", reservation.getPayment().getPaymentMethod());
			response.put("totalPrice", reservation.getPayment().getTotalPrice());
			response.put("status", "CHECKED_IN");
		} else {
			response.put("verified", false);
			response.put("message", "Invalid check-in token");
		}

		return response;
	}

	public static class UpdateReservationRequest {
		private String firstName;
		private String lastName;
		private LocalDate checkInDate;
		private LocalDate checkOutDate;
		private ReservationStatus status;

		public UpdateReservationRequest() {
		}

		public String getFirstName() {
			return firstName;
		}

		public void setFirstName(String firstName) {
			this.firstName = firstName;
		}

		public String getLastName() {
			return lastName;
		}

		public void setLastName(String lastName) {
			this.lastName = lastName;
		}

		@JsonFormat(pattern = "yyyy-MM-dd")
		public LocalDate getCheckInDate() {
			return checkInDate;
		}

		public void setCheckInDate(LocalDate checkInDate) {
			this.checkInDate = checkInDate;
		}

		@JsonFormat(pattern = "yyyy-MM-dd")
		public LocalDate getCheckOutDate() {
			return checkOutDate;
		}

		public void setCheckOutDate(LocalDate checkOutDate) {
			this.checkOutDate = checkOutDate;
		}

		public ReservationStatus getStatus() {
			return status;
		}

		public void setStatus(ReservationStatus status) {
			this.status = status;
		}
	}
}