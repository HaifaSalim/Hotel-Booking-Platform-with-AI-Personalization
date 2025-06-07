package reservation.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import reservation.model.Hotel;
import reservation.model.Reservation;
import reservation.model.Review;
import reservation.model.User;
import reservation.repository.HotelRepository;
import reservation.repository.PaymentRepository;
import reservation.repository.ReservationRepository;
import reservation.repository.ReviewRepository;
import reservation.repository.RoomRepository;
import reservation.repository.UserRepository;

import java.util.Optional;
import java.util.stream.Collectors;
import java.time.LocalDate;
import java.util.List;

@Service
public class HotelService {

	@Autowired
	private HotelRepository hotelRepository;

	@Autowired
	private RoomRepository roomR;
	@Autowired
	private PaymentRepository paymentR;
	@Autowired
	private ReviewRepository reviewR;
	@Autowired
	private UserRepository userR;
	@Autowired
	private ReservationRepository reservationR;

	public Hotel addHotel(Hotel hotel) {
		return hotelRepository.save(hotel);
	}

	public Hotel findById(Long hotelId) {
		return hotelRepository.findById(hotelId).orElseThrow();
	}

	public Review addReview(Long hotelId, Long userId, int rating, String comment) {
		Hotel hotel = hotelRepository.findById(hotelId)
				.orElseThrow(() -> new RuntimeException("Hotel not found with ID: " + hotelId));
		User user = userR.findById(userId).orElseThrow(() -> new RuntimeException("User not found with ID: " + userId));

		boolean hasReservation = reservationR.existsByUserIdAndHotelId(userId, hotelId);

		if (!hasReservation) {
			throw new RuntimeException("User must have a past reservation at this hotel to leave a review.");
		}

		Review review = new Review();
		review.setHotel(hotel);
		review.setUser(user);
		review.setRating(rating);
		review.setComment(comment);
		review.setReviewDate(LocalDate.now());

		Review savedReview = reviewR.save(review);

		return savedReview;
	}

	public long getReviewCountByHotelId(Long hotelId) {
		return reviewR.countByHotelId(hotelId);
	}

	public List<Hotel> getAllHotels() {
		List<Hotel> hotels = hotelRepository.findAll();

		return hotels.stream().map(hotel -> {
			long reviewCount = getReviewCountByHotelId(hotel.getId());
			hotel.setReviewCount(reviewCount);
			return hotel;
		}).collect(Collectors.toList());
	}

	public Optional<Hotel> getHotelById(Long id) {
		return hotelRepository.findById(id).map(hotel -> {
			long reviewCount = getReviewCountByHotelId(hotel.getId());
			hotel.setReviewCount(reviewCount);
			return hotel;
		});
	}

	public List<Review> getReviewsByHotelId(Long hotelId) {
		return reviewR.findByHotelId(hotelId);
	}

	public List<Review> getReviewsByUserId(Long userId) {
		return reviewR.findByUserId(userId);
	}

}
