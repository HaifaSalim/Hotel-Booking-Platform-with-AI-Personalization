package reservation.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import reservation.model.Review;
import reservation.service.HotelService;

import java.time.LocalDate;
import java.util.List;

@RestController
@RequestMapping("/api/reviews")
public class ReviewController {

	@Autowired
	private HotelService hotelService;

	@PostMapping
	public Review addReview(@RequestBody ReviewRequest request) {
		System.out.println(
				"Received review request: Hotel ID = " + request.getHotelId() + ", User ID = " + request.getUserId());

		return hotelService.addReview(request.getHotelId(), request.getUserId(), request.getRating(),
				request.getComment());
	}

	@GetMapping("/hotel/{hotelId}")
	public List<Review> getReviewsByHotelId(@PathVariable Long hotelId) {
		return hotelService.getReviewsByHotelId(hotelId);
	}

	@GetMapping("/user/{userId}")
	public List<Review> getReviewsByUserId(@PathVariable Long userId) {
		return hotelService.getReviewsByUserId(userId);
	}

	static class ReviewRequest {
		private Long hotelId;
		private Long userId;
		private int rating;
		private String comment;

		public Long getHotelId() {
			return hotelId;
		}

		public void setHotelId(Long hotelId) {
			this.hotelId = hotelId;
		}

		public Long getUserId() {
			return userId;
		}

		public void setUserId(Long userId) {
			this.userId = userId;
		}

		public int getRating() {
			return rating;
		}

		public void setRating(int rating) {
			this.rating = rating;
		}

		public String getComment() {
			return comment;
		}

		public void setComment(String comment) {
			this.comment = comment;
		}
	}

}
