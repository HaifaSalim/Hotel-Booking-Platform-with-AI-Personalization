package reservation.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reservation.model.Hotel;
import reservation.model.Reservation;
import reservation.model.ReservationRequest;
import reservation.model.Room;
import reservation.service.ReservationService;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/reservations")
@CrossOrigin
public class ReservationController {

	@Autowired
	private ReservationService reservationService;

	@GetMapping
	public ResponseEntity<List<Reservation>> getAllReservations() {
		return ResponseEntity.ok(reservationService.getAllReservations());
	}

	@PostMapping("/book")
	public ResponseEntity<Reservation> createReservation(@RequestBody ReservationRequest request) {
		return ResponseEntity.ok(reservationService.createReservation(request));

	}

	@GetMapping("/user/{userId}")
	public ResponseEntity<List<Reservation>> getUserReservations(@PathVariable Long userId) {
		List<Reservation> reservations = reservationService.getUserReservations(userId);
		return ResponseEntity.ok(reservations);
	}

	@DeleteMapping("/cancel/user")
	public ResponseEntity<?> cancelReservationWithUserId(@RequestParam("userId") Long userId,
			@RequestParam("bookingReference") String bookingReference) {

		try {

			if (userId == null || userId == 0 || bookingReference == null || bookingReference.isEmpty()) {
				return ResponseEntity.badRequest().body("User ID and booking reference are required.");
			}

			Reservation cancelledReservation = reservationService.cancelReservationWithUserId(userId, bookingReference);
			return ResponseEntity.ok(cancelledReservation);
		} catch (Exception e) {
			return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
					.body("An error occurred while cancelling the reservation: " + e.getMessage());
		}
	}

	@DeleteMapping("/cancel/guest")
	public ResponseEntity<?> cancelReservationWithEmail(@RequestParam("email") String email,
			@RequestParam("bookingReference") String bookingReference) {

		try {

			if (email == null || email.isEmpty() || bookingReference == null || bookingReference.isEmpty()) {
				return ResponseEntity.badRequest().body("Email and booking reference are required.");
			}

			Reservation cancelledReservation = reservationService.cancelReservationWithoutUserId(bookingReference,
					email);
			return ResponseEntity.ok(cancelledReservation);
		} catch (Exception e) {
			return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
					.body("An error occurred while cancelling the reservation: " + e.getMessage());
		}

	}

	@GetMapping("/details/{bookingReference}")
	public ResponseEntity<Reservation> getReservationDetails(@PathVariable String bookingReference) {
		Reservation reservation = reservationService.getReservationDetails(bookingReference);

		if (reservation != null) {

			Map<String, Integer> roomTypeCounts = new HashMap<>();

			if (reservation.getRoomQuantities() != null && !reservation.getRoomQuantities().isEmpty()) {
				for (Map.Entry<Room, Integer> entry : reservation.getRoomQuantities().entrySet()) {
					String roomType = entry.getKey().getRoomType();
					roomTypeCounts.put(roomType, roomTypeCounts.getOrDefault(roomType, 0) + entry.getValue());
				}

				reservation.setRoomTypeCounts(roomTypeCounts);
			}

			return ResponseEntity.ok(reservation);
		} else {
			return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
		}
	}

}
