package reservation.controller;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import reservation.model.Hotel;
import reservation.model.Reservation;
import reservation.model.Room;
import reservation.model.User;
import reservation.service.AdminHotelierService;

@RestController
@RequestMapping("/admin")
@CrossOrigin
public class AdminHotelierController {

	@Autowired
	private AdminHotelierService adminHotelierService;

	@GetMapping("/hotels")
	public List<Hotel> getAllHotels() {
		return adminHotelierService.getAllHotels();
	}

	@GetMapping("/hotels/{id}")
	public ResponseEntity<Hotel> getHotelById(@PathVariable Long id) {
		return adminHotelierService.getHotelById(id).map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
	}

	@PutMapping("/hotels/{id}")
	public ResponseEntity<Hotel> updateHotel(@PathVariable Long id, @RequestBody Hotel updatedHotel) {
		Hotel hotel = adminHotelierService.updateHotel(id, updatedHotel);
		return hotel != null ? ResponseEntity.ok(hotel) : ResponseEntity.notFound().build();
	}

	@DeleteMapping("/hotels/{id}")
	public ResponseEntity<Void> deleteHotel(@PathVariable Long id) {
		adminHotelierService.deleteHotel(id);
		return ResponseEntity.noContent().build();
	}

	@GetMapping("/rooms")
	public List<Room> getAllRooms() {
		return adminHotelierService.getAllRooms();
	}

	@GetMapping("/rooms/{id}")
	public ResponseEntity<Room> getRoomById(@PathVariable Long id) {
		return adminHotelierService.getRoomById(id).map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
	}

	@PutMapping("/rooms/{id}")
	public ResponseEntity<Room> updateRoom(@PathVariable Long id, @RequestBody Room updatedRoom) {
		Room room = adminHotelierService.updateRoom(id, updatedRoom);
		return room != null ? ResponseEntity.ok(room) : ResponseEntity.notFound().build();
	}

	@DeleteMapping("/rooms/{id}")
	public ResponseEntity<Void> deleteRoom(@PathVariable Long id) {
		adminHotelierService.deleteRoom(id);
		return ResponseEntity.noContent().build();
	}

	@GetMapping("/reservations")
	public List<Reservation> getAllReservations() {
		return adminHotelierService.getAllReservations();
	}

	@GetMapping("/reservations/{id}")
	public ResponseEntity<Reservation> getReservationById(@PathVariable Long id) {
		return adminHotelierService.getReservationById(id).map(ResponseEntity::ok)
				.orElse(ResponseEntity.notFound().build());
	}

	@PutMapping("/reservations/{id}")
	public ResponseEntity<Reservation> updateReservation(@PathVariable Long id,
			@RequestBody AdminHotelierService.UpdateReservationRequest updateRequest) {

		Reservation reservation = adminHotelierService.updateReservation(id, updateRequest);
		return reservation != null ? ResponseEntity.ok(reservation) : ResponseEntity.notFound().build();
	}

	@DeleteMapping("/reservations/{id}")
	public ResponseEntity<Void> deleteReservation(@PathVariable Long id) {
		adminHotelierService.deleteReservation(id);
		return ResponseEntity.noContent().build();
	}

	@GetMapping("/users")
	public List<User> getAllUsers() {
		return adminHotelierService.getAllUsers();
	}

	@GetMapping("/users/{id}")
	public ResponseEntity<User> getUserById(@PathVariable Long id) {
		return adminHotelierService.getUserById(id).map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
	}

	@PutMapping("/users/{id}")
	public ResponseEntity<User> updateUser(@PathVariable Long id, @RequestBody Map<String, Object> updates) {
		User updatedUser = adminHotelierService.UpdateUser(id, updates);
		return updatedUser != null ? ResponseEntity.ok(updatedUser) : ResponseEntity.notFound().build();
	}

	@DeleteMapping("/users/{id}")
	public ResponseEntity<Void> deleteUser(@PathVariable Long id) {
		adminHotelierService.deleteUser(id);
		return ResponseEntity.noContent().build();
	}

	@GetMapping("/dashboard")
	public ResponseEntity<Map<String, Object>> getDashboardData() {
		Map<String, Object> dashboardData = adminHotelierService.getDashboardData();
		return ResponseEntity.ok(dashboardData);
	}

	@GetMapping("/verify")
	public ResponseEntity<?> verifyCheckIn(@RequestParam String token) {
		Map<String, Object> response = adminHotelierService.verifyCheckIn(token);

		if ((Boolean) response.get("verified")) {
			return ResponseEntity.ok(response);
		} else {
			return ResponseEntity.badRequest().body(response);
		}
	}
}