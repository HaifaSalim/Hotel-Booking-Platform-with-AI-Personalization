package reservation.controller;

import java.time.LocalDate;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reservation.model.Hotel;
import reservation.model.Reservation;

import reservation.model.Room;
import reservation.service.ReservationService;
import reservation.service.RoomService;

@RestController
@RequestMapping("/api/rooms")
@CrossOrigin
public class RoomController {

	@Autowired
	private RoomService roomService;
	@Autowired
	private ReservationService reservationService;

	@GetMapping("/hotel/{hotelId}")
	public ResponseEntity<List<Room>> getRooms(@PathVariable Long hotelId) {
		return ResponseEntity.ok(roomService.getRoomsByHotel(hotelId));
	}

	@GetMapping("/hotel/{hotelId}/rooms")
	public ResponseEntity<List<Room>> getAvailableRooms(@PathVariable Long hotelId, @RequestParam LocalDate checkIn,
			@RequestParam LocalDate checkOut) {

		List<Room> availableRooms = reservationService.getAvailableRooms(hotelId, checkIn, checkOut);
		return ResponseEntity.ok(availableRooms);
	}

	@PostMapping("/book/{roomId}")
	public ResponseEntity<Room> bookRoom(@PathVariable Long roomId) {
		return ResponseEntity.ok(roomService.bookRoom(roomId));
	}

	@GetMapping("/{roomId}")
	public ResponseEntity<?> getRoomDetails(@PathVariable Long roomId) {
		try {
			Room room = roomService.findById(roomId);
			if (room != null) {
				return ResponseEntity.ok(room);
			}
			return ResponseEntity.notFound().build();
		} catch (Exception e) {
			return ResponseEntity.internalServerError().body("Error fetching room: " + e.getMessage());
		}
	}
}
