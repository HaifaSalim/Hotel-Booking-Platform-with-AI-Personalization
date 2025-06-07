package reservation.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reservation.model.Hotel;
import reservation.repository.HotelRepository;
import reservation.service.HotelService;
import reservation.service.ReservationService;
import java.util.Optional;
import java.util.List;

@RestController
@RequestMapping("/api/hotels")
@CrossOrigin

public class HotelController {

	@Autowired
	private HotelService hotelService;

	@GetMapping
	public List<Hotel> getAllHotels() {
		return hotelService.getAllHotels();
	}

	@PostMapping
	public Hotel addHotel(@RequestBody Hotel hotel) {
		return hotelService.addHotel(hotel);
	}

	@GetMapping("/with-coordinates")
	public List<Hotel> getHotelsWithCoordinates() {
		return HotelRepository.findAllByLatitudeIsNotNullAndLongitudeIsNotNull();
	}

	@GetMapping("/{id}")
	public ResponseEntity<Hotel> getHotelById(@PathVariable Long id) {
		Optional<Hotel> hotel = hotelService.getHotelById(id);
		return hotel.map(ResponseEntity::ok).orElseGet(() -> ResponseEntity.notFound().build());
	}

}
