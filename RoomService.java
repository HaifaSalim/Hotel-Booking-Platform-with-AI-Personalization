package reservation.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import reservation.model.Room;
import reservation.model.Availability;
import reservation.repository.RoomRepository;

import java.util.List;

@Service
public class RoomService {

	@Autowired
	private RoomRepository roomRepository;

	public List<Room> getRoomsByHotel(Long hotelId) {
		return roomRepository.findByHotelId(hotelId);
	}

	public List<Room> getAvailableRooms(Long hotelId) {
		return roomRepository.findByHotelIdAndAvailability(hotelId, Availability.Available);
	}

	public Room bookRoom(Long roomId) {
		Room room = roomRepository.findById(roomId).orElseThrow(() -> new RuntimeException("Room not found"));

		if (room.getAvailability() == Availability.Booked) {
			throw new RuntimeException("Room is already booked");
		}

		room.setAvailability(Availability.Booked);
		return roomRepository.save(room);
	}

	public Room findById(Long roomId) {
		return roomRepository.findById(roomId).orElseThrow();
	}
}
