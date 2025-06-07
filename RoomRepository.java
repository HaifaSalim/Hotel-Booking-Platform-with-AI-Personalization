package reservation.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import reservation.model.Availability;
import reservation.model.Room;

@Repository
public interface RoomRepository extends JpaRepository<Room, Long> {
    List<Room> findByHotelId(Long hotelId);
    List<Room> findByHotelIdAndAvailability(Long hotelId, Availability availability);
    @Query("SELECT SUM(r.quantity) FROM Room r")
    Integer sumQuantity();

}
