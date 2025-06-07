package reservation.repository;


import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import reservation.model.Hotel;
public interface HotelRepository extends JpaRepository<Hotel, Long> {

	static List<Hotel> findAllByLatitudeIsNotNullAndLongitudeIsNotNull() {
		// TODO Auto-generated method stub
		return null;
	}

}