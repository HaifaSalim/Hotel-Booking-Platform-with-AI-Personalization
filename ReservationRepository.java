package reservation.repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import reservation.model.Reservation;
import reservation.model.ReservationStatus;

public interface ReservationRepository extends JpaRepository<Reservation, Long> {
	   List<Reservation> findByUserId(Long userId);

	Optional<Reservation> findByBookingReference(String bookingReference);

	Optional<Reservation> findByUserIdAndBookingReference(Long userId, String bookingReference);

	Optional<Reservation> findByBookingReferenceAndEmail(String bookingReference, String email);

	boolean existsByUserIdAndHotelId(Long userId, Long hotelId);

  
    @Query("SELECT r FROM Reservation r WHERE r.hotel.hotelId = :hotelId " +
           "AND r.status <> :cancelledStatus " +
           "AND ((r.checkInDate <= :checkOutDate) AND (r.checkOutDate >= :checkInDate))")
    List<Reservation> findByHotelIdAndDatesOverlapping(
        @Param("hotelId") Long hotelId,
        @Param("checkInDate") LocalDate checkInDate,
        @Param("checkOutDate") LocalDate checkOutDate,
        @Param("cancelledStatus") ReservationStatus cancelledStatus);

	Optional<Reservation> findByCheckInToken(String token);

	long countByCheckInDateAfter(LocalDate thirtyDaysAgo);

	
	@Query("SELECT SUM(r.payment.totalPrice) FROM Reservation r WHERE r.checkInDate >= :date")
	Double sumTotalPriceByCheckInDateAfter(@Param("date") LocalDate date);

	
	@Query("SELECT COUNT(r) FROM Reservation r WHERE r.checkInDate <= CURRENT_DATE AND r.checkOutDate > CURRENT_DATE AND (r.status = 'CONFIRMED' OR r.status = 'CHECKED_IN')")
	Integer countOccupiedRoomsToday();

	long countByCheckInDateAndStatus(LocalDate date, ReservationStatus status);

	Long countByCheckInDate(LocalDate date);

	
	List<Reservation> findByCheckInDateBetweenAndStatus(
	    LocalDate startDate, LocalDate endDate, ReservationStatus status);
}
