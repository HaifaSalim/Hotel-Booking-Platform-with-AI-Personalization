package reservation.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import reservation.model.Payment;

@Repository
public interface PaymentRepository extends JpaRepository<Payment, Long> {
   
}
