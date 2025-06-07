package reservation.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import reservation.model.User;

public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByEmail(String email);
    Optional<User> findByUsername(String username);
	Optional<User> findByVerificationToken(String token);
	Optional<User> findByResetToken(String token);

}