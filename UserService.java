package reservation.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import reservation.model.User;
import reservation.model.User.UserRole;
import reservation.repository.UserRepository;

import java.util.Map;
import java.util.Optional;
import java.util.UUID;

@Service
public class UserService {

	@Autowired
	private UserRepository userRepository;

	@Autowired
	private PasswordEncoder passwordEncoder;

	@Autowired
	private EmailService emailService;

	public User registerUser(String username, String email, String password) {

		if (userRepository.findByEmail(email).isPresent()) {
			throw new RuntimeException("Email already in use");
		}
		validatePassword(password);

		String hashedPassword = passwordEncoder.encode(password);
		User user = new User(username, email, hashedPassword);
		user.setRole(UserRole.USER);

		String verificationToken = UUID.randomUUID().toString();
		user.setVerificationToken(verificationToken);
		user.setEnabled(false);

		User savedUser = userRepository.save(user);

		emailService.sendVerificationEmail(savedUser);

		return savedUser;
	}

	public boolean verifyAccount(String token) {
		User user = userRepository.findByVerificationToken(token)
				.orElseThrow(() -> new RuntimeException("Invalid verification token"));

		user.setVerificationToken(null);
		user.setEnabled(true);
		userRepository.save(user);

		return true;
	}

	public Optional<User> authenticate(String email, String password) {
		Optional<User> userOptional = userRepository.findByEmail(email);

		if (userOptional.isPresent()) {
			User user = userOptional.get();

			if (user.isEnabled() && passwordEncoder.matches(password, user.getPassword())) {
				return userOptional;
			}

			if (!user.isEnabled()) {
				throw new RuntimeException("Account not verified. Please check your email for verification link.");
			}
		}

		return Optional.empty();
	}

	public User getUserById(Long id) {
		return userRepository.findById(id).orElseThrow(() -> new RuntimeException("User not found"));
	}

	public Optional<User> getUserByEmail(String email) {
		return userRepository.findByEmail(email);
	}

	public Optional<User> getUserByUsername(String username) {
		return userRepository.findByUsername(username);
	}

	public void changePassword(String username, String oldPassword, String newPassword) {
		User user = userRepository.findByUsername(username).orElseThrow(() -> new RuntimeException("User not found"));

		if (!passwordEncoder.matches(oldPassword, user.getPassword())) {
			throw new RuntimeException("Current password is incorrect");
		}

		validatePassword(newPassword);
		String encodedNewPassword = passwordEncoder.encode(newPassword);

		user.setPassword(encodedNewPassword);
		userRepository.save(user);

		emailService.sendPasswordChangeConfirmation(user);
	}

	public User createUser(User user) {
		if (userRepository.findByEmail(user.getEmail()).isPresent()) {
			throw new RuntimeException("Email already in use");
		}

		validatePassword(user.getPassword());

		user.setPassword(passwordEncoder.encode(user.getPassword()));
		return userRepository.save(user);
	}

	public void initiatePasswordReset(String email) {
		User user = userRepository.findByEmail(email)
				.orElseThrow(() -> new RuntimeException("No account found with that email"));

		String resetToken = UUID.randomUUID().toString();
		user.setResetToken(resetToken);

		user.setResetTokenExpiry(System.currentTimeMillis() + 86400000);

		userRepository.save(user);

		emailService.sendPasswordResetEmail(user);
	}

	public boolean validateResetToken(String token) {
		User user = userRepository.findByResetToken(token)
				.orElseThrow(() -> new RuntimeException("Invalid reset token"));

		if (user.getResetTokenExpiry() < System.currentTimeMillis()) {
			throw new RuntimeException("Reset token has expired");
		}

		return true;
	}

	public void resetPassword(String token, String newPassword) {
		User user = userRepository.findByResetToken(token)
				.orElseThrow(() -> new RuntimeException("Invalid reset token"));

		if (user.getResetTokenExpiry() < System.currentTimeMillis()) {
			throw new RuntimeException("Reset token has expired");
		}

		validatePassword(newPassword);
		user.setPassword(passwordEncoder.encode(newPassword));

		user.setResetToken(null);
		user.setResetTokenExpiry(null);

		userRepository.save(user);

		emailService.sendPasswordChangeConfirmation(user);
	}

	public User updateProfile(Long userId, String username, String email) {
		User user = userRepository.findById(userId).orElseThrow(() -> new RuntimeException("User not found"));

		if (!user.getEmail().equals(email) && userRepository.findByEmail(email).isPresent()) {
			throw new RuntimeException("Email already in use");
		}

		user.setUsername(username);
		user.setEmail(email);

		return userRepository.save(user);
	}

	public void deactivateAccount(Long userId) {
		User user = userRepository.findById(userId).orElseThrow(() -> new RuntimeException("User not found"));

		user.setEnabled(false);
		userRepository.save(user);
	}

	private void validatePassword(String password) {
		if (password == null || password.length() < 8) {
			throw new RuntimeException("Password must be at least 8 characters long");
		}

		boolean hasUppercase = false;
		boolean hasLowercase = false;
		boolean hasDigit = false;
		boolean hasSpecial = false;

		for (char c : password.toCharArray()) {
			if (Character.isUpperCase(c))
				hasUppercase = true;
			if (Character.isLowerCase(c))
				hasLowercase = true;
			if (Character.isDigit(c))
				hasDigit = true;
			if (!Character.isLetterOrDigit(c))
				hasSpecial = true;
		}

		if (!(hasUppercase && hasLowercase && hasDigit && hasSpecial)) {
			throw new RuntimeException("Password must contain uppercase, lowercase, digit, and special character");
		}
	}

}
