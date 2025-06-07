package reservation.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import jakarta.servlet.http.HttpSession;
import reservation.model.User;
import reservation.service.UserService;

import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/users")
@CrossOrigin
public class UserController {

	@Autowired
	private UserService userService;

	@PostMapping("/register")
	public ResponseEntity<?> register(@RequestParam String username, @RequestParam String email,
			@RequestParam String password) {
		try {
			User user = userService.registerUser(username, email, password);
			return ResponseEntity.ok("User registered successfully. Please check your email to verify your account.");
		} catch (RuntimeException e) {
			return ResponseEntity.badRequest().body(e.getMessage());
		}
	}

	@GetMapping("/verify")
	public ResponseEntity<?> verifyAccount(@RequestParam String token) {
		try {
			boolean verified = userService.verifyAccount(token);
			if (verified) {
				return ResponseEntity.ok("Your account has been verified successfully. You can now login.");
			} else {
				return ResponseEntity.badRequest().body("Verification failed.");
			}
		} catch (RuntimeException e) {
			return ResponseEntity.badRequest().body(e.getMessage());
		}
	}

	@PostMapping("/login")
	public ResponseEntity<?> login(@RequestParam String email, @RequestParam String password, HttpSession session) {
		Optional<User> user = userService.authenticate(email, password);
		if (user.isPresent()) {
			User loggedInUser = user.get();

			session.setAttribute("userId", loggedInUser.getId());

			return ResponseEntity.ok(Map.of("message", "Login successful", "username", loggedInUser.getUsername(),
					"email", loggedInUser.getEmail()));
		}
		return ResponseEntity.status(401).body("Invalid credentials");
	}

	@GetMapping("/profile")
	public ResponseEntity<?> getUserProfile(HttpSession session) {
		Long userId = (Long) session.getAttribute("userId");

		if (userId == null) {
			return ResponseEntity.status(401).body("User not logged in");
		}

		try {
			User user = userService.getUserById(userId);
			return ResponseEntity.ok(user);
		} catch (RuntimeException e) {
			return ResponseEntity.badRequest().body(e.getMessage());
		}
	}

	@PostMapping("/logout")
	public ResponseEntity<?> logout(HttpSession session) {
		session.invalidate();
		return ResponseEntity.ok("Logged out successfully");
	}

	@PostMapping("/forgot-password")
	public ResponseEntity<?> forgotPassword(@RequestParam String email) {
		try {
			userService.initiatePasswordReset(email);
			return ResponseEntity.ok("Password reset instructions have been sent to your email");
		} catch (RuntimeException e) {
			return ResponseEntity.badRequest().body(e.getMessage());
		}
	}

	@GetMapping("/validate-reset-token")
	public ResponseEntity<?> validateResetToken(@RequestParam String token) {
		try {
			boolean valid = userService.validateResetToken(token);
			if (valid) {
				return ResponseEntity.ok("Token is valid");
			} else {
				return ResponseEntity.badRequest().body("Invalid token");
			}
		} catch (RuntimeException e) {
			return ResponseEntity.badRequest().body(e.getMessage());
		}
	}

	@PostMapping("/reset-password")
	public ResponseEntity<?> resetPassword(@RequestParam String token, @RequestParam String newPassword) {
		try {
			userService.resetPassword(token, newPassword);
			return ResponseEntity.ok("Password has been reset successfully");
		} catch (RuntimeException e) {
			return ResponseEntity.badRequest().body(e.getMessage());
		}
	}
}
