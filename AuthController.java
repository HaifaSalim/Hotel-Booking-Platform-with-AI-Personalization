package reservation.controller;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.web.bind.annotation.*;

import reservation.model.User;
import reservation.model.User.UserRole;
import reservation.repository.UserRepository;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/auth")
@CrossOrigin
public class AuthController {

	@Autowired
	private UserRepository userRepository;

	private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

	@PostMapping("/setup")
	public ResponseEntity<String> setup() {

		User admin = new User();
		admin.setUsername("admin");
		admin.setEmail("admin@hotel.com");
		admin.setPassword(passwordEncoder.encode("adminPass123"));
		admin.setRole(UserRole.ADMIN);
		userRepository.save(admin);

		User hotelier = new User();
		hotelier.setUsername("hotelier");
		hotelier.setEmail("hotelier@hotel.com");
		hotelier.setPassword(passwordEncoder.encode("hotelierPass123"));
		hotelier.setRole(UserRole.HOTELIER);
		userRepository.save(hotelier);

		return ResponseEntity.ok("Initial users created");
	}

	@PostMapping("/login")
	public ResponseEntity<?> login(@RequestBody Map<String, String> loginRequest, HttpServletRequest request) {
		String username = loginRequest.get("username");
		String password = loginRequest.get("password");

		Optional<User> userOptional = userRepository.findByUsername(username);

		if (userOptional.isPresent() && passwordEncoder.matches(password, userOptional.get().getPassword())) {

			User user = userOptional.get();
			HttpSession session = request.getSession(true);
			session.setAttribute("userId", user.getId());
			session.setAttribute("userRole", user.getRole());

			Map<String, Object> response = new HashMap<>();
			response.put("role", user.getRole());
			response.put("message", "Login successful");

			return ResponseEntity.ok(response);
		}

		return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("message", "Invalid credentials"));
	}

	@PostMapping("/logout")
	public ResponseEntity<?> logout(HttpServletRequest request) {
		HttpSession session = request.getSession(false);
		if (session != null) {
			session.invalidate();
		}
		return ResponseEntity.ok(Map.of("message", "Logout successful"));
	}

	@GetMapping("/verify-role")
	public ResponseEntity<?> verifyRole(HttpServletRequest request, @RequestParam String requiredRole) {
		HttpSession session = request.getSession(false);
		if (session != null && session.getAttribute("userId") != null) {
			UserRole userRole = (UserRole) session.getAttribute("userRole");
			boolean hasAccess = false;

			if (requiredRole.equals("ADMIN") && userRole == UserRole.ADMIN) {
				hasAccess = true;
			} else if (requiredRole.equals("HOTELIER")
					&& (userRole == UserRole.ADMIN || userRole == UserRole.HOTELIER)) {
				hasAccess = true;
			}

			return ResponseEntity.ok(Map.of("hasAccess", hasAccess));
		}
		return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("hasAccess", false));
	}

	@GetMapping("/check-auth")
	public ResponseEntity<?> checkAuth(HttpServletRequest request) {
		HttpSession session = request.getSession(false);
		if (session != null && session.getAttribute("userId") != null) {
			return ResponseEntity.ok(Map.of("authenticated", true, "role", session.getAttribute("userRole")));
		}
		return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("authenticated", false));
	}
}