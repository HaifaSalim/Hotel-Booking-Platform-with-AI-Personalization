package reservation;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class SecurityConfig {

	@Bean
	public PasswordEncoder passwordEncoder() {
		return new BCryptPasswordEncoder();
	}

	@Bean
	public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
		http.authorizeHttpRequests(auth -> auth
				.requestMatchers("/", "/index", "/home", "/css/**", "/js/**", "/users/register", "/users/login",
						"/admin", "/adminlogin", "/hoteliers", "/users/forgot-password", "/users/reset-password")
				.permitAll().requestMatchers("/reservations/**").authenticated().anyRequest().permitAll())
				.formLogin(login -> login.loginPage("/login").permitAll())
				.logout(logout -> logout.logoutUrl("/logout").permitAll())
				.requiresChannel(channel -> channel.anyRequest().requiresSecure())
				.headers(headers -> headers
						.httpStrictTransportSecurity(hsts -> hsts.includeSubDomains(true).maxAgeInSeconds(31536000)))
				.csrf(csrf -> csrf.disable());
		return http.build();
	}

	@Configuration
	public class WebConfig implements WebMvcConfigurer {
		@Override
		public void addCorsMappings(CorsRegistry registry) {
			registry.addMapping("/**").allowedOriginPatterns("*")
					.allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS").allowedHeaders("*")
					.allowCredentials(true);
		}
	}
}
