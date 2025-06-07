package reservation.model;

import jakarta.persistence.*;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonBackReference;
import com.fasterxml.jackson.annotation.JsonManagedReference;

@Entity
@Table(name = "reservation")
public class Reservation {

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Column(name = "reservation_id")
	private Long reservationId;

	@Column(name = "booking_reference", nullable = false, unique = true)
	private String bookingReference;
	private LocalDate checkInDate;
	private LocalDate checkOutDate;
	private String firstName;
	private String lastName;
	private String email;
	private String phone;

	@Column(name = "additionalRequests")
	private String additionalRequests;

	@ManyToOne
	@JoinColumn(name = "user_id", nullable = true)
	@JsonBackReference
	private User user;

	@ManyToOne
	@JoinColumn(name = "hotel_id", nullable = false)
	// @JsonBackReference
	private Hotel hotel;

	@ManyToMany
	@JoinTable(name = "reservation_rooms", joinColumns = @JoinColumn(name = "reservation_id"), inverseJoinColumns = @JoinColumn(name = "room_id"))
	@JsonManagedReference
	private List<Room> rooms = new ArrayList<>();

	@Enumerated(EnumType.STRING)
	private ReservationStatus status;

	@JsonManagedReference
	@OneToOne(mappedBy = "reservation", cascade = CascadeType.ALL)
	private Payment payment;

	@ElementCollection
	@CollectionTable(name = "reservation_rooms", joinColumns = @JoinColumn(name = "reservation_id"))
	@MapKeyJoinColumn(name = "room_id")
	@Column(name = "quantity", nullable = false)
	private Map<Room, Integer> roomQuantities = new HashMap<>();
	private String checkInToken;
	@Transient
	private Map<String, Integer> roomTypeCounts;

	private int children;

	private int adults;

	public Map<String, Integer> getRoomTypeCounts() {
		return roomTypeCounts;
	}

	public int getAdults() {
		return adults;
	}

	public void setAdults(int adults) {
		this.adults = adults;
	}

	public int getChildren() {
		return children;
	}

	public void setChildren(int children) {
		this.children = children;
	}

	public void setRoomTypeCounts(Map<String, Integer> roomTypeCounts) {
		this.roomTypeCounts = roomTypeCounts;
	}

	public Map<Room, Integer> getRoomQuantities() {
		return roomQuantities;
	}

	public void setRoomQuantities(Map<Room, Integer> roomQuantities) {
		this.roomQuantities = roomQuantities;
	}

	public User getUser() {
		return user;
	}

	public void setUser(User user) {
		this.user = user;
	}

	public List<Room> getRooms() {
		return rooms;
	}

	public void setRooms(List<Room> rooms) {
		this.rooms = rooms;
	}

	public ReservationStatus getStatus() {
		return status;
	}

	public void setStatus(ReservationStatus status) {
		this.status = status;
	}

	public Long getReservationId() {
		return reservationId;
	}

	public void setReservationId(Long reservationId) {
		this.reservationId = reservationId;
	}

	public String getBookingReference() {
		return bookingReference;
	}

	public void setBookingReference(String bookingReference) {
		this.bookingReference = bookingReference;
	}

	public LocalDate getCheckInDate() {
		return checkInDate;
	}

	public void setCheckInDate(LocalDate checkInDate) {
		this.checkInDate = checkInDate;
	}

	public LocalDate getCheckOutDate() {
		return checkOutDate;
	}

	public void setCheckOutDate(LocalDate checkOutDate) {
		this.checkOutDate = checkOutDate;
	}

	public String getFirstName() {
		return firstName;
	}

	public void setFirstName(String firstName) {
		this.firstName = firstName;
	}

	public String getLastName() {
		return lastName;
	}

	public void setLastName(String lastName) {
		this.lastName = lastName;
	}

	public String getEmail() {
		return email;
	}

	public void setEmail(String email) {
		this.email = email;
	}

	public String getPhone() {
		return phone;
	}

	public void setPhone(String phone) {
		this.phone = phone;
	}

	public String getAdditionalRequests() {
		return additionalRequests;
	}

	public void setAdditionalRequests(String additionalRequests) {
		this.additionalRequests = additionalRequests;
	}

	public Hotel getHotel() {
		return hotel;
	}

	public void setHotel(Hotel hotel) {
		this.hotel = hotel;
	}

	public void setPayment(Payment payment) {
		this.payment = payment;
		if (payment != null) {
			payment.setReservation(this);
		}
	}

	public Payment getPayment() {
		return payment;
	}

	public String getCheckInToken() {
		return checkInToken;
	}

	public void setCheckInToken(String checkInToken) {
		this.checkInToken = checkInToken;
	}

}
