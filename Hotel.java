package reservation.model;

import jakarta.persistence.*;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

@Entity
public class Hotel {

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private Long hotelId;
	private String cityName;
	private String name;
	private String rating;
	private String location;
	private double pricePerNight;
	private String imageUrl;
	private Double latitude;

	@Column(columnDefinition = "TEXT")
	private String attractions, description, hotelFacilities;
	private String faxNumber;
	private String phoneNumber;

	private String pincode;
	@OneToMany(mappedBy = "hotel", cascade = CascadeType.ALL)
	private List<Review> reviews;

	@Transient
	private Integer numericRating;
	@Transient
	private Long reviewCount;

	public Long getReviewCount() {
		return reviewCount;
	}

	public void setReviewCount(Long reviewCount) {
		this.reviewCount = reviewCount;
	}

	public Integer getNumericRating() {
		if (this.rating == null)
			return 0;
		switch (this.rating.toLowerCase()) {
		case "fivestar":
			return 5;
		case "fourstar":
			return 4;
		case "threestar":
			return 3;
		case "twostar":
			return 2;
		case "onestar":
			return 1;
		default:
			return 0;
		}
	}

	@JsonProperty("numericRating")
	public Integer getNumericRatingForJson() {
		return getNumericRating();
	}

	public Double getLatitude() {
		return latitude;
	}

	public void setLatitude(Double latitude) {
		this.latitude = latitude;
	}

	public Double getLongitude() {
		return longitude;
	}

	public void setLongitude(Double longitude) {
		this.longitude = longitude;
	}

	private Double longitude;

	public Long getHotelId() {
		return hotelId;
	}

	public void setHotelId(Long hotelId) {
		this.hotelId = hotelId;
	}

	public String getImageUrl() {
		return imageUrl;
	}

	public void setImageUrl(String imageUrl) {
		this.imageUrl = imageUrl;
	}

	public String getCityName() {
		return cityName;
	}

	public void setCityName(String cityName) {
		this.cityName = cityName;
	}

	public String getRating() {
		return rating;
	}

	public void setRating(String rating) {
		this.rating = rating;
	}

	public String getAttractions() {
		return attractions;
	}

	public void setAttractions(String attractions) {
		this.attractions = attractions;
	}

	public String getDescription() {
		return description;
	}

	public void setDescription(String description) {
		this.description = description;
	}

	public String getfaxNumber() {
		return faxNumber;
	}

	public void setfaxNumber(String faxNumber) {
		this.faxNumber = faxNumber;
	}

	public String gethotelFacilities() {
		return hotelFacilities;
	}

	public void sethotelFacilities(String hotelFacilities) {
		this.hotelFacilities = hotelFacilities;
	}

	public String getphoneNumber() {
		return phoneNumber;
	}

	public void setphoneNumber(String phoneNumber) {
		this.phoneNumber = phoneNumber;
	}

	public String getPincode() {
		return pincode;
	}

	public void setPincode(String pincode) {
		this.pincode = pincode;
	}

	public Long getId() {
		return hotelId;
	}

	public void setId(Long hotelId) {
		this.hotelId = hotelId;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public String getLocation() {
		return location;
	}

	public void setLocation(String location) {
		this.location = location;
	}

	public double getPricePerNight() {
		return pricePerNight;
	}

	public void setPricePerNight(double pricePerNight) {
		this.pricePerNight = pricePerNight;
	}

}
