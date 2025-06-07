package reservation.service;

import java.io.ByteArrayOutputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;

import com.google.zxing.BarcodeFormat;
import com.google.zxing.EncodeHintType;
import com.google.zxing.client.j2se.MatrixToImageWriter;
import com.google.zxing.common.BitMatrix;
import com.google.zxing.qrcode.QRCodeWriter;

import jakarta.mail.internet.MimeMessage;
import reservation.model.Payment;
import reservation.model.Reservation;
import reservation.model.Room;
import reservation.model.User;

@Service
public class EmailService {

    @Autowired
    private JavaMailSender mailSender;
    private static final String BASE_URL = "https://localhost:8443";
    
    private byte[] generateQRCode(String text) throws Exception {
        QRCodeWriter qrCodeWriter = new QRCodeWriter();
        Map<EncodeHintType, Object> hints = new HashMap<>();
        hints.put(EncodeHintType.CHARACTER_SET, "UTF-8");

        BitMatrix bitMatrix = qrCodeWriter.encode(text, BarcodeFormat.QR_CODE, 200, 200, hints);

        ByteArrayOutputStream pngOutputStream = new ByteArrayOutputStream();
        MatrixToImageWriter.writeToStream(bitMatrix, "PNG", pngOutputStream);
        return pngOutputStream.toByteArray();
    }

    public void ConfirmationEmail(Reservation reservation) {
        try {
    
            if (reservation.getCheckInToken() == null || reservation.getCheckInToken().isEmpty()) {
                reservation.setCheckInToken(UUID.randomUUID().toString());
            }

            String qrCodeText = "https://yourhotel.com/checkin?token=" + reservation.getCheckInToken();
            byte[] qrCodeImage = generateQRCode(qrCodeText);


            MimeMessage message = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message, true);

            helper.setTo(reservation.getEmail());
            helper.setSubject("Reservation Confirmation - " + reservation.getBookingReference());

            
            StringBuilder roomDetails = new StringBuilder();
            int totalRoomCount = 0;

            for (Map.Entry<Room, Integer> entry : reservation.getRoomQuantities().entrySet()) {
                Room room = entry.getKey();
                int quantity = entry.getValue();
                roomDetails.append(room.getRoomType()).append(": ").append(quantity).append(" room(s)\n");
                totalRoomCount += quantity;
            }

            
            String emailBody = String.format(
                "Dear %s %s,\n\n" +
                "Thank you for your reservation. Below are your details:\n\n" +
                "📌 **Booking Reference:** %s\n" +
                "**Hotel:** %s\n" +
                "**Check-In Date:** %s\n" +
                "**Check-Out Date:** %s\n" +
                "**Room Details:**\n%s\n" +
                "**Total Room Count:** %d\n" +
                "**Payment Method:** %s\n" +
                "**Total Price:** AED %.2f\n\n" +
                "**Check-In Instructions:**\n" +
                "Please present the attached QR code at check-in. Hotel staff will scan it for verification.\n\n" +
                "Best regards,\nThe InnovateStay Team",
                reservation.getFirstName(), reservation.getLastName(),
                reservation.getBookingReference(),
                reservation.getHotel().getName(),
                reservation.getCheckInDate(),
                reservation.getCheckOutDate(),
                roomDetails.toString(),
                totalRoomCount,
                reservation.getPayment().getPaymentMethod(),
                reservation.getPayment().getTotalPrice()
            );

            helper.setText(emailBody);

            helper.addAttachment("QR_Code.png", new ByteArrayResource(qrCodeImage));

           
            mailSender.send(message);
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Failed to send confirmation email: " + e.getMessage());
        }
    }
    public void sendVerificationEmail(User user) {
        SimpleMailMessage message = new SimpleMailMessage();
        message.setTo(user.getEmail());
        message.setSubject("Complete Your Registration");
        message.setText("Dear " + user.getUsername() + ",\n\n" +
                "Thank you for registering with our service. To complete your registration " +
                "and activate your account, please click on the link below:\n\n" +
                BASE_URL + "/users/verify?token=" + user.getVerificationToken() + "\n\n" +
                "This link will expire in 24 hours.\n\n" +
                "If you did not register for an account, please ignore this email.\n\n" +
                "Best regards,\n" +
                "The InnovateStay Team");
        
        mailSender.send(message);
    }

    public void sendPasswordResetEmail(User user) {
        SimpleMailMessage message = new SimpleMailMessage();
        message.setTo(user.getEmail());
        message.setSubject("Password Reset Request");
        message.setText("Dear " + user.getUsername() + ",\n\n" +
                "We received a request to reset your password. To proceed with the password reset, " +
                "please click on the link below:\n\n" +
                BASE_URL + "/reset-password.html?token=" + user.getResetToken() + "\n\n" +
                "This link will expire in 24 hours.\n\n" +
                "If you did not request a password reset, please ignore this email " +
                "or contact our support team if you believe this is an error.\n\n" +
                "Best regards,\n" +
                "The InnovateStay Team");
        
        mailSender.send(message);
    }
    public void sendPasswordChangeConfirmation(User user) {
        SimpleMailMessage message = new SimpleMailMessage();
        message.setTo(user.getEmail());
        message.setSubject("Password Changed Successfully");
        message.setText("Dear " + user.getUsername() + ",\n\n" +
                "Your password has been changed successfully.\n\n" +
                "If you did not make this change, please contact our support team immediately.\n\n" +
                "Best regards,\n" +
                "The InnovateStay Team");
        
        mailSender.send(message);
    }
    
    public void CancellationEmail(Reservation reservation) {
        SimpleMailMessage message = new SimpleMailMessage();
        message.setTo(reservation.getEmail());
        message.setSubject("Your Reservation Cancellation - " + reservation.getBookingReference());

        StringBuilder emailContent = new StringBuilder();
        emailContent.append("Dear ").append(reservation.getFirstName()).append(" ").append(reservation.getLastName()).append(",\n\n")
                    .append("We regret to inform you that your reservation at **")
                    .append(reservation.getHotel().getName()).append("** has been cancelled.\n\n")
                    .append("**Reservation Details:**\n")
                    .append("Booking Reference: ").append(reservation.getBookingReference()).append("\n")
                    .append("Check-in Date: ").append(reservation.getCheckInDate()).append("\n")
                    .append("Check-out Date: ").append(reservation.getCheckOutDate()).append("\n\n");

        
        if (reservation.getPayment() != null) {
            emailContent.append("**Refund Details:**\n")
                        .append("Payment Method: ").append(reservation.getPayment().getPaymentMethod()).append("\n")
                        .append("Total Amount: AED ").append(reservation.getPayment().getTotalPrice()).append("\n\n")
                        .append("Refunds are processed according to our cancellation policy. ")
                        .append("Please allow up to **5-7 business days** for the amount to reflect in your account.\n\n");
        }

        emailContent.append("Thank you for choosing us. We hope to serve you again in the future.\n\n")
                    .append("Best regards,\n")
                    .append("**The InnovateStay Team**");

        message.setText(emailContent.toString());
        mailSender.send(message);
    }


}