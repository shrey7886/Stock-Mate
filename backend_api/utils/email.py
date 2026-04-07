import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from backend_api.core.config import settings


def send_password_reset_email(to_email: str, token: str) -> None:
    if not settings.smtp_username or not settings.smtp_password:
        print(f"Skipping email to {to_email}. SMTP not configured. (Use App Passwords for Gmail)")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Foleo - Password Reset Request"
    msg["From"] = settings.smtp_username
    msg["To"] = to_email

    reset_link = f"{settings.frontend_url}/reset-password?token={token}"

    text = f"Hello,\n\nWe received a request to reset your password. Click the link below to reset it:\n{reset_link}\n\nThis link expires in 15 minutes. If you did not request this, please ignore this email."
    
    html = f"""\
    <html>
      <body style="font-family: 'Inter', Arial, sans-serif; color: #333; line-height: 1.6; background-color: #f9fbfd; padding: 40px;">
        <div style="max-width: 500px; margin: 0 auto; background: white; padding: 40px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); text-align: center;">
            <h2 style="color: #00BA7C; font-size: 24px; margin-bottom: 8px;">Foleo</h2>
            <h3 style="color: #1a1a1a; margin-top: 0;">Reset Your Password</h3>
            <p style="color: #666; font-size: 15px; margin-bottom: 30px;">
              We received a request to reset the password for the Foleo account associated with this email address.
            </p>
            <a href="{reset_link}" style="background-color: #00BA7C; color: white; padding: 14px 28px; text-decoration: none; border-radius: 50px; font-weight: 600; display: inline-block; font-size: 15px; box-shadow: 0 4px 12px rgba(0, 186, 124, 0.2);">Reset My Password</a>
            <p style="color: #999; font-size: 13px; margin-top: 30px;">
              This link is valid for exactly <strong>15 minutes</strong>. If you did not request a password reset, you can safely ignore this email.
            </p>
            <p style="word-break: break-all; color: #aaa; font-size: 11px; margin-top: 40px; border-top: 1px solid #eee; padding-top: 20px;">
              If the button doesn't work, copy and paste this link into your browser:<br/>
              <a href="{reset_link}" style="color: #00BA7C;">{reset_link}</a>
            </p>
        </div>
      </body>
    </html>
    """

    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")

    msg.attach(part1)
    msg.attach(part2)

    try:
        with smtplib.SMTP(settings.smtp_server, settings.smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(settings.smtp_username, settings.smtp_password)
            server.send_message(msg)
            print(f"Password reset email sent to {to_email}")
    except Exception as e:
        print(f"Error sending email to {to_email}: {e}")
