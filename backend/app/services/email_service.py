import asyncio
from email.message import EmailMessage
from smtplib import SMTP

from app.config import get_settings


class EmailDeliveryError(RuntimeError):
    pass


def is_email_delivery_configured() -> bool:
    settings = get_settings()
    return bool(settings.smtp_host and settings.smtp_from_email)


def _send_email(to_address: str, subject: str, body: str) -> None:
    settings = get_settings()
    if not settings.smtp_host or not settings.smtp_from_email:
        raise EmailDeliveryError("SMTP email delivery is not configured")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.smtp_from_email
    message["To"] = to_address
    message.set_content(body)

    try:
        with SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
            if settings.smtp_use_tls:
                smtp.starttls()
            if settings.smtp_username:
                smtp.login(settings.smtp_username, settings.smtp_password or "")
            smtp.send_message(message)
    except Exception as exc:
        raise EmailDeliveryError("Unable to send authentication email") from exc


async def send_auth_email(to_address: str, subject: str, body: str) -> None:
    await asyncio.to_thread(_send_email, to_address, subject, body)
