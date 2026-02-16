from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def _log_alert(message: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def send_alert(message: str, log_path: Path = Path("logs/alerts.log")) -> None:
    print(message)
    _log_alert(message, log_path)

    host = os.getenv("SMTP_HOST")
    port = os.getenv("SMTP_PORT")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    recipient = os.getenv("ALERT_TO")

    if not all([host, port, user, password, recipient]):
        return

    msg = EmailMessage()
    msg["Subject"] = "TradingLab Alert"
    msg["From"] = user
    msg["To"] = recipient
    msg.set_content(message)

    with smtplib.SMTP(host, int(port)) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
