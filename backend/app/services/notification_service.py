# backend/app/services/notification_service.py
"""
Service de notifications - Email, SMS, WebSocket
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ..config import settings
from ..database import get_db
from ..redis_client import redis_client

logger = logging.getLogger(__name__)


class NotificationService:
    """Service d'envoi de notifications"""
    
    def __init__(self):
        self._websocket_connections: Dict[int, Any] = {}  # user_id -> websocket
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None
    ) -> bool:
        """Envoyer un email"""
        if not settings.SMTP_HOST:
            logger.warning("SMTP not configured, skipping email")
            return False
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = settings.SMTP_FROM_EMAIL
            msg["To"] = to_email
            
            # Ajouter le texte
            part_text = MIMEText(body, "plain")
            msg.attach(part_text)
            
            # Ajouter le HTML si fourni
            if html_body:
                part_html = MIMEText(html_body, "html")
                msg.attach(part_html)
            
            # Envoyer
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                if settings.SMTP_USER and settings.SMTP_PASSWORD:
                    server.starttls()
                    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.send_message(msg)
            
            logger.info(f"Email sent to {to_email}: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    async def send_sms(self, phone_number: str, message: str) -> bool:
        """Envoyer un SMS"""
        if not settings.SMS_ENABLED:
            logger.warning("SMS not enabled, skipping")
            return False
        
        try:
            if settings.SMS_PROVIDER == "twilio":
                from twilio.rest import Client
                client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
                client.messages.create(
                    body=message,
                    from_=settings.TWILIO_PHONE_NUMBER,
                    to=phone_number
                )
                logger.info(f"SMS sent to {phone_number}")
                return True
            else:
                logger.warning(f"Unknown SMS provider: {settings.SMS_PROVIDER}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send SMS to {phone_number}: {e}")
            return False
    
    async def send_websocket(
        self,
        user_id: int,
        event_type: str,
        data: Dict[str, Any]
    ) -> bool:
        """Envoyer une notification WebSocket à un utilisateur"""
        connection = self._websocket_connections.get(user_id)
        if not connection:
            return False
        
        try:
            await connection.send_json({
                "type": event_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to send WebSocket to user {user_id}: {e}")
            # Supprimer la connexion défaillante
            if user_id in self._websocket_connections:
                del self._websocket_connections[user_id]
            return False
    
    async def broadcast_websocket(
        self,
        user_ids: List[int],
        event_type: str,
        data: Dict[str, Any]
    ) -> int:
        """Diffuser une notification WebSocket à plusieurs utilisateurs"""
        sent = 0
        for user_id in user_ids:
            if await self.send_websocket(user_id, event_type, data):
                sent += 1
        return sent
    
    async def send_alert(
        self,
        user_id: int,
        alert_type: str,
        title: str,
        message: str,
        severity: str = "info"
    ) -> None:
        """Envoyer une alerte via tous les canaux configurés"""
        # Envoyer via WebSocket (temps réel)
        await self.send_websocket(user_id, "alert", {
            "type": alert_type,
            "title": title,
            "message": message,
            "severity": severity
        })
        
        # Stocker dans Redis pour historique
        await redis_client.lpush(f"alerts:user:{user_id}", {
            "type": alert_type,
            "title": title,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        })
        await redis_client.ltrim(f"alerts:user:{user_id}", 0, 99)
        
        # Pour les alertes critiques, envoyer aussi par email/SMS
        if severity in ["critical", "high"]:
            # Récupérer l'email de l'utilisateur
            from .user_service import user_service
            async for db in get_db():
                user = await user_service.get_user(db, user_id)
                if user and user.email:
                    await self.send_email(
                        user.email,
                        f"[{severity.upper()}] {title}",
                        message
                    )
    
    def register_websocket(self, user_id: int, websocket) -> None:
        """Enregistrer une connexion WebSocket"""
        self._websocket_connections[user_id] = websocket
    
    def unregister_websocket(self, user_id: int) -> None:
        """Supprimer une connexion WebSocket"""
        if user_id in self._websocket_connections:
            del self._websocket_connections[user_id]
    
    async def send_vaccination_reminder(
        self,
        user_id: int,
        animal_identification: str,
        maladie: str,
        due_date: datetime
    ) -> None:
        """Envoyer un rappel de vaccination"""
        days_until = (due_date - datetime.now()).days
        
        await self.send_alert(
            user_id=user_id,
            alert_type="vaccination_reminder",
            title="Rappel de vaccination",
            message=f"Vaccination {maladie} pour {animal_identification} prévue dans {days_until} jours",
            severity="high" if days_until <= 3 else "medium"
        )
    
    async def send_mortality_alert(
        self,
        user_id: int,
        espece: str,
        mortality_rate: float,
        enclos_name: str
    ) -> None:
        """Envoyer une alerte de mortalité élevée"""
        await self.send_alert(
            user_id=user_id,
            alert_type="mortality_alert",
            title="Alerte mortalité",
            message=f"Taux de mortalité élevé pour {espece} dans {enclos_name}: {mortality_rate}%",
            severity="critical" if mortality_rate > 10 else "high"
        )
    
    async def send_water_quality_alert(
        self,
        user_id: int,
        parameter: str,
        value: float,
        threshold: float,
        enclos_name: str
    ) -> None:
        """Envoyer une alerte qualité d'eau"""
        await self.send_alert(
            user_id=user_id,
            alert_type="water_quality_alert",
            title="Alerte qualité d'eau",
            message=f"{parameter} critique dans {enclos_name}: {value} (seuil: {threshold})",
            severity="critical"
        )


notification_service = NotificationService()