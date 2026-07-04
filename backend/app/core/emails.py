# backend/app/core/email.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any
from pathlib import Path

import jinja2

from .logging import logger
from app.config import settings

# backend/app/core/email.py

# Configuration du moteur de templates Jinja2
TEMPLATE_DIR = Path(__file__).parent / "email_templates"
template_loader = jinja2.FileSystemLoader(TEMPLATE_DIR)
template_env = jinja2.Environment(loader=template_loader, autoescape=True)


def _render_template(template_name: str, context: Dict[str, Any]) -> str:
    """Rendre un template HTML avec Jinja2"""
    try:
        template = template_env.get_template(template_name)
        return template.render(**context)
    except Exception as e:
        logger.error(f"Erreur lors du rendu du template {template_name}: {e}")
        raise


def _send_email(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: Optional[str] = None
) -> bool:
    """
    Envoyer un email avec les paramètres SMTP configurés
    """
    if not settings.SMTP_HOST or not settings.SMTP_FROM_EMAIL:
        logger.warning("SMTP non configuré, email non envoyé")
        return False
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_FROM_EMAIL
    msg["To"] = to_email
    
    # Contenu texte par défaut (version brute)
    if not text_content:
        # Supprimer les balises HTML pour la version texte
        import re
        text_content = re.sub(r'<[^>]+>', ' ', html_content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
    
    part_text = MIMEText(text_content, "plain", "utf-8")
    part_html = MIMEText(html_content, "html", "utf-8")
    
    msg.attach(part_text)
    msg.attach(part_html)
    
    try:
        if settings.SMTP_USE_TLS:
            server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT)
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT)
        
        if settings.SMTP_USER and settings.SMTP_PASSWORD:
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email envoyé à {to_email}")
        return True
    except Exception as e:
        logger.error(f"Échec d'envoi de l'email à {to_email}: {e}")
        return False


async def send_welcome_email(
    to_email: str,
    full_name: str,
    username: str,
    password: str,
    employee_id: Optional[str] = None,
    position: Optional[str] = None,
) -> bool:
    """
    Envoyer un email de bienvenue après création de compte
    """
    frontend_url = settings.FRONTEND_URL or "http://localhost:5173"
    login_url = f"{frontend_url}/auth/login"
    
    context = {
        "full_name": full_name,
        "email": to_email,
        "username": username,
        "password": password,
        "employee_id": employee_id,
        "position": position,
        "login_url": login_url,
        "frontend_url": frontend_url,
    }
    
    try:
        html_content = _render_template("welcome.html", context)
        subject = "Bienvenue sur Farm Manager - Votre compte a été créé"
        return _send_email(to_email, subject, html_content)
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'email de bienvenue: {e}")
        return False


async def send_user_updated_email(
    to_email: str,
    full_name: str,
    changes: Dict[str, Any],
    new_password: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> bool:
    """
    Envoyer un email de notification de mise à jour du compte
    """
    frontend_url = settings.FRONTEND_URL or "http://localhost:5173"
    login_url = f"{frontend_url}/auth/login"
    
    # Formater les changements pour l'affichage
    formatted_changes = {}
    field_labels = {
        "email": "📧 Email",
        "phone": "📱 Téléphone",
        "username": "👤 Nom d'utilisateur",
        "full_name": "📝 Nom complet",
        "is_active": "🔓 Statut",
        "employee_id": "📋 Matricule",
        "position": "💼 Poste",
        "department": "🏢 Département",
        "base_salary": "💰 Salaire de base",
        "employee_status": "📊 Statut employé",
        "employee_type": "📌 Type d'employé",
    }
    
    for key, value in changes.items():
        label = field_labels.get(key, key.replace("_", " ").title())
        formatted_changes[label] = value
    
    context = {
        "full_name": full_name,
        "changes": formatted_changes,
        "new_password": new_password,
        "password_updated": new_password is not None,
        "is_active": is_active,
        "login_url": login_url,
        "frontend_url": frontend_url,
    }
    
    try:
        html_content = _render_template("user_updated.html", context)
        subject = "Mise à jour de votre compte Farm Manager"
        return _send_email(to_email, subject, html_content)
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'email de mise à jour: {e}")
        return False


# === FONCTIONS EXISTANTES ===    
async def send_password_reset_email(
    to_email: str,
    token: str,
    reset_url: Optional[str] = None
) -> bool:
    """
    Envoyer un email de réinitialisation de mot de passe
    """
    if not settings.SMTP_HOST or not settings.SMTP_FROM_EMAIL:
        logger.warning("SMTP not configured, skipping email send")
        return False
    
    if not reset_url:
        frontend_url = settings.FRONTEND_URL or "http://localhost:5173"
        reset_url = f"{frontend_url}/auth/reset-password?token={token}"
    
    subject = "Réinitialisation de votre mot de passe - Farm Manager"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: white; padding: 30px; text-align: center; }}
            .content {{ padding: 30px; background: #f8fafc; }}
            .button {{ display: inline-block; padding: 12px 24px; background: #1e293b; color: white; text-decoration: none; border-radius: 8px; margin: 20px 0; }}
            .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #64748b; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🐄 Farm Manager</h1>
                <p>Réinitialisation du mot de passe</p>
            </div>
            <div class="content">
                <p>Bonjour,</p>
                <p>Vous avez demandé à réinitialiser votre mot de passe pour votre compte Farm Manager.</p>
                <p>Cliquez sur le bouton ci-dessous pour créer un nouveau mot de passe :</p>
                <p style="text-align: center;">
                    <a href="{reset_url}" class="button">Réinitialiser mon mot de passe</a>
                </p>
                <p>Ou copiez ce lien dans votre navigateur :</p>
                <p><code style="background: #e2e8f0; padding: 8px; display: block; word-break: break-all;">{reset_url}</code></p>
                <p><strong>⚠️ Ce lien est valable 1 heure seulement.</strong></p>
                <p>Si vous n'êtes pas à l'origine de cette demande, ignorez cet email.</p>
                <hr>
                <p style="font-size: 14px; color: #64748b;">L'équipe Farm Manager</p>
            </div>
            <div class="footer">
                <p>© 2024 Farm Manager - Gestion d'élevage intelligent</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_content = f"""
    Réinitialisation de votre mot de passe - Farm Manager
    
    Bonjour,
    
    Vous avez demandé à réinitialiser votre mot de passe pour votre compte Farm Manager.
    
    Cliquez sur le lien ci-dessous pour créer un nouveau mot de passe :
    {reset_url}
    
    ⚠️ Ce lien est valable 1 heure seulement.
    
    Si vous n'êtes pas à l'origine de cette demande, ignorez cet email.
    
    L'équipe Farm Manager
    """
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_FROM_EMAIL
    msg["To"] = to_email
    
    part_text = MIMEText(text_content, "plain", "utf-8")
    part_html = MIMEText(html_content, "html", "utf-8")
    
    msg.attach(part_text)
    msg.attach(part_html)
    
    try:
        if settings.SMTP_USE_TLS:
            server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT)
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT)
        
        if settings.SMTP_USER and settings.SMTP_PASSWORD:
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Password reset email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return False