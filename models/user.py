from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, func
from sqlalchemy.orm import Session, relationship

from utils.security import hash_passw, verify_passw
from models import Base, add_object

# ✅ Modèle User
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), nullable=False)
    email = Column(String(64), unique=True, nullable=False, index=True)
    phone = Column(String(20), unique=True, nullable=False)
    password = Column(String(128), nullable=False)
    
    is_active = Column(Boolean, default=True, nullable=False)
    notifications = Column(Boolean, default=True, nullable=False)
    role = Column(String(16), default='admin', nullable=False)  # 'admin', 'user'
    lang = Column(String(2), default='fr', nullable=True)
    devices = relationship("UserDevice", back_populates="user")
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))  # Toujours stocké en UTC
    last_login = Column(DateTime, nullable=True)  # Dernière connexion de l'utilisateur (peut être null)

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email}, role={self.role})>"

    # Sauvegarde de l'utilisateur en base
    def save_user(self, db: Session):
        """Ajoute l'utilisateur en base de données après hachage du mot de passe."""
        self.password = hash_passw(self.password)
        add_object(db, self)

    # Mise à jour du mot de passe
    def update_password(self, new_password: str, db: Session):
        """Met à jour le mot de passe après l'avoir haché."""
        self.password = hash_passw(new_password)
        db.commit()

    # Vérifier le mot de passe
    def verify_password(self, plain_password: str) -> bool:
        return verify_passw(plain_password, self.password)

class UserDevice(Base):
    __tablename__ = "user_devices"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    app_version = Column(String, nullable=True)     # version de l'app installée
    device_name = Column(String, nullable=True)     # nom du périphérique (ex: iPhone 12)
    device_token = Column(String, nullable=False)    # FCM/APNs token
    platform = Column(String, nullable=False)       # 'ios' ou 'android'
    last_used_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relation avec l'utilisateur
    user = relationship("User", back_populates="devices")

class UserConnection(Base):
    __tablename__ = "user_connections"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    last_connected = Column(DateTime)
    last_disconnected = Column(DateTime, nullable=True)
    connection_data = Column(String)  # JSON serialized connection metadata
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)