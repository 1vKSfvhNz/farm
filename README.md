# 🚜 Farm Gestion - Plateforme d'Elevage

> **Version 1.0.0** - Gestion intelligente d'exploitation agricole

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![SvelteKit](https://img.shields.io/badge/SvelteKit-2.0+-orange.svg)](https://kit.svelte.dev)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-24+-blue.svg)](https://docker.com)

================================================================================

## 📖 À propos

Farm Gestion est une plateforme complète de gestion d'élevage conçue pour les exploitants modernes. Elle centralise l'ensemble des opérations :

- 🐄 Gestion du cheptel (bovins, ovins, caprins)
- 💉 Santé animale (vaccinations, suivi vétérinaire)
- 📊 Bien-être animal (BEA)
- 💰 Comptabilité et rentabilité
- 🌡️ Qualité de l'eau en temps réel
- 🗑️ Compostage et gestion des déchets
- 🤖 Prédictions IA
- 📹 Vidéosurveillance
- 📈 Tableau de bord personnalisé

================================================================================

## 🏗️ Architecture

+-------------------------------------------------------------+
|                          Frontend                           |
|                    (SvelteKit + Tailwind)                   |
|                          Port 5173                         |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                         API Gateway                         |
|                     (FastAPI + Uvicorn)                    |
|                          Port 8000                         |
+-------------------------------------------------------------+
                              |
          +-------------------+-------------------+
          v                   v                   v
+-----------------+ +-----------------+ +-----------------+
|   PostgreSQL    | |      Redis      | |   File Storage  |
|   (Base de      | |   (Cache &      | |   (Uploads &    |
|    données)     | |    Sessions)    | |    Exports)     |
|    Port 5432    | |    Port 6379    | |                 |
+-----------------+ +-----------------+ +-----------------+

================================================================================

## 🛠️ Technologies

| Composant       | Technologie     | Version |
|-----------------|-----------------|---------|
| Backend         | FastAPI         | 0.100+  |
| ORM             | SQLAlchemy (async) | 2.0+  |
| Migrations      | Alembic         | 1.18+   |
| Validation      | Pydantic        | 2.0+    |
| Auth            | JWT (PyJWT)     | 2.13+   |
| Database        | PostgreSQL      | 15+     |
| Cache           | Redis           | 7+      |
| Frontend        | SvelteKit       | 2.0+    |
| Styling         | TailwindCSS     | 3.0+    |
| Charts          | Chart.js        | 4.0+    |
| Container       | Docker          | 24+     |
| Orchestration   | Kubernetes      | 1.28+   |
| Monitoring      | Prometheus      | -       |

================================================================================

## 🚀 Démarrage rapide avec Docker

### Prérequis

# Installer Docker et Docker Compose
docker --version
docker-compose --version

# Cloner le projet
git clone https://github.com/1vKSfvhNz/farm_manager.git
cd farm_manager

# Copier les fichiers d'environnement
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# Éditer les variables (au minimum la base de données)
nano backend/.env

### Avec Docker

# Construire et démarrer tous les services
docker-compose up -d --build

# Voir les logs
docker-compose logs -f

# Arrêter
docker-compose down

# Arrêter et supprimer les volumes (⚠️ supprime les données)
docker-compose down -v

### Sans Docker

# Backend
cd backend

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos configurations

# Appliquer les migrations
alembic upgrade head

# Démarrer le serveur
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd ../frontend

# Installer les dépendances
npm install

# Démarrer en développement
npm run dev -- --open

# Construire pour production
npm run build
npm run preview

================================================================================

## Configuration .env (backend)

# Base de données
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=farm_manager

# JWT
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production-minimum-32-characters
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# Redis (cache)
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
REDIS_ENABLED=true

# Météo (OpenWeatherMap)
WEATHER_API_ENABLED=true
WEATHER_API_PROVIDER=openweathermap
WEATHER_API_KEY=your-openweather-api-key
WEATHER_LATITUDE=48.8566
WEATHER_LONGITUDE=2.3522

# Blockchain (optionnel - premium)
BLOCKCHAIN_ENABLED=false
BLOCKCHAIN_NETWORK=sepolia

# Email (reset password)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM_EMAIL=noreply@farm-manager.com

# Uploads
UPLOAD_DIR=./uploads
UPLOAD_MAX_SIZE=5242880

# CORS
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
ALLOWED_HOSTS=["localhost","127.0.0.1"]

# Application
APP_NAME=Farm Manager API
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
API_PREFIX=/api/v1

================================================================================

## Configuration .env (frontend)

PUBLIC_API_URL=http://localhost:8000/api/v1

================================================================================

## 📦 Commandes utiles

### Backend

# Démarrer avec Uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Créer une migration
alembic revision --autogenerate -m "Description"

# Appliquer les migrations
alembic upgrade head

# Rollback
alembic downgrade -1

# Générer un token admin
python scripts/create_admin.py

# Exporter la base de données
pg_dump -U postgres -h localhost farm_manager > backup.sql

### Frontend

# Développement
npm run dev

# Construction
npm run build

# Preview de production
npm run preview

# Analyse du bundle
npm run analyze

### Docker

# Démarrer tous les services
docker-compose up -d

# Vérifier l'état
docker-compose ps

# Voir les logs
docker-compose logs -f backend

# Arrêter
docker-compose down

# Arrêter et supprimer les volumes (⚠️ supprime les données)
docker-compose down -v

================================================================================

## 🧪 Tests

# Backend
cd backend
pytest

# Frontend
cd frontend
npm run test

================================================================================

## 📚 Documentation API

Une fois le serveur démarré :

- Swagger UI : http://localhost:8000/docs
- ReDoc : http://localhost:8000/redoc

================================================================================

## 🔐 Authentification

L'API utilise JWT (JSON Web Tokens) pour l'authentification.

### Endpoints d'authentification

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | /api/v1/auth/login | Connexion utilisateur |
| POST | /api/v1/auth/register | Inscription utilisateur |
| POST | /api/v1/auth/refresh | Rafraîchir le token |
| POST | /api/v1/auth/logout | Déconnexion |

### Utilisation

# 1. Obtenir un token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "Ozias", "password": "12345678"}'

# 2. Utiliser le token
curl -X GET http://localhost:8000/api/v1/users/me \
  -H "Authorization: Bearer <votre_token>"

================================================================================

## 👤 Utilisateur par défaut

L'application crée automatiquement un utilisateur admin au premier démarrage :

| Champ | Valeur |
|-------|--------|
| Nom d'utilisateur | Ozias |
| Téléphone | +22661506121 |
| Mot de passe | 12345678 |
| Rôles | Tous les rôles |

================================================================================

## 🗄️ Structure du projet

farm_manager/
+-- backend/
|   +-- app/
|   |   +-- api/
|   |   |   +-- v1/
|   |   |       +-- __init__.py
|   |   |       +-- auth.py
|   |   |       +-- users.py
|   |   |       +-- enclos.py
|   |   |       +-- bovins.py
|   |   |       +-- ovins.py
|   |   |       +-- caprins.py
|   |   |       +-- avicoles.py
|   |   |       +-- piscicoles.py
|   |   |       +-- entomoculture.py
|   |   |       +-- apiary.py
|   |   |       +-- vaccinations.py
|   |   |       +-- accounting.py
|   |   |       +-- dashboard.py
|   |   |       +-- predictions.py
|   |   |       +-- alerts.py
|   |   |       +-- exports.py
|   |   |       +-- water_quality.py
|   |   |       +-- video.py
|   |   |       +-- weather.py
|   |   |       +-- bea.py
|   |   |       +-- blockchain.py
|   |   |       +-- odoni.py
|   |   |       +-- compost.py
|   |   |       +-- pesees.py
|   |   |       +-- experimental.py
|   |   +-- core/
|   |   |   +-- __init__.py
|   |   |   +-- security.py
|   |   |   +-- logging.py
|   |   |   +-- exceptions.py
|   |   +-- models/
|   |   |   +-- __init__.py
|   |   |   +-- base.py
|   |   |   +-- user.py
|   |   |   +-- animal.py
|   |   |   +-- enclos.py
|   |   |   +-- vaccination.py
|   |   |   +-- accounting.py
|   |   |   +-- water_quality.py
|   |   |   +-- video.py
|   |   |   +-- weather.py
|   |   |   +-- bea.py
|   |   |   +-- blockchain.py
|   |   |   +-- experimental.py
|   |   +-- schemas/
|   |   |   +-- __init__.py
|   |   |   +-- user.py
|   |   |   +-- animal.py
|   |   |   +-- enclos.py
|   |   |   +-- vaccination.py
|   |   |   +-- accounting.py
|   |   +-- middleware/
|   |   |   +-- __init__.py
|   |   |   +-- auth.py
|   |   |   +-- logging.py
|   |   |   +-- rate_limit.py
|   |   +-- config.py
|   |   +-- database.py
|   |   +-- redis_client.py
|   |   +-- lifespan.py
|   |   +-- main.py
|   +-- migrations/
|   +-- tests/
|   +-- requirements.txt
|   +-- .env.example
|   +-- Dockerfile
+-- frontend/
|   +-- src/
|   |   +-- lib/
|   |   +-- routes/
|   |   +-- app.html
|   |   +-- app.css
|   +-- static/
|   +-- package.json
|   +-- .env.example
|   +-- Dockerfile
+-- docker-compose.yml
+-- README.md
+-- .gitignore

================================================================================

## 📝 Notes de version

### v1.0.0 (2024)

- Version initiale
- Gestion complète du cheptel
- Système d'authentification JWT
- Tableau de bord personnalisé
- Prédictions IA
- API REST complète
- Documentation Swagger/ReDoc

================================================================================

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez suivre ces étapes :

1. Fork le projet
2. Créer une branche (git checkout -b feature/AmazingFeature)
3. Commit les changements (git commit -m 'Add some AmazingFeature')
4. Push sur la branche (git push origin feature/AmazingFeature)
5. Ouvrir une Pull Request

================================================================================

## 📧 Contact

- Projet: https://github.com/1vKSfvhNz/farm
- Email: noomwende4@gmail.com.bf
- Téléphone: +22661506121

================================================================================

## 🙏 Remerciements

- FastAPI pour le framework backend
- SvelteKit pour le framework frontend
- PostgreSQL pour la base de données
- Redis pour le caching
- Tous les contributeurs open-source

================================================================================

Fait avec ❤️ par l'équipe Farm Manager