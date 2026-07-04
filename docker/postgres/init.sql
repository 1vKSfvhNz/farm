-- Initialisation de la base de données
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Créer l'utilisateur si nécessaire
-- CREATE USER farm_user WITH PASSWORD 'password';
-- GRANT ALL PRIVILEGES ON DATABASE farm_manager TO farm_user;