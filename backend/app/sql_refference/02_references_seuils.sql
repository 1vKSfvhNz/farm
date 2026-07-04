-- ================================================================================
-- FICHIER 2: 02_references_seuils.sql
-- Table: references_seuils
-- ================================================================================

-- QUALITE EAU - TILAPIA
INSERT INTO references_seuils (espece, parametre, unite, seuil_min, seuil_max, seuil_optimal_min, seuil_optimal_max, niveau_alerte, source) VALUES
('tilapia', 'oxygene_dissous', 'mg/L', 3.0, NULL, 5.0, 8.0, 'critical', 'standard'),
('tilapia', 'ph', NULL, 6.5, 9.0, 7.0, 8.0, 'critical', 'standard'),
('tilapia', 'temperature', '°C', 18, 36, 28, 32, 'critical', 'standard'),
('tilapia', 'ammoniac', 'mg/L', NULL, 0.1, NULL, NULL, 'critical', 'standard'),
('tilapia', 'nitrites', 'mg/L', NULL, 1.0, NULL, NULL, 'warning', 'standard'),
('tilapia', 'nitrates', 'mg/L', NULL, 100, NULL, NULL, 'warning', 'standard');

-- QUALITE EAU - CARPE
INSERT INTO references_seuils (espece, parametre, unite, seuil_min, seuil_max, seuil_optimal_min, seuil_optimal_max, niveau_alerte, source) VALUES
('carpe', 'oxygene_dissous', 'mg/L', 4.0, NULL, 5.0, 7.0, 'critical', 'standard'),
('carpe', 'ph', NULL, 6.0, 8.5, 6.5, 8.0, 'critical', 'standard'),
('carpe', 'temperature', '°C', 15, 32, 25, 30, 'critical', 'standard'),
('carpe', 'ammoniac', 'mg/L', NULL, 0.05, NULL, NULL, 'critical', 'standard'),
('carpe', 'nitrites', 'mg/L', NULL, 0.5, NULL, NULL, 'warning', 'standard'),
('carpe', 'nitrates', 'mg/L', NULL, 80, NULL, NULL, 'warning', 'standard');

-- QUALITE EAU - SILURE
INSERT INTO references_seuils (espece, parametre, unite, seuil_min, seuil_max, seuil_optimal_min, seuil_optimal_max, niveau_alerte, source) VALUES
('silure', 'oxygene_dissous', 'mg/L', 4.0, NULL, 5.0, 7.0, 'critical', 'standard'), 
('silure', 'ph', NULL, 6.0, 8.5, 6.5, 8.0, 'critical', 'standard'),
('silure', 'temperature', '°C', 20, 35, 25, 30, 'critical', 'standard'),
('silure', 'ammoniac', 'mg/L', NULL, 0.05, NULL, NULL, 'critical', 'standard'),
('silure', 'nitrites', 'mg/L', NULL, 0.5, NULL, NULL, 'warning', 'standard'),
('silure', 'nitrates', 'mg/L', NULL, 80, NULL, NULL, 'warning', 'standard');

-- MORTALITE PAR ESPECE
INSERT INTO references_seuils (espece, parametre, unite, seuil_min, seuil_max, niveau_alerte, source) VALUES
('bovin', 'mortalite', '%', 2.0, 5.0, 'warning', 'standard'),
('bovin', 'mortalite', '%', 5.0, NULL, 'critical', 'standard'),
('ovin', 'mortalite', '%', 3.0, 6.0, 'warning', 'standard'),
('ovin', 'mortalite', '%', 6.0, NULL, 'critical', 'standard'),
('caprin', 'mortalite', '%', 3.0, 6.0, 'warning', 'standard'),
('caprin', 'mortalite', '%', 6.0, NULL, 'critical', 'standard'),
('avicole', 'mortalite', '%', 5.0, 10.0, 'warning', 'standard'),
('avicole', 'mortalite', '%', 10.0, NULL, 'critical', 'standard'),
('piscicole', 'mortalite', '%', 1.0, 3.0, 'warning', 'standard'),
('piscicole', 'mortalite', '%', 3.0, NULL, 'critical', 'standard');

-- TEMPERATURES CRITIQUES
INSERT INTO references_seuils (espece, parametre, unite, seuil_min, seuil_max, niveau_alerte, source) VALUES
('bovin', 'temperature_critique', '°C', 5, 35, 'critical', 'standard'),
('ovin', 'temperature_critique', '°C', 5, 38, 'critical', 'standard'),
('caprin', 'temperature_critique', '°C', 5, 38, 'critical', 'standard'),
('avicole', 'temperature_critique', '°C', 15, 35, 'critical', 'standard');

-- DENSITE MAXIMALE
INSERT INTO references_seuils (espece, parametre, unite, seuil_min, seuil_max, niveau_alerte, source) VALUES
('bovin', 'densite_max', 'm²/animal', NULL, 10, 'warning', 'standard'),
('ovin', 'densite_max', 'm²/animal', NULL, 5, 'warning', 'standard'),
('caprin', 'densite_max', 'm²/animal', NULL, 4, 'warning', 'standard'),
('avicole', 'densite_max', 'animaux/m²', NULL, 15, 'warning', 'standard'),
('piscicole', 'densite_max', 'kg/m³', NULL, 30, 'warning', 'standard');

-- TAUX DE CONVERSION ALIMENTAIRE (FCR)
INSERT INTO references_seuils (espece, parametre, unite, seuil_min, seuil_max, niveau_alerte, source) VALUES
('bovin', 'fcr_viande', NULL, 7, 9, 'warning', 'standard'),
('bovin', 'fcr_viande', NULL, 9, NULL, 'critical', 'standard'),
('bovin', 'fcr_lait', 'kg aliment/L lait', 1.2, 1.5, 'warning', 'standard'),
('bovin', 'fcr_lait', 'kg aliment/L lait', 1.5, NULL, 'critical', 'standard'),
('ovin', 'fcr', NULL, 6, 8, 'warning', 'standard'),
('ovin', 'fcr', NULL, 8, NULL, 'critical', 'standard'),
('caprin', 'fcr', NULL, 6, 8, 'warning', 'standard'),
('caprin', 'fcr', NULL, 8, NULL, 'critical', 'standard'),
('avicole', 'fcr_viande', NULL, 2.0, 2.5, 'warning', 'standard'),
('avicole', 'fcr_viande', NULL, 2.5, NULL, 'critical', 'standard'),
('avicole', 'fcr_ponte', 'kg/dozaine', 2.0, 2.4, 'warning', 'standard'),
('tilapia', 'fcr', NULL, 1.6, 2.0, 'warning', 'standard'),
('tilapia', 'fcr', NULL, 2.0, NULL, 'critical', 'standard'),
('carpe', 'fcr', NULL, 1.7, 2.1, 'warning', 'standard'),
('carpe', 'fcr', NULL, 2.1, NULL, 'critical', 'standard'),
('silure', 'fcr', NULL, 1.8, 2.2, 'warning', 'standard'),
('silure', 'fcr', NULL, 2.2, NULL, 'critical', 'standard');
