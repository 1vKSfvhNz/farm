-- ================================================================================
-- FICHIER 1: 01_references_croissance.sql
-- Table: references_croissance
-- ================================================================================

-- BOVINS
INSERT INTO references_croissance (espece, race, age_jours, poids_min, poids_moyen, poids_max, source, is_active) VALUES
('bovin', 'Zebu local', 0, 20, 25, 30, 'standard', true),
('bovin', 'Zebu local', 30, 40, 50, 60, 'standard', true),
('bovin', 'Zebu local', 90, 70, 85, 100, 'standard', true),
('bovin', 'Zebu local', 180, 130, 160, 190, 'standard', true),
('bovin', 'Zebu local', 365, 220, 260, 300, 'standard', true),
('bovin', 'Metis Zebu', 0, 25, 30, 35, 'standard', true),
('bovin', 'Metis Zebu', 30, 50, 60, 70, 'standard', true),
('bovin', 'Metis Zebu', 90, 90, 110, 130, 'standard', true),
('bovin', 'Metis Zebu', 180, 170, 210, 250, 'standard', true),
('bovin', 'Metis Zebu', 365, 290, 340, 390, 'standard', true);

-- OVINS
INSERT INTO references_croissance (espece, race, age_jours, poids_min, poids_moyen, poids_max, source, is_active) VALUES
('ovin', 'Oudah', 0, 2.0, 2.8, 3.5, 'standard', true),
('ovin', 'Oudah', 30, 6, 8, 10, 'standard', true),
('ovin', 'Oudah', 90, 15, 20, 25, 'standard', true),
('ovin', 'Oudah', 180, 25, 35, 45, 'standard', true),
('ovin', 'Mossi', 0, 2.0, 2.7, 3.4, 'standard', true),
('ovin', 'Mossi', 30, 5.5, 7.5, 9.5, 'standard', true),
('ovin', 'Mossi', 90, 14, 19, 24, 'standard', true),
('ovin', 'Mossi', 180, 24, 33, 42, 'standard', true);

-- CAPRINS
INSERT INTO references_croissance (espece, race, age_jours, poids_min, poids_moyen, poids_max, source, is_active) VALUES
('caprin', 'Mossi', 0, 1.8, 2.5, 3.2, 'standard', true),
('caprin', 'Mossi', 30, 4, 6, 8, 'standard', true),
('caprin', 'Mossi', 90, 12, 17, 22, 'standard', true),
('caprin', 'Mossi', 180, 20, 30, 40, 'standard', true),
('caprin', 'Sahelienne', 0, 2.0, 2.8, 3.6, 'standard', true),
('caprin', 'Sahelienne', 30, 5, 7, 9, 'standard', true),
('caprin', 'Sahelienne', 90, 14, 19, 24, 'standard', true),
('caprin', 'Sahelienne', 180, 24, 34, 44, 'standard', true);

-- AVICOLES
INSERT INTO references_croissance (espece, race, age_jours, poids_min, poids_moyen, poids_max, source, is_active) VALUES
('avicole', 'Poulet local', 0, 0.025, 0.030, 0.035, 'standard', true),
('avicole', 'Poulet local', 30, 0.120, 0.160, 0.200, 'standard', true),
('avicole', 'Poulet local', 60, 0.350, 0.450, 0.550, 'standard', true),
('avicole', 'Poulet local', 90, 0.600, 0.800, 1.000, 'standard', true),
('avicole', 'Poulet local', 120, 0.900, 1.200, 1.500, 'standard', true),
('avicole', 'Poulet amelioré ponte', 0, 0.030, 0.035, 0.040, 'standard', true),
('avicole', 'Poulet amelioré ponte', 30, 0.150, 0.200, 0.250, 'standard', true),
('avicole', 'Poulet amelioré ponte', 60, 0.450, 0.550, 0.650, 'standard', true),
('avicole', 'Poulet amelioré ponte', 90, 0.800, 1.000, 1.200, 'standard', true),
('avicole', 'Poulet amelioré ponte', 120, 1.200, 1.500, 1.800, 'standard', true);

-- PISCICOLES
INSERT INTO references_croissance (espece, race, age_jours, poids_min, poids_moyen, poids_max, source, is_active) VALUES
('piscicole', 'Tilapia', 0, 0.00002, 0.00003, 0.00004, 'standard', true),
('piscicole', 'Tilapia', 30, 0.005, 0.008, 0.012, 'standard', true),
('piscicole', 'Tilapia', 60, 0.025, 0.040, 0.060, 'standard', true),
('piscicole', 'Tilapia', 90, 0.060, 0.100, 0.150, 'standard', true),
('piscicole', 'Tilapia', 120, 0.120, 0.200, 0.300, 'standard', true),
('piscicole', 'Tilapia', 180, 0.250, 0.400, 0.600, 'standard', true),
('piscicole', 'Carpe', 0, 0.00003, 0.00004, 0.00005, 'standard', true),
('piscicole', 'Carpe', 30, 0.006, 0.010, 0.015, 'standard', true),
('piscicole', 'Carpe', 60, 0.030, 0.050, 0.070, 'standard', true),
('piscicole', 'Carpe', 90, 0.080, 0.130, 0.180, 'standard', true),
('piscicole', 'Carpe', 120, 0.150, 0.250, 0.350, 'standard', true),
('piscicole', 'Carpe', 180, 0.300, 0.500, 0.700, 'standard', true),
('piscicole', 'Silure', 0, 0.00002, 0.00003, 0.00004, 'standard', true),
('piscicole', 'Silure', 30, 0.003, 0.005, 0.008, 'standard', true),
('piscicole', 'Silure', 60, 0.015, 0.025, 0.040, 'standard', true),
('piscicole', 'Silure', 90, 0.040, 0.070, 0.110, 'standard', true),
('piscicole', 'Silure', 120, 0.080, 0.150, 0.230, 'standard', true),
('piscicole', 'Silure', 180, 0.180, 0.350, 0.550, 'standard', true);
