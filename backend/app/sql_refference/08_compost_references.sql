-- ================================================================================
-- FICHIER 8: 08_compost_references.sql
-- Table: compost_references (paramètres de compostage)
-- ================================================================================

-- CREATE TABLE IF NOT EXISTS compost_references (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
--     type_matiere VARCHAR(50) NOT NULL,
--     parametre VARCHAR(50) NOT NULL,
--     valeur_min FLOAT,
--     valeur_max FLOAT,
--     unite VARCHAR(20),
--     niveau_alerte VARCHAR(20),
--     source VARCHAR(255),
--     created_at DATETIME,
--     updated_at DATETIME
-- );

INSERT INTO compost_references (type_matiere, parametre, valeur_min, valeur_max, unite, niveau_alerte, source) VALUES
('mixte', 'cn_rapport', 25, 30, 'ratio', NULL, 'standard'),
('mixte', 'temperature', 55, 65, '°C', NULL, 'standard'),
('mixte', 'humidite', 50, 60, '%', NULL, 'standard'),
('mixte', 'ph', 6.5, 8.0, NULL, NULL, 'standard'),
('mixte', 'temperature', 45, NULL, '°C', 'warning', 'standard'),
('mixte', 'temperature', NULL, 70, '°C', 'warning', 'standard'),
('mixte', 'humidite', 40, NULL, '%', 'warning', 'standard'),
('mixte', 'humidite', NULL, 70, '%', 'warning', 'standard'),
('mixte', 'ph', 5.5, NULL, NULL, 'warning', 'standard'),
('mixte', 'ph', NULL, 8.5, NULL, 'warning', 'standard'),
('dechets_verts', 'cn_rapport', 30, 50, 'ratio', NULL, 'standard'),
('dechets_verts', 'duree_active', 3, 6, 'semaines', NULL, 'standard'),
('fumier', 'cn_rapport', 15, 25, 'ratio', NULL, 'standard'),
('fumier', 'duree_active', 4, 8, 'semaines', NULL, 'standard');
