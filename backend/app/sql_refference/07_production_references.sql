-- ================================================================================
-- FICHIER 7: 07_production_references.sql
-- Tables additionnelles pour les productions de référence
-- ================================================================================

-- Table: productions_references (à créer si nécessaire)
-- CREATE TABLE IF NOT EXISTS productions_references (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
--     espece VARCHAR(50) NOT NULL,
--     race VARCHAR(100),
--     parametre VARCHAR(50) NOT NULL,
--     valeur_min FLOAT,
--     valeur_moyenne FLOAT,
--     valeur_max FLOAT,
--     unite VARCHAR(20),
--     source VARCHAR(255),
--     created_at DATETIME,
--     updated_at DATETIME
-- );

INSERT INTO productions_references (espece, race, parametre, valeur_min, valeur_moyenne, valeur_max, unite, source) VALUES
('bovin', 'Zebu local', 'lait_quotidien', 1.5, 2.0, 2.5, 'litres/jour', 'standard'),
('bovin', 'Zebu local', 'lait_pic', 2.0, 3.0, 4.0, 'litres/jour', 'standard'),
('bovin', 'Metis Zebu', 'lait_quotidien', 4.0, 5.5, 7.0, 'litres/jour', 'standard'),
('bovin', 'Metis Zebu', 'lait_pic', 6.0, 8.0, 10.0, 'litres/jour', 'standard'),
('avicole', 'Poulet local', 'oeufs_an', 40, 50, 60, 'oeufs/an', 'standard'),
('avicole', 'Poulet local', 'poids_oeuf', 35, 38, 40, 'grammes', 'standard'),
('avicole', 'Poulet amelioré ponte', 'oeufs_an', 180, 200, 220, 'oeufs/an', 'standard'),
('avicole', 'Poulet amelioré ponte', 'poids_oeuf', 50, 53, 55, 'grammes', 'standard'),
('entomoculture', 'Hermetia illucens', 'production_jour', 5, 7.5, 10, '% biomasse/jour', 'standard'),
('entomoculture', 'Hermetia illucens', 'cycle_jours', 14, 18, 21, 'jours', 'standard'),
('entomoculture', 'Hermetia illucens', 'fcr', 2.5, 3.0, 3.5, 'ratio', 'standard');

