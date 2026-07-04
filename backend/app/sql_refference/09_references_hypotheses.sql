-- ================================================================================
-- FICHIER 9: 09_references_hypotheses.sql
-- Table: references_hypotheses (pour l'apprentissage et ajustements)
-- ================================================================================

-- INSERT INTO references_hypotheses (utilisateur_id, espece, race, parametre, valeur_estimee, unite, date_creation, validee, source)
-- Exemple d'hypothèse par défaut (utilisateur_id NULL = système)
INSERT INTO references_hypotheses (utilisateur_id, espece, race, parametre, valeur_estimee, unite, date_creation, validee, source) VALUES
(NULL, 'bovin', 'Zebu local', 'croissance', 0.65, 'kg/jour', DATE('now'), true, 'standard'),
(NULL, 'bovin', 'Metis Zebu', 'croissance', 0.85, 'kg/jour', DATE('now'), true, 'standard'),
(NULL, 'ovin', 'Oudah', 'croissance', 0.18, 'kg/jour', DATE('now'), true, 'standard'),
(NULL, 'caprin', 'Sahelienne', 'croissance', 0.19, 'kg/jour', DATE('now'), true, 'standard'),
(NULL, 'avicole', 'Poulet local', 'croissance', 12.5, 'g/jour', DATE('now'), true, 'standard'),
(NULL, 'tilapia', NULL, 'croissance', 2.2, 'g/jour', DATE('now'), true, 'standard'),
(NULL, 'silure', NULL, 'croissance', 1.9, 'g/jour', DATE('now'), true, 'standard');


