-- ================================================================================
-- FICHIER 6: 06_rations_alimentaires.sql
-- Table: rations_alimentaires (référence supplémentaire)
-- ================================================================================

INSERT INTO rations_alimentaires (espece, race, categorie, proteines_pourcent, energie_kcal, calcium_g, phosphore_g, quantite_recommandee_kg, source) VALUES
('bovin', 'Zebu local', 'veau_0_3_mois', 18, NULL, 7, 5, 2.5, 'standard'),
('bovin', 'Zebu local', 'genisse_3_12_mois', 14, NULL, 5, 3, 5.0, 'standard'),
('bovin', 'Zebu local', 'vache_allaitante', 10, NULL, 4, 3, 8.0, 'standard'),
('bovin', 'Metis Zebu', 'veau_0_3_mois', 20, NULL, 8, 6, 3.0, 'standard'),
('bovin', 'Metis Zebu', 'genisse_3_12_mois', 16, NULL, 6, 4, 6.0, 'standard'),
('bovin', 'Metis Zebu', 'vache_laitiere', 16, NULL, 9, 6, 12.0, 'standard'),
('ovin', 'Oudah', 'agneau', 18, NULL, 6, 4, 0.5, 'standard'),
('ovin', 'Oudah', 'brebis_allaitement', 14, NULL, 5, 3, 1.2, 'standard'),
('caprin', 'Sahelienne', 'chevreau', 18, NULL, 6, 4, 0.4, 'standard'),
('caprin', 'Sahelienne', 'chevre_allaitement', 14, NULL, 5, 3, 1.0, 'standard'),
('avicole', 'Poulet local', 'poussin_0_4_sem', 18, 2800, 0.9, NULL, 0.050, 'standard'),
('avicole', 'Poulet local', 'croissance_4_10_sem', 15, 2600, 0.8, NULL, 0.080, 'standard'),
('avicole', 'Poulet amelioré ponte', 'pondeuse', 16, 2700, 3.5, NULL, 0.120, 'standard'),
('tilapia', NULL, 'croissance', 30, NULL, NULL, NULL, NULL, 'standard'),
('silure', NULL, 'croissance', 38, NULL, NULL, NULL, NULL, 'standard');

