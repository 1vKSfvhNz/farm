-- ================================================================================
-- FICHIER 4: 04_references_nutrition.sql
-- Table: references_nutrition
-- ================================================================================

-- BOVINS
INSERT INTO references_nutrition (espece, categorie, proteines_pourcent, energie_kcal, calcium_g, phosphore_g, source) VALUES
('bovin', 'Veau_0_3_mois', 22, NULL, 8, 6, 'standard'),
('bovin', 'Veau_0_3_mois', 22, NULL, 12, 9, 'standard'),
('bovin', 'Genisse_3_12_mois', 16, NULL, 6, 4, 'standard'),
('bovin', 'Genisse_3_12_mois', 16, NULL, 8, 6, 'standard'),
('bovin', 'Vache_laitiere_production', 16, NULL, 8, 5, 'standard'),
('bovin', 'Vache_laitiere_production', 16, NULL, 10, 7, 'standard'),
('bovin', 'Vache_allaitante_tarie', 10, NULL, 4, 3, 'standard'),
('bovin', 'Vache_allaitante_tarie', 10, NULL, 5, 4, 'standard'),
('bovin', 'Boeuf_engraissement', 14, NULL, 5, 4, 'standard'),
('bovin', 'Boeuf_engraissement', 14, NULL, 7, 5, 'standard');

-- OVINS
INSERT INTO references_nutrition (espece, categorie, proteines_pourcent, energie_kcal, calcium_g, phosphore_g, source) VALUES
('ovin', 'Agneaux', 20, NULL, 6, 4, 'standard'),
('ovin', 'Agneaux', 20, NULL, 8, 6, 'standard'),
('ovin', 'Brebis_allaitement', 15, NULL, 5, 3, 'standard'),
('ovin', 'Brebis_allaitement', 15, NULL, 7, 5, 'standard'),
('ovin', 'Engraissement', 12, NULL, 4, 3, 'standard'),
('ovin', 'Engraissement', 12, NULL, 5, 4, 'standard');

-- CAPRINS
INSERT INTO references_nutrition (espece, categorie, proteines_pourcent, energie_kcal, calcium_g, phosphore_g, source) VALUES
('caprin', 'Chevreaux', 20, NULL, 6, 4, 'standard'),
('caprin', 'Chevreaux', 20, NULL, 8, 6, 'standard'),
('caprin', 'Chevre_allaitement', 15, NULL, 5, 3, 'standard'),
('caprin', 'Chevre_allaitement', 15, NULL, 7, 5, 'standard'),
('caprin', 'Engraissement', 12, NULL, 4, 3, 'standard'),
('caprin', 'Engraissement', 12, NULL, 5, 4, 'standard');

-- AVICOLES
INSERT INTO references_nutrition (espece, categorie, proteines_pourcent, energie_kcal, calcium_g, phosphore_g, source) VALUES
('avicole', 'Poussin_0_4_sem', 20, 2900, 0.9, NULL, 'standard'),
('avicole', 'Poussin_0_4_sem', 22, 3000, 1.0, NULL, 'standard'),
('avicole', 'Poulet_local_4_10_sem', 16, 2700, 0.8, NULL, 'standard'),
('avicole', 'Poulet_local_4_10_sem', 16, 2800, 0.8, NULL, 'standard'),
('avicole', 'Poulet_ameliore_chair', 18, 3000, 0.8, NULL, 'standard'),
('avicole', 'Poulet_ameliore_chair', 20, 3200, 0.9, NULL, 'standard'),
('avicole', 'Pondeuse', 16, 2650, 3.5, NULL, 'standard'),
('avicole', 'Pondeuse', 17, 2800, 4.0, NULL, 'standard');

-- PISCICOLES (en % de la ration)
INSERT INTO references_nutrition (espece, categorie, proteines_pourcent, energie_kcal, source) VALUES
('tilapia', 'Standard', 30, NULL, 'standard'),
('tilapia', 'Standard', 32, NULL, 'standard'),
('carpe', 'Standard', 30, NULL, 'standard'),
('carpe', 'Standard', 32, NULL, 'standard'),
('silure', 'Standard', 35, NULL, 'standard'),
('silure', 'Standard', 40, NULL, 'standard');
