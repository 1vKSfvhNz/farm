-- ================================================================================
-- FICHIER 3: 03_references_vaccination.sql
-- Table: references_vaccination
-- ================================================================================

-- BOVINS
INSERT INTO references_vaccination (espece, maladie, age_recommande_jours, rappel_mois, saison_recommandee, vaccin_nom, source) VALUES
('bovin', 'Pasteurellose', 90, 6, 'Debut saison des pluies', 'Pasteurella multocida', 'standard'),
('bovin', 'Clostridiose', 60, 12, 'Toute année', 'Clostridial vaccine', 'standard'),
('bovin', 'Charbon symptomatique', 90, 12, 'Avant saison des pluies', 'Charbon vaccine', 'standard'),
('bovin', 'Fièvre aphteuse', 120, 6, 'Selon zone', 'Aphteux vaccine', 'standard'),
('bovin', 'IBR', 120, 12, 'Printemps', 'IBR vaccine', 'standard'),
('bovin', 'BVD', 120, 12, 'Printemps', 'BVD vaccine', 'standard');

-- OVINS
INSERT INTO references_vaccination (espece, maladie, age_recommande_jours, rappel_mois, saison_recommandee, vaccin_nom, source) VALUES
('ovin', 'Clostridiose', 30, 12, 'Avant mise à l''herbe', 'Clostridial vaccine', 'standard'),
('ovin', 'Pasteurellose', 60, 12, 'Debut saison des pluies', 'Pasteurella vaccine', 'standard'),
('ovin', 'Charbon symptomatique', 60, 12, 'Avant saison des pluies', 'Charbon vaccine', 'standard'),
('ovin', 'Chlamydiose', 150, 12, 'Avant reproduction', 'Chlamydia vaccine', 'standard'),
('ovin', 'Piétin', NULL, NULL, 'Automne/Printemps', 'Footrot vaccine', 'standard');

-- CAPRINS
INSERT INTO references_vaccination (espece, maladie, age_recommande_jours, rappel_mois, saison_recommandee, vaccin_nom, source) VALUES
('caprin', 'Clostridiose', 30, 12, 'Avant mise à l''herbe', 'Clostridial vaccine', 'standard'),
('caprin', 'Pasteurellose', 60, 12, 'Debut saison des pluies', 'Pasteurella vaccine', 'standard'),
('caprin', 'Charbon symptomatique', 60, 12, 'Avant saison des pluies', 'Charbon vaccine', 'standard'),
('caprin', 'Chlamydiose', 150, 12, 'Avant reproduction', 'Chlamydia vaccine', 'standard');

-- AVICOLES
INSERT INTO references_vaccination (espece, maladie, age_recommande_jours, rappel_mois, saison_recommandee, vaccin_nom, source) VALUES
('avicole', 'Maladie de Newcastle', 3, 1, 'Toute année', 'Newcastle vaccine La Sota', 'standard'),
('avicole', 'Gumboro', 10, 0, 'Toute année', 'Gumboro vaccine', 'standard'),
('avicole', 'Bronchite infectieuse', 3, 0, 'Toute année', 'IBV vaccine', 'standard'),
('avicole', 'Variole aviaire', 28, 0, 'Saison sèche', 'Fowl pox vaccine', 'standard'),
('avicole', 'Salmonellose', 1, 0, 'À l''éclosion', 'Salmonella vaccine', 'standard');

-- PISCICOLES
INSERT INTO references_vaccination (espece, maladie, age_recommande_jours, rappel_mois, saison_recommandee, vaccin_nom, source) VALUES
('piscicole', 'Streptococcose', 60, 12, 'Toute année', 'Streptococcus vaccine', 'standard'),
('piscicole', 'VHS', NULL, 12, 'Toute année', 'VHS vaccine', 'standard'),
('piscicole', 'Yersiniose', NULL, 12, 'Toute année', 'Yersinia vaccine', 'standard');
