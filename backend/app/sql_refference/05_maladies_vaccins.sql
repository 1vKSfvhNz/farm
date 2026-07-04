-- ================================================================================
-- FICHIER 5: 05_maladies_vaccins.sql
-- Tables: maladies, vaccins (références pour les modèles existants)
-- ================================================================================

-- MALADIES
INSERT INTO maladies (nom, especes_concernees, description, vaccin_disponible) VALUES
('Pasteurellose', 'bovin,ovin,caprin', 'Infection respiratoire bactérienne', 'Pasteurella vaccine'),
('Clostridiose', 'bovin,ovin,caprin', 'Entérotoxémie due à Clostridium', 'Clostridial vaccine'),
('Charbon symptomatique', 'bovin,ovin,caprin', 'Infection bactérienne aiguë', 'Charbon vaccine'),
('Fièvre aphteuse', 'bovin', 'Maladie virale hautement contagieuse', 'Aphteux vaccine'),
('IBR', 'bovin', 'Rhinotrachéite infectieuse bovine', 'IBR vaccine'),
('BVD', 'bovin', 'Diarrhée virale bovine', 'BVD vaccine'),
('Maladie de Newcastle', 'avicole', 'Maladie virale respiratoire et nerveuse', 'Newcastle vaccine'),
('Gumboro', 'avicole', 'Bursite infectieuse aviaire', 'Gumboro vaccine'),
('Bronchite infectieuse', 'avicole', 'Infection respiratoire virale', 'IBV vaccine'),
('Variole aviaire', 'avicole', 'Infection virale cutanée', 'Fowl pox vaccine'),
('Salmonellose', 'avicole', 'Infection bactérienne zoonotique', 'Salmonella vaccine'),
('Streptococcose', 'piscicole', 'Infection bactérienne chez les poissons', 'Streptococcus vaccine'),
('Chlamydiose', 'ovin,caprin', 'Cause d''avortement chez les petits ruminants', 'Chlamydia vaccine'),
('Piétin', 'ovin', 'Infection podale bactérienne', 'Footrot vaccine'),
('VHS', 'piscicole', 'Septicémie hémorragique virale', 'VHS vaccine'),
('Yersiniose', 'piscicole', 'Infection bactérienne entérique', 'Yersinia vaccine');

-- VACCINS
INSERT INTO vaccins (nom, fabricant, lot, maladie_id) VALUES
('Pasteurella multocida', 'Merial', 'PAS2024-01', (SELECT id FROM maladies WHERE nom = 'Pasteurellose' LIMIT 1)),
('Clostridial vaccine', 'Ceva', 'CLO2024-01', (SELECT id FROM maladies WHERE nom = 'Clostridiose' LIMIT 1)),
('Charbon vaccine', 'Boehringer', 'CHA2024-01', (SELECT id FROM maladies WHERE nom = 'Charbon symptomatique' LIMIT 1)),
('Aphteux vaccine', 'Merial', 'APH2024-01', (SELECT id FROM maladies WHERE nom = 'Fièvre aphteuse' LIMIT 1)),
('Newcastle vaccine La Sota', 'Ceva', 'NEW2024-01', (SELECT id FROM maladies WHERE nom = 'Maladie de Newcastle' LIMIT 1)),
('Gumboro vaccine', 'Merial', 'GUM2024-01', (SELECT id FROM maladies WHERE nom = 'Gumbaro' LIMIT 1)),
('Streptococcus vaccine', 'Pharmaq', 'STR2024-01', (SELECT id FROM maladies WHERE nom = 'Streptococcose' LIMIT 1));
