// lib/types/animal.ts
export type Sexe = 'male' | 'femelle' | 'hermaphrodite';
export type StatutAnimal = 'vivant' | 'vendu' | 'decede' | 'transfere';

export interface AnimalBase {
    type_espece: string;
    race: string;
    sexe: Sexe;
    date_naissance?: string;
    date_arrivee: string;
    provenance: string;
    prix_achat?: number;
    enclos_id: number;
    enclos_name: string;
    statut: StatutAnimal;
    type_production?: string;
    photo_url?: string;
    notes?: string;

    // === NOUVEAUX CHAMPS DE VENTE ===
    prix_vente?: number | null;
    date_vente?: string | null;
    client_acheteur?: string | null;
    note_vente?: string | null;
}

// Interface pour les filtres
export interface AnimauxFilters {
    searchQuery: string;
    selectedRace: string;
    selectedEnclos: string | number;
    selectedStatuts: StatutAnimal[];
    selectedSexes: Sexe[];
}
export interface LengthResponse{
    users_length: number;
    enclos_length: number;
    bovins_length: number;
    ovins_length: number;
    caprins_length: number;
    avicoles_length: number;
    piscicoles_length: number;
    ruches_length: number;
    nids_length: number;
}

// === NOUVEAU TYPE POUR L'ENREGISTREMENT DE VENTE ===
export interface AnimalVenteStats {
    total_ventes: number;
    montant_total_ventes: number;
    prix_vente_moyen: number;
}

// === NOUVEAU TYPE POUR L'ENREGISTREMENT DE VENTE ===
export interface AnimalVenteCreate {
    prix_vente: number;
    date_vente?: string | null;
    client_acheteur?: string | null;
    note_vente?: string | null;
    statut?: string;
}