// lib/types/accounting.ts
export type CategorieDepense =
    | 'achat_animaux' | 'achat_oeufs' | 'alimentation'
    | 'vaccins_soins' | 'equipement' | 'personnel'
    | 'eau_electricite' | 'entretien' | 'compostage'
    | 'transport' | 'frais_divers';

export type CategorieRecette =
    | 'vente_animaux_vivants' | 'vente_viande' | 'vente_lait'
    | 'vente_laine' | 'vente_oeufs' | 'vente_larves'
    | 'vente_compost' | 'vente_fumier' | 'subventions' | 'autres';

export interface DepenseBase {
    categorie: CategorieDepense;
    montant: number;
    date: string;
    description?: string;
    fournisseur?: string;
    quantite?: number;
    prix_unitaire?: number;
    animal_id?: number;
    lot_entomo_id?: number;
    piece_jointe_url?: string;
}

export interface DepenseCreate extends DepenseBase { }
export interface DepenseUpdate extends Partial<DepenseBase> { }

export interface DepenseResponse extends DepenseBase {
    id: number;
    created_at: string;
    updated_at: string;
}

export interface RecetteBase {
    categorie: CategorieRecette;
    montant: number;
    date: string;
    description?: string;
    client?: string;
    quantite?: number;
    prix_unitaire?: number;
    animal_id?: number;
    lot_entomo_id?: number;
}

export interface RecetteCreate extends RecetteBase { }
export interface RecetteUpdate extends Partial<RecetteBase> { }

export interface RecetteResponse extends RecetteBase {
    id: number;
    created_at: string;
    updated_at: string;
}

export interface AccountSummary {
    total_depenses: number;
    total_recettes: number;
    benefice: number;
    marge_brute_pourcent: number;
    depenses_par_categorie: Record<string, number>;
    recettes_par_categorie: Record<string, number>;
    tresorerie_previsionnelle: Record<string, number>;
}