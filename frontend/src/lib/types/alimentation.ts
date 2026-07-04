// lib/types/alimentation.ts
export interface AlimentationBase {
    animal_id?: number;
    lot_entomo_id?: number;
    date: string;
    poids_nourriture: number;
    type_nourriture: string;
    composition?: string;
    cout?: number;
}

export interface AlimentationCreate extends AlimentationBase { }
export interface AlimentationUpdate extends Partial<AlimentationBase> { }

export interface AlimentationResponse extends AlimentationBase {
    id: number;
    created_at: string;
    updated_at: string;
}

export interface RationAlimentaireBase {
    espece: string;
    race?: string;
    categorie: string;
    proteines_pourcent?: number;
    energie_kcal?: number;
    calcium_g?: number;
    phosphore_g?: number;
    quantite_recommandee_kg?: number;
    notes?: string;
}

export interface RationAlimentaireCreate extends RationAlimentaireBase { }
export interface RationAlimentaireResponse extends RationAlimentaireBase {
    id: number;
    created_at: string;
}