// lib/types/entomoculture.ts
export type StadeInsecte = 'oeuf' | 'larve' | 'pupe' | 'adulte';
export type TypeProductionInsecte = 'larves' | 'reproduction' | 'oeufs';

export interface EntomocultureLotBase {
    identification: string;
    espece: string;
    stade_actuel: StadeInsecte;
    date_arrivee: string;
    provenance?: string;
    prix_achat?: number;
    poids_initial?: number;
    quantite_estimative?: number;
    enclos_id?: number;
    type_production: TypeProductionInsecte;
    notes?: string;
}

export interface EntomocultureLotCreate extends EntomocultureLotBase { }
export interface EntomocultureLotUpdate extends Partial<EntomocultureLotBase> { }

export interface EntomocultureLotResponse extends EntomocultureLotBase {
    id: number;
    taux_mortalite?: number;
    created_at: string;
    updated_at: string;
}

export interface EntomocultureCycleBase {
    lot_id: number;
    date_debut: string;
    date_fin?: string;
    stade_debut: StadeInsecte;
    stade_fin?: StadeInsecte;
    production_grammes?: number;
    taux_mortalite?: number;
    substrat_utilise?: string;
}

export interface EntomocultureCycleCreate extends EntomocultureCycleBase { }
export interface EntomocultureCycleResponse extends EntomocultureCycleBase {
    id: number;
    created_at: string;
}

export interface EntomocultureStats {
    total_lots: number;
    en_cours: number;
    termines: number;
    total_cycles: number;
    production_totale_grammes: number;
    taux_mortalite_moyen: number;
    especes_distinctes: number;
}

export interface EntomocultureCycleStats {
    total_cycles: number;
    cycles_termines: number;
    duree_moyenne_jours: number;
    production_moyenne_grammes: number;
}