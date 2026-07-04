// lib/types/bovin.ts

import type { AnimalBase, AnimalVenteStats } from "./animal";

export interface BovinBase extends AnimalBase {
    production_laitiere: boolean;
    production_viande: boolean;
    production_reproduction: boolean;
    lactation_en_cours: boolean;
    production_lait_quotidienne?: number;
    poids_initial?: number;
    photo_base64?: string;
}

export interface BovinCreate extends BovinBase { }
export interface BovinUpdate extends Partial<BovinCreate> { }

export interface BovinResponse extends BovinBase {
    id: number;
    identification: string;    
    photo_url: string;
    age_mois: number;
    dernier_poids: number;
    created_at: string;
    updated_at: string;
}

export interface BovinStats extends AnimalVenteStats {
    total_bovins: number;
    production_laitiere: number;
    production_viande: number;
    production_reproduction: number;
    lactation_en_cours: number;
    production_lait_moyenne: number;
}