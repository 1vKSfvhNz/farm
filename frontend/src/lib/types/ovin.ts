// lib/types/ovin.ts
import type { AnimalBase, AnimalVenteStats } from './animal';

export interface OvinBase extends AnimalBase {
    production_viande: boolean;
    production_reproduction: boolean;
    production_laine: boolean;
    qualite_laine?: string;
    poids_initial?: number;
    photo_base64?: string;
}

export interface OvinCreate extends OvinBase { }
export interface OvinUpdate extends Partial<OvinBase> { }

export interface OvinResponse extends OvinBase {
    id: number;
    identification: string;    
    photo_url: string;
    age_mois: number;
    dernier_poids: number;
    created_at: string;
    updated_at: string;
}

export interface OvinStats extends AnimalVenteStats {
    total_ovins: number;
    production_viande: number;
    production_reproduction: number;
    production_laine: number;
}