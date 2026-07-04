// lib/types/caprin.ts
import type { AnimalBase, AnimalVenteStats } from './animal';

export interface CaprinBase extends AnimalBase {
    production_viande: boolean;
    production_reproduction: boolean;
    poids_initial?: number;
    photo_base64?: string;
}

export interface CaprinCreate extends CaprinBase { }
export interface CaprinUpdate extends Partial<CaprinBase> { }

export interface CaprinResponse extends CaprinBase {
    id: number;
    identification: string;    
    photo_url: string;
    age_mois: number;
    dernier_poids: number;
    created_at: string;
    updated_at: string;
}

export interface CaprinStats extends AnimalVenteStats {
    total_caprins: number;
    production_viande: number;
    production_reproduction: number;
}