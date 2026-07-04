// lib/types/avicole.ts
import type { AnimalBase } from './animal';

export interface AvicoleBase extends AnimalBase {
    production_viande: boolean;
    production_reproduction: boolean;
    production_ponte: boolean;
    oeufs_pondus_total: number;
    poids_oeufs_total: number;
}

export interface AvicoleCreate extends AvicoleBase { }
export interface AvicoleUpdate extends Partial<AvicoleBase> { }

export interface AvicoleResponse extends AvicoleBase {
    id: number;
    created_at: string;
    updated_at: string;
}

export interface AvicoleStats {
    total_avicoles: number;
    production_viande: number;
    production_reproduction: number;
    production_ponte: number;
    total_oeufs_pondus: number;
    total_poids_oeufs_kg: number;
    moyenne_oeufs_par_avicole: number;
    moyenne_poids_oeufs_kg_par_avicole: number;
}

export interface EggProductionStats {
    moyenne_par_jour: number;
    total_oeufs: number;
    total_poids_kg: number;
    moyenne_poids_par_oeuf_g: number;
    nombre_poules_pondeuses: number;
    periode_jours: number;
    date_debut: string;
    date_fin: string;
    tendance: "hausse" | "baisse" | "stable";
    variation_pourcent: number;
}

export interface EggProductionHistoryPoint {
    date: string;
    oeufs: number;
    poids_kg: number;
    moyenne_par_poule: number;
}

export interface EggProductionRecord {
    avicole_id: number;
    egg_count: number;
    egg_weight_grams: number;
    date_record?: string;
}