// lib/types/piscicole.ts
import type { AnimalBase } from './animal';

export interface PiscicoleBase extends AnimalBase {
    production_viande: boolean;
    production_reproduction: boolean;
    taille_moyenne?: number;
    poids_moyen?: number;
}

export interface PiscicoleCreate extends PiscicoleBase { }
export interface PiscicoleUpdate extends Partial<PiscicoleBase> { }

export interface PiscicoleResponse extends PiscicoleBase {
    id: number;
    created_at: string;
    updated_at: string;
}

export interface PiscicoleStats {
    total_piscicoles: number;
    production_viande: number;
    production_reproduction: number;
    taille_moyenne_globale: number;
    taille_min: number;
    taille_max: number;
}