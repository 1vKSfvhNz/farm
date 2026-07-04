// lib/types/apiary.ts
export type StatutRuche = 'active' | 'orpheline' | 'en_essaimage' | 'morte';

export interface RucheBase {
    emplacement?: string;
    date_installation: string;
    race?: string;
    statut: StatutRuche;
    nombre_cadres?: number;
    notes?: string;
}

export interface RucheCreate extends RucheBase { }
export interface RucheUpdate extends Partial<RucheCreate> { }

export interface RucheResponse extends RucheBase {
    id: number;
    identification: string;
    created_at: string;
    updated_at: string;
}

export interface ReineBase {
    ruche_id: number;
    annee_naissance: number;
    origine?: string;
    date_remplacement?: string;
    est_marquee: boolean;
    notes?: string;
}

export interface ReineCreate extends ReineBase { }
export interface ReineResponse extends ReineBase {
    id: number;
    created_at: string;
    updated_at: string;
}

export interface RecolteMielBase {
    ruche_id: number;
    date_recolte: string;
    poids_kg: number;
    qualite?: string;
    taux_eau?: number;
    notes?: string;
}

export interface RecolteMielCreate extends RecolteMielBase { }
export interface RecolteMielResponse extends RecolteMielBase {
    id: number;
    created_at: string;
    updated_at: string;
}

export interface RucheStats {
    total_ruches: number;
    actives: number;
    orphelines: number;
    en_essaimage: number;
    mortes: number;
    total_reines: number;
    total_recoltes: number;
    poids_miel_total: number;
    poids_miel_moyen: number;
}

export interface ReineResponse {
    id: number;
    ruche_id: number;
    annee_naissance: number;
    origine?: string;
    date_remplacement?: string;
    est_marquee: boolean;
    notes?: string;
    created_at: string;
    updated_at: string;
}

export interface RecolteMielResponse {
    id: number;
    ruche_id: number;
    date_recolte: string;
    poids_kg: number;
    qualite?: string;
    taux_eau?: number;
    notes?: string;
    created_at: string;
    updated_at: string;
}
