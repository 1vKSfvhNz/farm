// lib/types/water_quality.ts
export interface WaterQualityBase {
    enclos_id: number;
    timestamp: string;
    ph?: number;
    temperature?: number;
    oxygene_dissous?: number;
    ammoniac?: number;
    nitrites?: number;
    nitrates?: number;
    conductivite?: number;
    turbidite?: number;
    source?: string;
}

export interface WaterQualityCreate extends WaterQualityBase { }
export interface WaterQualityResponse extends WaterQualityBase {
    id: number;
    alerte_generee: boolean;
    created_at: string;
}

export interface WaterQualityAlerteResponse {
    id: number;
    water_quality_id: number;
    parametre: string;
    valeur: number;
    seuil: number;
    niveau: string;
    message?: string;
    traitee: boolean;
    created_at: string;
}