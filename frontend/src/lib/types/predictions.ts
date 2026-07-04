// lib/types/predictions.ts (complété)
export interface PredictionRequest {
    espece: string;
    race?: string;
    animal_id?: number;
    lot_id?: number;
    prediction_type: 'growth' | 'production' | 'health' | 'cashflow' | 'compost' | 'water_quality';
    horizon_jours: number;
}

export interface PredictionResponse {
    prediction_id: string;
    espece: string;
    prediction_type: string;
    horizon_jours: number;
    predictions: Record<string, any>;
    confidence: number;
    confidence_level: string;
    warnings: string[];
    generated_at: string;
}

export interface GrowthPredictionResponse {
    animal_id: number;
    age_actuel_jours: number;
    poids_actuel_kg: number;
    poids_prevu_jours: Array<{
        jour: number;
        poids_min: number;
        poids_max: number;
        poids_moyen: number;
    }>;
    date_atteinte_poids_vente?: string;
    retard_croissance_detecte: boolean;
    recommandations: string[];
}

export interface ProductionPredictionResponse {
    espece: string;
    type_production: string;
    production_quotidienne_actuelle: number;
    production_prevue_15j: number;
    production_prevue_30j: number;
    production_prevue_90j: number;
    saisonnalite_impact?: number;
    recommandations: string[];
}

export interface CashflowPredictionResponse {
    tresorerie_actuelle: number;
    entrees_prevues_30j: number;
    sorties_prevues_30j: number;
    tresorerie_prevue_30j: number;
    tresorerie_prevue_60j: number;
    tresorerie_prevue_90j: number;
    seuil_alerte_atteint: boolean;
    recommandations: string[];
}