// lib/types/experimental.ts
export interface ReferenceHypothesisBase {
    espece: string;
    race?: string;
    parametre: string;
    valeur_estimee: number;
    unite?: string;
}

export interface ReferenceHypothesisCreate extends ReferenceHypothesisBase { }

export interface ReferenceHypothesisResponse extends ReferenceHypothesisBase {
    id: number;
    utilisateur_id?: number;
    date_creation: string;
    validee: boolean;
    date_validation?: string;
    created_at: string;
}

export interface ExperimentalModeResponse {
    mode: string;
    jours_collecte: number;
    nombre_donnees_par_espece: Record<string, number>;
    confiance_moyenne: number;
    seuils_atteints: string[];
    recommandations: string[];
}

export interface ConfidenceResponse {
    prediction_type: string;
    confidence: number;
    confidence_label: string;
    facteurs: string[];
    donnees_manquantes: string[];
    recommandations: string[];
}

export interface ReferenceGenerationRequest {
    espece: string;
    force_regenerate: boolean;
}

export interface ReferenceGenerationResponse {
    espece: string;
    nouvelles_references: Array<Record<string, any>>;
    nombre_donnees_utilisees: number;
    confiance_atteinte: number;
    message: string;
}