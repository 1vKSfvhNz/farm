// lib/types/alerts.ts
export interface AlertBase {
    type: string;
    niveau: 'info' | 'warning' | 'critical';
    message: string;
    espece?: string;
    animal_id?: number;
    enclos_id?: number;
    entite_type?: string;
    entite_id?: number;
    date_alerte: string;
    date_limite?: string;
}

export interface AlertCreate extends AlertBase { }
export interface AlertResponse extends AlertBase {
    id: number;
    est_lue: boolean;
    est_traitee: boolean;
    utilisateur_traitement_id?: number;
    date_traitement?: string;
    created_at: string;
}