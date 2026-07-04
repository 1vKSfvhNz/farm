// lib/types/websocket.ts
export type WebSocketEventType =
    | 'alert'
    | 'notification'
    | 'animal_update'
    | 'enclos_update'
    | 'production_update'
    | 'financial_update'
    | 'vaccination_reminder'
    | 'water_quality_alert'
    | 'bea_update'
    | 'prediction_update'
    | 'compost_update'
    | 'apiculture_update'
    | 'entomoculture_update'
    | 'system_status';

export interface WebSocketMessage {
    type: WebSocketEventType;
    data: any;
    timestamp: string;
    id?: string;
}

export interface AlertWebSocketData {
    alert_id: number;
    niveau: 'info' | 'warning' | 'critical';
    message: string;
    espece?: string;
    animal_id?: number;
    enclos_id?: number;
}

export interface AnimalUpdateWebSocketData {
    animal_id: number;
    espece: string;
    action: 'create' | 'update' | 'delete' | 'status_change';
    data: any;
}

export interface EnclosUpdateWebSocketData {
    enclos_id: number;
    action: 'create' | 'update' | 'delete' | 'occupation_change';
    data: any;
}

export interface ProductionUpdateWebSocketData {
    espece: string;
    type: string;
    quantite: number;
    date: string;
    enclos_id?: number;
}

export interface VaccinationReminderWebSocketData {
    vaccination_id: number;
    animal_id: number;
    animal_identification: string;
    maladie: string;
    date_prevue: string;
    jours_restants: number;
}

export interface WaterQualityAlertWebSocketData {
    enclos_id: number;
    enclos_name: string;
    parametre: string;
    valeur: number;
    seuil: number;
    niveau: 'warning' | 'critical';
}

export interface WebSocketConnectionStatus {
    connected: boolean;
    reconnecting: boolean;
    error: string | null;
    lastMessage: Date | null;
}