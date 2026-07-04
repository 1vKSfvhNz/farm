// lib/types/compost.ts
export type CompostType = 'déchets verts' | 'fumier' | 'mixte';

export interface CompostBase {
    name: string;
    type: CompostType;
    date_demarrage: string;
    volume_initial: number;
    volume_final?: number;
    date_maturite_estimee?: string;
    date_maturite_reelle?: string;
    utilisation_finale?: string;
    notes?: string;
}

export interface CompostCreate extends CompostBase { }
export interface CompostUpdate extends Partial<CompostBase> { }

export interface CompostResponse extends CompostBase {
    id: number;
    created_at: string;
    updated_at: string;
}

export interface RetournementCompostBase {
    compost_id: number;
    date_retournement: string;
    responsable?: string;
    temperature_avant?: number;
    temperature_apres?: number;
    humidite_avant?: number;
    humidite_apres?: number;
    notes?: string;
}

export interface RetournementCompostCreate extends RetournementCompostBase { }
export interface RetournementCompostResponse extends RetournementCompostBase {
    id: number;
    created_at: string;
}