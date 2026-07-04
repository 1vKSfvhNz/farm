// lib/types/pesee.ts
export interface PeseeBase {
    animal_id?: number;
    lot_entomo_id?: number;
    lot_avicole_id?: number;
    date_pesee: string;
    poids: number;
    methode?: string;
    video_record_id?: number;
    notes?: string;
}

export interface PeseeCreate extends PeseeBase { }
export interface PeseeUpdate extends Partial<PeseeBase> { }

export interface PeseeResponse extends PeseeBase {
    id: number;
    created_at: string;
    updated_at: string;
}