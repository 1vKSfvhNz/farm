// lib/types/video.ts
export interface CameraBase {
    name: string;
    url: string;
    enclos_id?: number;
    is_active: boolean;
    roi_config?: string;
    fps: number;
}

export interface CameraCreate extends CameraBase { }
export interface CameraUpdate extends Partial<CameraBase> { }

export interface CameraResponse extends CameraBase {
    id: number;
    created_at: string;
    updated_at: string;
}

export interface VideoRecordBase {
    camera_id: number;
    animal_id?: number;
    enclos_id?: number;
    debut: string;
    fin?: string;
    duree_secondes?: number;
    file_path: string;
    file_size_mb?: number;
    action_type?: string;
    action_id?: number;
    utilisateur_id?: number;
}

export interface VideoRecordCreate extends VideoRecordBase { }
export interface VideoRecordResponse extends VideoRecordBase {
    id: number;
    ia_anomalie_detectee: boolean;
    ia_anomalie_type?: string;
    ia_confiance?: number;
    created_at: string;
}