// lib/types/vaccination.ts
export interface MaladieBase {
    nom: string;
    especes_concernees: string;
    description?: string;
    vaccin_disponible?: string;
}

export interface MaladieCreate extends MaladieBase { }
export interface MaladieResponse extends MaladieBase {
    id: number;
    created_at: string;
}

export interface VaccinBase {
    nom: string;
    fabricant?: string;
    lot?: string;
    maladie_id: number;
}

export interface VaccinCreate extends VaccinBase { }
export interface VaccinResponse extends VaccinBase {
    id: number;
    created_at: string;
}

export interface VaccinationBase {
    animal_id: number;
    maladie_id: number;
    vaccin_id?: number;
    date_prevue: string;
    date_realisee?: string;
    dose?: string;
    rappel_necessaire: boolean;
    date_prochain_rappel?: string;
    veterinaire_responsable?: string;
    cout?: number;
    notes?: string;
}

export interface VaccinationCreate extends VaccinationBase { }
export interface VaccinationUpdate extends Partial<VaccinationBase> { }

export interface VaccinationResponse extends VaccinationBase {
    id: number;
    est_a_jour: boolean;
    created_at: string;
    updated_at: string;
}