// lib/types/bea.ts
export interface BienEtreIndiceBase {
    enclos_id: number;
    date: string;
    indice_global: number;
    indice_proprete?: number;
    indice_acces_eau?: number;
    indice_densite?: number;
    indice_comportement?: number;
    notes?: string;
}

export interface BienEtreIndiceCreate extends BienEtreIndiceBase { }
export interface BienEtreIndiceResponse extends BienEtreIndiceBase {
    id: number;
    created_at: string;
}

export interface BienEtreCritereResponse {
    id: number;
    nom: string;
    description?: string;
    ponderation: number;
    seuil_alerte?: number;
}