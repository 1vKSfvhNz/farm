// lib/types/enclos.ts
export type EnclosType = 'enclos' | 'bassin' | 'pâturage' | 'cage' | 'bac';

export interface EnclosBase {
    name: string;
    type: EnclosType;
    longueur: number;
    largeur: number;
    hauteur?: number;  // Optionnel, utilisé pour les bassins/bacs
    localisation_gps?: string;
    zone?: string;
    description?: string;
}

export interface EnclosCreate extends EnclosBase { }
export interface EnclosUpdate extends Partial<EnclosBase> { }

export interface EnclosResponse extends EnclosBase {
    id: number;
    surface: number;      // Calculé: longueur * largeur
    volume?: number;      // Calculé: surface * hauteur (si hauteur existe)
    created_at: string;
    updated_at: string;
}

export interface EnclosStats {
    enclos_id: number;
    name: string;
    type: EnclosType;
    surface_m2: number;
    volume_m3?: number;
    animaux_par_espece: Record<string, number>;
    densite: number;           // animaux/m² ou poissons/m³ selon le type
    densite_unite: string;     // "animaux/m²" ou "poissons/m³"
    occupation_actuelle: number;
}

export interface EnclosDetailStats {
    id: number;
    name: string;
    type: EnclosType;
    surface_m2: number;
    volume_m3?: number;
    occupation_actuelle: number;
    densite: number;
    densite_unite: string;
    animaux_par_espece: Record<string, number>;
    animaux_par_sexe: Record<string, number>;
    zone?: string;
    localisation_gps?: string;
    description?: string;
}

export interface AllEnclosStatsResponse {
    total_enclos: number;
    surface_totale_m2: number;
    volume_total_m3: number;
    animaux_totaux: number;
    densite_moyenne_surfacique: number;  // animaux/m² moyen tous enclos confondus
    densite_moyenne_volumique: number;   // poissons/m³ moyen (bassins/bacs)
    statistiques_par_type: Record<string, {
        count: number;
        surface_totale_m2: number;
        volume_total_m3: number;
        animaux_actuels: number;
    }>;
}

// Alertes basées sur la densité plutôt que la capacité
export interface EnclosDensityAlert {
    enclos_id: number;
    enclos_name: string;
    type: EnclosType;
    densite_actuelle: number;
    densite_unite: string;
    niveau_alerte: "critical" | "warning" | "normal";
    recommandation?: string;
}

// Seuils de densité recommandés (à ajuster selon les espèces)
export const DENSITY_THRESHOLDS = {
    // Enclos terrestres (animaux/m²)
    enclos: { warning: 0.5, critical: 1.0 },
    cage: { warning: 1.0, critical: 2.0 },
    pâturage: { warning: 0.1, critical: 0.3 },
    // Milieux aquatiques (poissons/m³)
    bassin: { warning: 5.0, critical: 10.0 },
    bac: { warning: 10.0, critical: 20.0 }
};

export const getNiveauAlerteDensite = (
    type: EnclosType, 
    densite: number
): "critical" | "warning" | "normal" => {
    const thresholds = DENSITY_THRESHOLDS[type];
    if (!thresholds) return "normal";
    
    if (densite >= thresholds.critical) return "critical";
    if (densite >= thresholds.warning) return "warning";
    return "normal";
};

export const getRecommandationDensite = (
    type: EnclosType,
    densite: number,
    volume_m3?: number
): string => {
    const niveau = getNiveauAlerteDensite(type, densite);
    
    if (niveau === "normal") {
        return "Densité appropriée, conditions de vie optimales.";
    }
    
    if (type === 'bassin' || type === 'bac') {
        const volume = volume_m3 || 0;
        if (niveau === "warning") {
            return `Densité élevée (${densite} poissons/m³). Envisagez d'agrandir le volume ou de répartir les poissons dans d'autres bassins.`;
        }
        return `Densité critique (${densite} poissons/m³)! Risque de stress et de maladie. Réduisez immédiatement la population ou augmentez le volume d'eau.`;
    } else {
        if (niveau === "warning") {
            return `Densité élevée (${densite} animaux/m²). Envisagez d'agrandir la surface ou de répartir les animaux.`;
        }
        return `Densité critique (${densite} animaux/m²)! Espace insuffisant pour le bien-être animal. Agrandissez l'enclos ou réduisez la population.`;
    }
};