// lib/types/dashboard.ts
export interface DashboardData {
    animaux: {
        total: number;
        par_espece: Record<string, number>;
        par_statut: Record<string, number>;
    };
    enclos: {
        total: number;
        taux_occupation_moyen: number;
        alerte_surpopulation: number;
    };
    production: {
        production_quotidienne: Record<string, number>;
        production_mensuelle: Record<string, number>;
    };
    financier: {
        ca_mois: number;
        depenses_mois: number;
        benefice_mois: number;
        tresorerie: number;
    };
    alertes: {
        non_lues: number;
        critiques: number;
    };
}

export interface HealthStatus {
    sante_globale: string;
    vaccins_a_jour: number;
    vaccins_en_retard: number;
    mortalite_30j: number;
    alertes_sanitaires: number;
}

export interface DashboardRecentActivity {
    id: number;
    type: "naissance" | "mortalite" | "vaccination" | "vente" | "alerte" | "recolte";
    title: string;
    description: string;
    date: string;
    entity_id?: number;
    entity_type?: string;
    severity?: "info" | "warning" | "critical";
    icon?: string;
    color?: string;
}

export interface DashboardStats {
    total_animaux: number;
    naissances_mois: number;
    mortalites_mois: number;
    alertes_critiques: number;
    chiffre_affaires_mois: number;
    depenses_mois: number;
    benefice_mois: number;
}

export interface DashboardProduction {
    lait_jour: number;
    oeufs_jour: number;
    oeufs_poids_kg: number;
    miel_kg: number;
    larves_kg: number;
}

export interface DashboardResponse {
    animals: {
        total: number;
        by_species: Record<string, number>;
    };
    enclos: {
        total: number;
        occupation_moyenne: number;
    };
    financial: {
        ca_mois: number;
        ca_mois_dernier: number;
        depenses_mois: number;
    };
    alerts: {
        critical: number;
        warning: number;
        info: number;
        total: number;
    };
    production: DashboardProduction;
    last_update: string;
}

export type RecentActivitiesResponse = DashboardRecentActivity[];