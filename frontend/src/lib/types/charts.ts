// lib/types/charts.ts
export interface ChartDataPoint {
    label: string;
    value: number;
    date?: string;
}

export interface ChartSeries {
    name: string;
    data: ChartDataPoint[];
    color?: string;
}

export interface ChartConfig {
    title: string;
    type: 'line' | 'bar' | 'pie' | 'area' | 'radar';
    xAxisLabel?: string;
    yAxisLabel?: string;
    height?: number;
    showLegend?: boolean;
    showTooltip?: boolean;
    animation?: boolean;
}

export interface GrowthChartData {
    animalId: number;
    animalName: string;
    weighings: Array<{
        date: string;
        poids: number;
        age_jours: number;
    }>;
    predictions?: Array<{
        date: string;
        poids_min: number;
        poids_max: number;
        poids_moyen: number;
    }>;
}

export interface ProductionChartData {
    espece: string;
    type: 'lait' | 'oeufs' | 'miel' | 'larves';
    daily: ChartDataPoint[];
    monthly: ChartDataPoint[];
    yearly: ChartDataPoint[];
}

export interface FinancialChartData {
    depenses: ChartDataPoint[];
    recettes: ChartDataPoint[];
    benefice: ChartDataPoint[];
    period: 'day' | 'month' | 'year';
}

export interface HealthChartData {
    vaccinations: Array<{
        date: string;
        maladie: string;
        status: 'realise' | 'prevue' | 'retard';
    }>;
    mortality: ChartDataPoint[];
    bea: ChartDataPoint[];
}

export interface DashboardChartData {
    animalsBySpecies: ChartDataPoint[];
    animalsByStatus: ChartDataPoint[];
    enclosOccupation: ChartDataPoint[];
    productionBySpecies: ChartSeries[];
    financialEvolution: FinancialChartData;
}