// lib/stores/charts.ts
import { writable } from 'svelte/store';
import type {
    DashboardChartData,
    GrowthChartData,
    ProductionChartData,
    FinancialChartData,
    HealthChartData
} from '../types/charts';

interface ChartsState {
    dashboard: DashboardChartData | null;
    growth: Map<number, GrowthChartData>;
    production: Map<string, ProductionChartData>;
    financial: FinancialChartData | null;
    health: HealthChartData | null;
    isLoading: boolean;
    selectedPeriod: 'day' | 'week' | 'month' | 'year';
}

function createChartsStore() {
    const initialState: ChartsState = {
        dashboard: null,
        growth: new Map(),
        production: new Map(),
        financial: null,
        health: null,
        isLoading: false,
        selectedPeriod: 'month'
    };

    const { subscribe, set, update } = writable<ChartsState>(initialState);

    return {
        subscribe,

        setDashboardData: (data: DashboardChartData) => {
            update(state => ({ ...state, dashboard: data }));
        },

        setGrowthData: (animalId: number, data: GrowthChartData) => {
            update(state => {
                const growth = new Map(state.growth);
                growth.set(animalId, data);
                return { ...state, growth };
            });
        },

        setProductionData: (key: string, data: ProductionChartData) => {
            update(state => {
                const production = new Map(state.production);
                production.set(key, data);
                return { ...state, production };
            });
        },

        setFinancialData: (data: FinancialChartData) => {
            update(state => ({ ...state, financial: data }));
        },

        setHealthData: (data: HealthChartData) => {
            update(state => ({ ...state, health: data }));
        },

        setSelectedPeriod: (period: ChartsState['selectedPeriod']) => {
            update(state => ({ ...state, selectedPeriod: period }));
        },

        setLoading: (isLoading: boolean) => {
            update(state => ({ ...state, isLoading }));
        },

        clearGrowthData: (animalId?: number) => {
            update(state => {
                if (animalId) {
                    const growth = new Map(state.growth);
                    growth.delete(animalId);
                    return { ...state, growth };
                }
                return { ...state, growth: new Map() };
            });
        },

        clearAll: () => {
            set(initialState);
        }
    };
}

export const chartsStore = createChartsStore();