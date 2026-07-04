// lib/stores/dashboard.ts
import { writable } from 'svelte/store';
import type { DashboardData, HealthStatus } from '../types/dashboard';

interface DashboardState {
    main: DashboardData | null;
    health: HealthStatus | null;
    lastUpdate: string | null;
    isLoading: boolean;
    error: string | null;
}

function createDashboardStore() {
    const initialState: DashboardState = {
        main: null,
        health: null,
        lastUpdate: null,
        isLoading: false,
        error: null
    };

    const { subscribe, set, update } = writable<DashboardState>(initialState);

    return {
        subscribe,

        setDashboard: (data: DashboardData) => {
            update(state => ({
                ...state,
                main: data,
                lastUpdate: new Date().toISOString(),
                isLoading: false,
                error: null
            }));
        },

        setHealth: (health: HealthStatus) => {
            update(state => ({ ...state, health }));
        },

        setLoading: (isLoading: boolean) => {
            update(state => ({ ...state, isLoading }));
        },

        setError: (error: string | null) => {
            update(state => ({ ...state, error, isLoading: false }));
        },

        refresh: () => {
            update(state => ({ ...state, isLoading: true }));
        },

        reset: () => {
            set(initialState);
        }
    };
}

export const dashboardStore = createDashboardStore();