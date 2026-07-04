// lib/stores/enclos.ts
import { writable } from 'svelte/store';

interface EnclosState {
    selectedEnclosId: number | null;
    filters: {
        type?: string;
        zone?: string;
        minCapacity?: number;
        maxCapacity?: number;
    };
    isLoading: boolean;
}

function createEnclosStore() {
    const initialState: EnclosState = {
        selectedEnclosId: null,
        filters: {},
        isLoading: false
    };

    const { subscribe, set, update } = writable<EnclosState>(initialState);

    return {
        subscribe,

        setSelectedEnclos: (id: number | null) => {
            update(state => ({ ...state, selectedEnclosId: id }));
        },

        setFilters: (filters: Partial<EnclosState['filters']>) => {
            update(state => ({
                ...state,
                filters: { ...state.filters, ...filters }
            }));
        },

        clearFilters: () => {
            update(state => ({ ...state, filters: {} }));
        },

        setLoading: (isLoading: boolean) => {
            update(state => ({ ...state, isLoading }));
        },

        reset: () => {
            set(initialState);
        }
    };
}

export const enclosStore = createEnclosStore();