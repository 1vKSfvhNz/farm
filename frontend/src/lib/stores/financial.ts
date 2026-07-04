// lib/stores/financial.ts
import { writable } from 'svelte/store';

interface FinancialState {
    dateRange: {
        startDate: string | null;
        endDate: string | null;
    };
    selectedCategories: {
        depenses: string[];
        recettes: string[];
    };
    isLoading: boolean;
}

function createFinancialStore() {
    const initialState: FinancialState = {
        dateRange: {
            startDate: null,
            endDate: null
        },
        selectedCategories: {
            depenses: [],
            recettes: []
        },
        isLoading: false
    };

    const { subscribe, set, update } = writable<FinancialState>(initialState);

    return {
        subscribe,

        setDateRange: (startDate: string | null, endDate: string | null) => {
            update(state => ({
                ...state,
                dateRange: { startDate, endDate }
            }));
        },

        setDepenseCategories: (categories: string[]) => {
            update(state => ({
                ...state,
                selectedCategories: { ...state.selectedCategories, depenses: categories }
            }));
        },

        setRecetteCategories: (categories: string[]) => {
            update(state => ({
                ...state,
                selectedCategories: { ...state.selectedCategories, recettes: categories }
            }));
        },

        toggleDepenseCategory: (category: string) => {
            update(state => {
                const current = state.selectedCategories.depenses;
                const updated = current.includes(category)
                    ? current.filter(c => c !== category)
                    : [...current, category];
                return {
                    ...state,
                    selectedCategories: { ...state.selectedCategories, depenses: updated }
                };
            });
        },

        toggleRecetteCategory: (category: string) => {
            update(state => {
                const current = state.selectedCategories.recettes;
                const updated = current.includes(category)
                    ? current.filter(c => c !== category)
                    : [...current, category];
                return {
                    ...state,
                    selectedCategories: { ...state.selectedCategories, recettes: updated }
                };
            });
        },

        clearCategories: () => {
            update(state => ({
                ...state,
                selectedCategories: { depenses: [], recettes: [] }
            }));
        },

        setLoading: (isLoading: boolean) => {
            update(state => ({ ...state, isLoading }));
        },

        reset: () => {
            set(initialState);
        }
    };
}

export const financialStore = createFinancialStore();