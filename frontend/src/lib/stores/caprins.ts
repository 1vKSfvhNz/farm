// $lib/stores/caprins.ts
import { writable } from 'svelte/store';
import type { CaprinResponse } from '$lib/types/caprin';
import type { AnimauxFilters } from '$lib/types/animal';
import { defaultFilters } from './animal';

const STORAGE_KEY = 'caprin_detail';
const FILTERS_STORAGE_KEY = 'caprins_filters';

function createCaprinsStore() {
    const stored = typeof window !== 'undefined' ? sessionStorage.getItem(STORAGE_KEY) : null;
    const initialValue: CaprinResponse | null = stored ? JSON.parse(stored) : null;
    
    const { subscribe, set, update } = writable<CaprinResponse | null>(initialValue);

    // Store pour les filtres - Utiliser sessionStorage
    const filtersStored = typeof window !== 'undefined' ? sessionStorage.getItem(FILTERS_STORAGE_KEY) : null;
    const initialFilters: AnimauxFilters = filtersStored ? JSON.parse(filtersStored) : defaultFilters;
    
    const filters = writable<AnimauxFilters>(initialFilters);

    return {
        subscribe,
        set,
        update,
        
        // Store des filtres
        filters: {
            subscribe: filters.subscribe,
            set: (newFilters: AnimauxFilters) => {
                filters.set(newFilters);
                if (typeof window !== 'undefined') {
                    sessionStorage.setItem(FILTERS_STORAGE_KEY, JSON.stringify(newFilters));
                }
            },
            update: (fn: (filters: AnimauxFilters) => AnimauxFilters) => {
                filters.update(current => {
                    const newFilters = fn(current);
                    if (typeof window !== 'undefined') {
                        sessionStorage.setItem(FILTERS_STORAGE_KEY, JSON.stringify(newFilters));
                    }
                    return newFilters;
                });
            },
            reset: () => {
                filters.set(defaultFilters);
                if (typeof window !== 'undefined') {
                    sessionStorage.removeItem(FILTERS_STORAGE_KEY);
                }
            },
            get: (): AnimauxFilters => {
                let result: AnimauxFilters = defaultFilters;
                const unsubscribe = filters.subscribe(value => {
                    result = value;
                });
                unsubscribe();
                return result;
            }
        },
        
        // Méthodes pour le caprin
        setCaprin: (caprin: CaprinResponse) => {
            set(caprin);
            if (typeof window !== 'undefined') {
                sessionStorage.setItem(STORAGE_KEY, JSON.stringify(caprin));
            }
        },
        
        updateCaprin: (updatedCaprin: CaprinResponse) => {
            set(updatedCaprin);
            if (typeof window !== 'undefined') {
                sessionStorage.setItem(STORAGE_KEY, JSON.stringify(updatedCaprin));
            }
        },
        
        getCaprin: (): CaprinResponse | null => {
            let result: CaprinResponse | null = null;
            const unsubscribe = subscribe(value => {
                result = value;
            });
            unsubscribe();
            return result;
        },
        
        clear: () => {
            set(null);
            if (typeof window !== 'undefined') {
                sessionStorage.removeItem(STORAGE_KEY);
            }
        },
        
        // Nettoyer tout (caprin + filtres)
        clearAll: () => {
            set(null);
            filters.set(defaultFilters);
            if (typeof window !== 'undefined') {
                sessionStorage.removeItem(STORAGE_KEY);
                sessionStorage.removeItem(FILTERS_STORAGE_KEY);
            }
        }
    };
}

export const caprinsStore = createCaprinsStore();