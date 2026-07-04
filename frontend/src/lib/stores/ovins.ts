// $lib/stores/ovins.ts
import { writable } from 'svelte/store';
import type { OvinResponse } from '$lib/types/ovin';
import type { AnimauxFilters } from '$lib/types/animal';
import { defaultFilters } from './animal';

const STORAGE_KEY = 'ovin_detail';
const FILTERS_STORAGE_KEY = 'ovins_filters';

function createOvinsStore() {
    const stored = typeof window !== 'undefined' ? sessionStorage.getItem(STORAGE_KEY) : null;
    const initialValue: OvinResponse | null = stored ? JSON.parse(stored) : null;
    
    const { subscribe, set, update } = writable<OvinResponse | null>(initialValue);

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
        
        // Méthodes pour l'ovin
        setOvin: (ovin: OvinResponse) => {
            set(ovin);
            if (typeof window !== 'undefined') {
                sessionStorage.setItem(STORAGE_KEY, JSON.stringify(ovin));
            }
        },
        
        updateOvin: (updatedOvin: OvinResponse) => {
            set(updatedOvin);
            if (typeof window !== 'undefined') {
                sessionStorage.setItem(STORAGE_KEY, JSON.stringify(updatedOvin));
            }
        },
        
        getOvin: (): OvinResponse | null => {
            let result: OvinResponse | null = null;
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
        
        // Nettoyer tout (ovin + filtres)
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

export const ovinsStore = createOvinsStore();