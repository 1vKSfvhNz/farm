// $lib/stores/bovins.ts
import { writable } from 'svelte/store';
import type { BovinResponse } from '$lib/types/bovin';
import type { AnimauxFilters } from '$lib/types/animal';
import { defaultFilters } from './animal';

const STORAGE_KEY = 'bovin_detail';
const FILTERS_STORAGE_KEY = 'bovins_filters';


function createBovinsStore() {
    const stored = typeof window !== 'undefined' ? sessionStorage.getItem(STORAGE_KEY) : null;
    const initialValue: BovinResponse | null = stored ? JSON.parse(stored) : null;
    
    const { subscribe, set, update } = writable<BovinResponse | null>(initialValue);

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
        
        // Méthodes pour le bovin
        setBovin: (bovin: BovinResponse) => {
            set(bovin);
            if (typeof window !== 'undefined') {
                sessionStorage.setItem(STORAGE_KEY, JSON.stringify(bovin));
            }
        },
        
        updateBovin: (updatedBovin: BovinResponse) => {
            set(updatedBovin);
            if (typeof window !== 'undefined') {
                sessionStorage.setItem(STORAGE_KEY, JSON.stringify(updatedBovin));
            }
        },
        
        getBovin: (): BovinResponse | null => {
            let result: BovinResponse | null = null;
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
        
        // Nettoyer tout (bovin + filtres)
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

export const bovinsStore = createBovinsStore();