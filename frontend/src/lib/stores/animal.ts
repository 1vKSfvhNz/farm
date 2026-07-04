// lib/stores/animal.ts
import type { AnimauxFilters } from '$lib/types';
import { writable } from 'svelte/store';

interface AnimalState {
    selectedAnimalId: number | null;
    selectedSpecies: string | null;
    filters: {
        race?: string;
        enclos_id?: number;
        statut?: string;
        sexe?: string;
        search?: string;
    };
    isLoading: boolean;
}

// Filtres par défaut
export const defaultFilters: AnimauxFilters = {
    searchQuery: "",
    selectedRace: "",
    selectedEnclos: "",
    selectedStatuts: ["vivant"],
    selectedSexes: []
};

export function getStatutBadge(statut: string) {
    switch (statut) {
        case "vivant":
            return { niveau: "info" as const, label: "Vivant" };
        case "vendu":
            return { niveau: "warning" as const, label: "Vendu" };
        case "decede":
            return { niveau: "critical" as const, label: "Décédé" };
        case "transfere":
            return { niveau: "warning" as const, label: "Transféré" };
        default:
            return { niveau: "info" as const, label: statut };
    }
}

export function calculateAge(dateNaissance?: string): string {
    if (!dateNaissance) return "Non renseigné";
    const birth = new Date(dateNaissance);
    const today = new Date();
    const ageInDays = Math.floor((today.getTime() - birth.getTime()) / (1000 * 60 * 60 * 24));
    if (ageInDays < 30) return `${ageInDays} jours`;
    if (ageInDays < 365) return `${Math.floor(ageInDays / 30)} mois`;
    const years = Math.floor(ageInDays / 365);
    const months = Math.floor((ageInDays % 365) / 30);
    return months > 0 ? `${years} an(s) et ${months} mois` : `${years} an(s)`;
}

function createAnimalStore() {
    const initialState: AnimalState = {
        selectedAnimalId: null,
        selectedSpecies: null,
        filters: {},
        isLoading: false
    };

    const { subscribe, set, update } = writable<AnimalState>(initialState);

    return {
        subscribe,

        setSelectedAnimal: (id: number | null, species: string | null = null) => {
            update(state => ({
                ...state,
                selectedAnimalId: id,
                selectedSpecies: species
            }));
        },

        setFilters: (filters: Partial<AnimalState['filters']>) => {
            update(state => ({
                ...state,
                filters: { ...state.filters, ...filters }
            }));
        },

        clearFilters: () => {
            update(state => ({
                ...state,
                filters: {}
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

export const animalStore = createAnimalStore();