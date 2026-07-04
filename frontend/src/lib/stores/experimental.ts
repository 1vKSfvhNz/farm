// lib/stores/experimental.ts - Version corrigée
import { writable } from 'svelte/store';
import { experimentalApi } from '../api/experimental';
import type {
    ExperimentalModeResponse,
    ConfidenceResponse,
    ReferenceHypothesisResponse
} from '../types/experimental';

interface CollectionStats {
    totalDonnees: number;
    parEspece: Record<string, number>;
    dernieresCollectes: Array<{ espece: string; date: string; quantite: number }>;
}

interface ExperimentalState {
    modeStatus: ExperimentalModeResponse | null;
    confidenceLevels: Map<string, ConfidenceResponse>;
    hypotheses: ReferenceHypothesisResponse[];
    selectedHypothesis: ReferenceHypothesisResponse | null;
    selectedEspece: string | null;
    selectedPredictionType: string | null;
    isLoading: boolean;
    isGenerating: boolean;
    collectionStats: CollectionStats | null;
}

function createExperimentalStore() {
    const initialState: ExperimentalState = {
        modeStatus: null,
        confidenceLevels: new Map(),
        hypotheses: [],
        selectedHypothesis: null,
        selectedEspece: null,
        selectedPredictionType: null,
        isLoading: false,
        isGenerating: false,
        collectionStats: null
    };

    const { subscribe, set, update } = writable<ExperimentalState>(initialState);

    // Helper pour obtenir l'état courant
    const getCurrentState = (): ExperimentalState => {
        let currentState: ExperimentalState = initialState;
        const unsubscribe = subscribe(s => { currentState = s; });
        unsubscribe();
        return currentState;
    };

    return {
        subscribe,

        loadModeStatus: async (espece?: string) => {
            update(state => ({ ...state, isLoading: true }));
            try {
                const status = await experimentalApi.getExperimentalStatus(espece);
                update(state => ({ ...state, modeStatus: status, isLoading: false }));
            } catch (error) {
                console.error('Failed to load experimental mode status:', error);
                update(state => ({ ...state, isLoading: false }));
            }
        },

        loadConfidence: async (espece: string, predictionType: string) => {
            update(state => ({ ...state, isLoading: true }));
            try {
                const confidence = await experimentalApi.getConfidence(espece, predictionType);
                update(state => {
                    const confidenceLevels = new Map(state.confidenceLevels);
                    confidenceLevels.set(`${espece}:${predictionType}`, confidence);
                    return { ...state, confidenceLevels, isLoading: false };
                });
            } catch (error) {
                console.error('Failed to load confidence:', error);
                update(state => ({ ...state, isLoading: false }));
            }
        },

        loadHypotheses: async (espece?: string, validee?: boolean) => {
            update(state => ({ ...state, isLoading: true }));
            try {
                const hypotheses = await experimentalApi.getHypotheses(espece, validee);
                update(state => ({ ...state, hypotheses, isLoading: false }));
            } catch (error) {
                console.error('Failed to load hypotheses:', error);
                update(state => ({ ...state, isLoading: false }));
            }
        },

        loadCollectionStats: async () => {
            update(state => ({ ...state, isLoading: true }));
            try {
                const stats = await experimentalApi.getCollectionStats();
                update(state => ({ ...state, collectionStats: stats as CollectionStats, isLoading: false }));
            } catch (error) {
                console.error('Failed to load collection stats:', error);
                update(state => ({ ...state, isLoading: false }));
            }
        },

        generateReference: async (espece: string, forceRegenerate: boolean = false) => {
            update(state => ({ ...state, isGenerating: true }));
            try {
                const result = await experimentalApi.generateReference({ espece, force_regenerate: forceRegenerate });
                update(state => ({ ...state, isGenerating: false }));
                return result;
            } catch (error) {
                console.error('Failed to generate reference:', error);
                update(state => ({ ...state, isGenerating: false }));
                throw error;
            }
        },

        createHypothesis: async (data: any) => {
            update(state => ({ ...state, isLoading: true }));
            try {
                const hypothesis = await experimentalApi.createHypothesis(data);
                update(state => ({
                    ...state,
                    hypotheses: [hypothesis, ...state.hypotheses],
                    isLoading: false
                }));
                return hypothesis;
            } catch (error) {
                console.error('Failed to create hypothesis:', error);
                update(state => ({ ...state, isLoading: false }));
                throw error;
            }
        },

        validateHypothesis: async (hypothesisId: number) => {
            update(state => ({ ...state, isLoading: true }));
            try {
                await experimentalApi.validateHypothesis(hypothesisId);
                update(state => ({
                    ...state,
                    hypotheses: state.hypotheses.map(h =>
                        h.id === hypothesisId ? { ...h, validee: true } : h
                    ),
                    isLoading: false
                }));
            } catch (error) {
                console.error('Failed to validate hypothesis:', error);
                update(state => ({ ...state, isLoading: false }));
            }
        },

        setSelectedEspece: (espece: string | null) => {
            update(state => ({ ...state, selectedEspece: espece }));
        },

        setSelectedPredictionType: (predictionType: string | null) => {
            update(state => ({ ...state, selectedPredictionType: predictionType }));
        },

        setSelectedHypothesis: (hypothesis: ReferenceHypothesisResponse | null) => {
            update(state => ({ ...state, selectedHypothesis: hypothesis }));
        },

        getConfidenceForCurrent: async () => {
            const currentState = getCurrentState();
            if (currentState.selectedEspece && currentState.selectedPredictionType) {
                return await experimentalApi.getConfidence(currentState.selectedEspece, currentState.selectedPredictionType);
            }
            return null;
        },

        reset: () => {
            set(initialState);
        }
    };
}

export const experimentalStore = createExperimentalStore();