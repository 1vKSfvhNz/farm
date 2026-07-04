// lib/stores/reports.ts
import { writable } from 'svelte/store';

interface ReportState {
    selectedReport: string | null;
    filters: {
        startDate: string | null;
        endDate: string | null;
        species: string | null;
        enclosId: number | null;
        format: 'csv' | 'pdf';
    };
    isGenerating: boolean;
    downloadUrl: string | null;
}

function createReportStore() {
    const initialState: ReportState = {
        selectedReport: null,
        filters: {
            startDate: null,
            endDate: null,
            species: null,
            enclosId: null,
            format: 'csv'
        },
        isGenerating: false,
        downloadUrl: null
    };

    const { subscribe, set, update } = writable<ReportState>(initialState);

    return {
        subscribe,

        setSelectedReport: (report: string | null) => {
            update(state => ({ ...state, selectedReport: report }));
        },

        setDateRange: (startDate: string | null, endDate: string | null) => {
            update(state => ({
                ...state,
                filters: { ...state.filters, startDate, endDate }
            }));
        },

        setSpecies: (species: string | null) => {
            update(state => ({
                ...state,
                filters: { ...state.filters, species }
            }));
        },

        setEnclosId: (enclosId: number | null) => {
            update(state => ({
                ...state,
                filters: { ...state.filters, enclosId }
            }));
        },

        setFormat: (format: 'csv' | 'pdf') => {
            update(state => ({
                ...state,
                filters: { ...state.filters, format }
            }));
        },

        setGenerating: (isGenerating: boolean) => {
            update(state => ({ ...state, isGenerating }));
        },

        setDownloadUrl: (url: string | null) => {
            update(state => ({ ...state, downloadUrl: url }));
        },

        clearFilters: () => {
            update(state => ({
                ...state,
                filters: initialState.filters
            }));
        },

        reset: () => {
            set(initialState);
        }
    };
}

export const reportStore = createReportStore();