// lib/api/exports.ts - Version corrigée
import { apiClient } from './client';

export const exportsApi = {
    exportAnimalsCSV: (espece?: string, enclos_id?: number) =>
        apiClient.get<Blob>('/exports/animals/csv', {
            params: { espece, enclos_id },
            responseType: 'blob'
        }),

    exportFinancialCSV: (start_date: string, end_date: string) =>
        apiClient.get<Blob>('/exports/financial/csv', {
            params: { start_date, end_date },
            responseType: 'blob'
        }),

    exportWeighingsCSV: (espece?: string, start_date?: string, end_date?: string) =>
        apiClient.get<Blob>('/exports/pesees/csv', {
            params: { espece, start_date, end_date },
            responseType: 'blob'
        }),

    exportMortalityCSV: (espece?: string, start_date?: string, end_date?: string) =>
        apiClient.get<Blob>('/exports/mortality/csv', {
            params: { espece, start_date, end_date },
            responseType: 'blob'
        }),

    exportVaccinationsCSV: (espece?: string, start_date?: string, end_date?: string) =>
        apiClient.get<Blob>('/exports/vaccinations/csv', {
            params: { espece, start_date, end_date },
            responseType: 'blob'
        })
};