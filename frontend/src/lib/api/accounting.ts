// lib/api/accounting.ts
import { apiClient } from './client';
import type {
    DepenseCreate, DepenseUpdate, DepenseResponse,
    RecetteCreate, RecetteUpdate, RecetteResponse,
    ComptabiliteSummary
} from '../types/accounting';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';

export const accountingApi = {
    getSummary: (start_date?: string, end_date?: string) =>
        apiClient.get<ComptabiliteSummary>('/accounting/summary', { params: { start_date, end_date } }),

    getDepenses: (params?: Partial<PaginationParams> & { start_date?: string; end_date?: string; categorie?: string }) =>
        apiClient.get<PaginatedResponse<DepenseResponse>>('/accounting/depenses', { params }),

    getRecettes: (params?: Partial<PaginationParams> & { start_date?: string; end_date?: string; categorie?: string }) =>
        apiClient.get<PaginatedResponse<RecetteResponse>>('/accounting/recettes', { params }),

    createDepense: (data: DepenseCreate) =>
        apiClient.post<DepenseResponse>('/accounting/depenses', data),

    createRecette: (data: RecetteCreate) =>
        apiClient.post<RecetteResponse>('/accounting/recettes', data),

    updateDepense: (id: number, data: DepenseUpdate) =>
        apiClient.put<DepenseResponse>(`/accounting/depenses/${id}`, data),

    updateRecette: (id: number, data: RecetteUpdate) =>
        apiClient.put<RecetteResponse>(`/accounting/recettes/${id}`, data),

    deleteDepense: (id: number) =>
        apiClient.delete(`/accounting/depenses/${id}`),

    deleteRecette: (id: number) =>
        apiClient.delete(`/accounting/recettes/${id}`),

    getProfitabilityBySpecies: (start_date?: string, end_date?: string) =>
        apiClient.get('/accounting/profitability/species', { params: { start_date, end_date } })
};