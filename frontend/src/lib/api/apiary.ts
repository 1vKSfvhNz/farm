// lib/api/apiary.ts (corrigé - simplifié)
import { apiClient } from './client';
import type { PaginationParams } from '../types/pagination';
import type { RecolteMielResponse, ReineResponse, RucheCreate, RucheResponse, RucheUpdate } from '$lib/types';

export const apiaryApi = {
    // Ruches
    getRuches: (params?: Partial<PaginationParams> & { statut?: string; emplacement?: string }) =>
        apiClient.get<{ items: RucheResponse[]; total: number; skip: number; limit: number }>('/apiary/ruches', { params }),

    getRuche: (id: number) =>
        apiClient.get<RucheResponse>(`/apiary/ruches/${id}`),

    getRucheByIdentification: (identification: string) =>
        apiClient.get<RucheResponse>(`/apiary/ruches/identification/${identification}`),

    createRuche: (data: RucheCreate) =>
        apiClient.post<RucheResponse>('/apiary/ruches', data),

    updateRuche: (id: number, data: RucheUpdate) =>
        apiClient.put<RucheResponse>(`/apiary/ruches/${id}`, data),

    deleteRuche: (id: number, softDelete = true) =>
        apiClient.delete(`/apiary/ruches/${id}`, { params: { soft_delete: softDelete } }),

    // Reines
    getReines: (ruche_id: number) =>
        apiClient.get<ReineResponse[]>(`/apiary/ruches/${ruche_id}/reines`),

    getReine: (id: number) =>
        apiClient.get<ReineResponse>(`/apiary/reines/${id}`),

    addReine: (ruche_id: number, data: any) =>
        apiClient.post<ReineResponse>(`/apiary/ruches/${ruche_id}/reines`, data),

    updateReine: (id: number, data: any) =>
        apiClient.put<ReineResponse>(`/apiary/reines/${id}`, data),

    // Récoltes
    getRecoltes: (ruche_id: number, params?: Partial<PaginationParams> & { year?: number }) =>
        apiClient.get<{ items: RecolteMielResponse[]; total: number; skip: number; limit: number }>(`/apiary/ruches/${ruche_id}/recoltes`, { params }),

    getRecolte: (id: number) =>
        apiClient.get<RecolteMielResponse>(`/apiary/recoltes/${id}`),

    addRecolte: (ruche_id: number, data: any) =>
        apiClient.post<RecolteMielResponse>(`/apiary/ruches/${ruche_id}/recoltes`, data),

    updateRecolte: (id: number, data: any) =>
        apiClient.put<RecolteMielResponse>(`/apiary/recoltes/${id}`, data),

    deleteRecolte: (id: number) =>
        apiClient.delete(`/apiary/recoltes/${id}`),

    // Statistiques
    getProductionStats: (year?: number) =>
        apiClient.get('/apiary/stats/production', { params: { year } }),

    getRuchesStats: () =>
        apiClient.get('/apiary/stats/ruches'),

    getReinesStats: () =>
        apiClient.get('/apiary/stats/reines'),

    getDashboard: () =>
        apiClient.get('/apiary/dashboard'),

    getAlerts: () =>
        apiClient.get('/apiary/alerts'),

    getProductionEvolution: (years?: number) =>
        apiClient.get('/apiary/production/evolution', { params: { years } }),

    // Inspections
    addInspection: (ruche_id: number, data: any) =>
        apiClient.post(`/apiary/ruches/${ruche_id}/inspections`, data),

    getInspections: (ruche_id: number, limit?: number) =>
        apiClient.get(`/apiary/ruches/${ruche_id}/inspections`, { params: { limit } }),

    // Essaimage
    recordSwarming: (ruche_id: number, data: any) =>
        apiClient.post(`/apiary/ruches/${ruche_id}/essaimage`, data),

    getSwarmingHistory: (ruche_id: number) =>
        apiClient.get(`/apiary/ruches/${ruche_id}/historique/essaimages`)
};