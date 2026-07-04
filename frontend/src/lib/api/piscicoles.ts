// lib/api/piscicoles.ts
import { apiClient } from './client';
import type { PiscicoleCreate, PiscicoleUpdate, PiscicoleResponse } from '../types/piscicole';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';
import type { StatutAnimal } from '../types/animal';

export const piscicolesApi = {
    getPiscicoles: (params?: Partial<PaginationParams> & {
        race?: string;
        enclos_id?: number;
        statut?: StatutAnimal;
    }) =>
        apiClient.get<PaginatedResponse<PiscicoleResponse>>('/piscicoles', { params }),

    getPiscicole: (id: number) =>
        apiClient.get<PiscicoleResponse>(`/piscicoles/${id}`),

    createPiscicole: (data: PiscicoleCreate) =>
        apiClient.post<PiscicoleResponse>('/piscicoles', data),

    updatePiscicole: (id: number, data: PiscicoleUpdate) =>
        apiClient.put<PiscicoleResponse>(`/piscicoles/${id}`, data),

    deletePiscicole: (id: number, softDelete = true) =>
        apiClient.delete(`/piscicoles/${id}`, { params: { soft_delete: softDelete } }),

    getBiomass: (enclos_id: number) =>
        apiClient.get<{ enclos_id: number; biomass_kg: number }>(`/piscicoles/bassin/${enclos_id}/biomasse`),

    getStats: (enclos_id?: number) =>
        apiClient.get('/piscicoles/stats/global', { params: { enclos_id } })
};