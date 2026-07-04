// lib/api/avicoles.ts
import { apiClient } from './client';
import type { AvicoleCreate, AvicoleUpdate, AvicoleResponse } from '../types/avicole';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';
import type { StatutAnimal } from '../types/animal';

export const avicolesApi = {
    getAvicoles: (params?: Partial<PaginationParams> & {
        race?: string;
        enclos_id?: number;
        production_type?: string;
        statut?: StatutAnimal;
    }) =>
        apiClient.get<PaginatedResponse<AvicoleResponse>>('/avicoles', { params }),

    getAvicole: (id: number) =>
        apiClient.get<AvicoleResponse>(`/avicoles/${id}`),

    createAvicole: (data: AvicoleCreate) =>
        apiClient.post<AvicoleResponse>('/avicoles', data),

    updateAvicole: (id: number, data: AvicoleUpdate) =>
        apiClient.put<AvicoleResponse>(`/avicoles/${id}`, data),

    addEggProduction: (id: number, egg_count: number, egg_weight_grams?: number) =>
        apiClient.post(`/avicoles/${id}/oeufs`, { egg_count, egg_weight_grams }),

    getEggStats: (enclos_id?: number, days?: number) =>
        apiClient.get('/avicoles/production/oeufs/stats', { params: { enclos_id, days } }),

    getStats: (enclos_id?: number) =>
        apiClient.get('/avicoles/stats/global', { params: { enclos_id } })
};