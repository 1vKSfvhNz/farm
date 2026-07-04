// lib/api/bea.ts
import { apiClient } from './client';
import type { BienEtreIndiceCreate, BienEtreIndiceResponse, BienEtreCritereResponse } from '../types/bea';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';

export const beaApi = {
    getLatestIndex: (enclos_id: number) =>
        apiClient.get<BienEtreIndiceResponse>(`/bea/indices/${enclos_id}/latest`),

    getIndices: (enclos_id: number, params?: Partial<PaginationParams> & { start_date?: string; end_date?: string }) =>
        apiClient.get<PaginatedResponse<BienEtreIndiceResponse>>(`/bea/indices/${enclos_id}`, { params }),

    createIndex: (data: BienEtreIndiceCreate) =>
        apiClient.post<BienEtreIndiceResponse>('/bea/indices', data),

    getCriteres: () =>
        apiClient.get<BienEtreCritereResponse[]>('/bea/criteres'),

    getDashboard: (enclos_id: number) =>
        apiClient.get(`/bea/dashboard/${enclos_id}`)
};