// lib/api/pesee.ts
import { apiClient } from './client';
import type { PeseeCreate, PeseeUpdate, PeseeResponse } from '../types/pesee';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';

export const peseeApi = {
    getPesees: (params?: Partial<PaginationParams> & { animal_id?: number; lot_entomo_id?: number }) =>
        apiClient.get<PaginatedResponse<PeseeResponse>>('/pesees', { params }),

    getPesee: (id: number) =>
        apiClient.get<PeseeResponse>(`/pesees/${id}`),

    createPesee: (data: PeseeCreate) =>
        apiClient.post<PeseeResponse>('/pesees/create', data),

    updatePesee: (id: number, data: PeseeUpdate) =>
        apiClient.put<PeseeResponse>(`/pesees/update/${id}`, data),
};