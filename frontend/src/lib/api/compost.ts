// lib/api/compost.ts
import { apiClient } from './client';
import type {
    CompostCreate, CompostUpdate, CompostResponse,
    RetournementCompostCreate, RetournementCompostResponse
} from '../types/compost';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';

export const compostApi = {
    getComposts: (params?: Partial<PaginationParams> & { compost_type?: string; is_mature?: boolean }) =>
        apiClient.get<PaginatedResponse<CompostResponse>>('/compost', { params }),

    getCompost: (id: number) =>
        apiClient.get<CompostResponse>(`/compost/${id}`),

    getCompostStatus: (id: number, temperature?: number, humidity?: number) =>
        apiClient.get(`/compost/${id}/status`, { params: { temperature, humidity } }),

    createCompost: (data: CompostCreate) =>
        apiClient.post<CompostResponse>('/compost', data),

    updateCompost: (id: number, data: CompostUpdate) =>
        apiClient.put<CompostResponse>(`/compost/${id}`, data),

    deleteCompost: (id: number) =>
        apiClient.delete(`/compost/${id}`),

    getTurnings: (compost_id: number, params?: Partial<PaginationParams>) =>
        apiClient.get<PaginatedResponse<RetournementCompostResponse>>(`/compost/${compost_id}/retournements`, { params }),

    addTurning: (compost_id: number, data: RetournementCompostCreate) =>
        apiClient.post<RetournementCompostResponse>(`/compost/${compost_id}/retournements`, data),

    markAsMature: (compost_id: number, volume_final?: number) =>
        apiClient.post(`/compost/${compost_id}/mature`, null, { params: { volume_final } }),

    getStats: () =>
        apiClient.get('/compost/stats/global')
};