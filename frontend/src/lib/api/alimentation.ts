// lib/api/alimentation.ts
import { apiClient } from './client';
import type {
    AlimentationCreate, AlimentationUpdate, AlimentationResponse,
    RationAlimentaireCreate, RationAlimentaireResponse
} from '../types/alimentation';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';

export const alimentationApi = {
    getAlimentations: (params?: Partial<PaginationParams> & { animal_id?: number; lot_entomo_id?: number }) =>
        apiClient.get<PaginatedResponse<AlimentationResponse>>('/alimentation', { params }),

    getAlimentation: (id: number) =>
        apiClient.get<AlimentationResponse>(`/alimentation/${id}`),

    createAlimentation: (data: AlimentationCreate) =>
        apiClient.post<AlimentationResponse>('/alimentation', data),

    updateAlimentation: (id: number, data: AlimentationUpdate) =>
        apiClient.put<AlimentationResponse>(`/alimentation/${id}`, data),

    deleteAlimentation: (id: number) =>
        apiClient.delete(`/alimentation/${id}`),

    getRations: (params?: { espece?: string }) =>
        apiClient.get<RationAlimentaireResponse[]>('/alimentation/rations', { params }),

    getRation: (id: number) =>
        apiClient.get<RationAlimentaireResponse>(`/alimentation/rations/${id}`),

    createRation: (data: RationAlimentaireCreate) =>
        apiClient.post<RationAlimentaireResponse>('/alimentation/rations', data),

    updateRation: (id: number, data: RationAlimentaireCreate) =>
        apiClient.put<RationAlimentaireResponse>(`/alimentation/rations/${id}`, data),

    deleteRation: (id: number) =>
        apiClient.delete(`/alimentation/rations/${id}`)
};