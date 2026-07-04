// lib/api/entomoculture.ts
import { apiClient } from './client';
import type {
    EntomocultureLotCreate, EntomocultureLotUpdate, EntomocultureLotResponse,
    EntomocultureCycleCreate, EntomocultureCycleResponse
} from '../types/entomoculture';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';

export const entomocultureApi = {
    getLots: (params?: Partial<PaginationParams> & { espece?: string; enclos_id?: number }) =>
        apiClient.get<PaginatedResponse<EntomocultureLotResponse>>('/entomoculture/lots', { params }),

    getLot: (id: number) =>
        apiClient.get<EntomocultureLotResponse>(`/entomoculture/lots/${id}`),

    createLot: (data: EntomocultureLotCreate) =>
        apiClient.post<EntomocultureLotResponse>('/entomoculture/lots', data),

    updateLot: (id: number, data: EntomocultureLotUpdate) =>
        apiClient.put<EntomocultureLotResponse>(`/entomoculture/lots/${id}`, data),

    getCycles: (lot_id: number) =>
        apiClient.get<EntomocultureCycleResponse[]>(`/entomoculture/lots/${lot_id}/cycles`),

    addCycle: (data: EntomocultureCycleCreate) =>
        apiClient.post<EntomocultureCycleResponse>('/entomoculture/cycles', data),

    getStats: () =>
        apiClient.get('/entomoculture/stats/global')
};