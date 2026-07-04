// lib/api/water_quality.ts
import { apiClient } from './client';
import type { WaterQualityCreate, WaterQualityResponse, WaterQualityAlerteResponse } from '../types/water_quality';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';

export const waterQualityApi = {
    createMeasurement: (data: WaterQualityCreate) =>
        apiClient.post<WaterQualityResponse>('/water-quality/measurements', data),

    getLatestMeasurement: (enclos_id: number) =>
        apiClient.get<WaterQualityResponse>(`/water-quality/measurements/${enclos_id}/latest`),

    getMeasurements: (enclos_id: number, params?: Partial<PaginationParams> & { start_date?: string; end_date?: string }) =>
        apiClient.get<PaginatedResponse<WaterQualityResponse>>(`/water-quality/measurements/${enclos_id}`, { params }),

    getAlerts: (params?: Partial<PaginationParams> & { enclos_id?: number; traitee?: boolean }) =>
        apiClient.get<PaginatedResponse<WaterQualityAlerteResponse>>('/water-quality/alerts', { params }),

    resolveAlert: (alert_id: number) =>
        apiClient.post(`/water-quality/alerts/${alert_id}/resolve`)
};