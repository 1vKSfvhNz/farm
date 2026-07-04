// lib/api/alerts.ts
import { apiClient } from './client';
import type { AlertResponse } from '../types/alerts';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';

export const alertsApi = {
    getAlerts: (params?: Partial<PaginationParams> & { niveau?: string; est_lue?: boolean; espece?: string }) =>
        apiClient.get<PaginatedResponse<AlertResponse>>('/alerts', { params }),

    getUnreadCount: () =>
        apiClient.get<{ unread_count: number }>('/alerts/unread/count'),

    getAlert: (id: number) =>
        apiClient.get<AlertResponse>(`/alerts/${id}`),

    markAsRead: (id: number) =>
        apiClient.post(`/alerts/${id}/read`),

    resolveAlert: (id: number, resolution_note?: string) =>
        apiClient.post(`/alerts/${id}/resolve`, null, { params: { resolution_note } }),

    markAllAsRead: () =>
        apiClient.post('/alerts/mark-all-read'),

    generateVaccinationAlerts: () =>
        apiClient.post('/alerts/generate/vaccination'),

    generateWaterQualityAlerts: () =>
        apiClient.post('/alerts/generate/water-quality')
};