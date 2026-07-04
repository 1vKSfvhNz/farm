// lib/api/dashboard.ts
import { apiClient } from './client';
import type { DashboardData, HealthStatus } from '../types/dashboard';

export const dashboardApi = {
    getDashboard: () =>
        apiClient.get<DashboardData>('/dashboard'),

    getAnimalsSummary: () =>
        apiClient.get('/dashboard/animals'),

    getProductionSummary: (days?: number) =>
        apiClient.get('/dashboard/production', { params: { days } }),

    getFinancialSummary: () =>
        apiClient.get('/dashboard/financial'),

    getRecentAlerts: (limit?: number) =>
        apiClient.get<{ alerts: any[] }>('/dashboard/alerts', { params: { limit } }),

    getHealthStatus: () =>
        apiClient.get<HealthStatus>('/dashboard/health'),

    getWaterQualitySummary: () =>
        apiClient.get('/dashboard/water-quality'),

    getCompostSummary: () =>
        apiClient.get('/dashboard/compost')
};