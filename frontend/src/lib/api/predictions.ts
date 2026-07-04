// lib/api/predictions.ts
import { apiClient } from './client';
import type {
    PredictionRequest, PredictionResponse,
    GrowthPredictionResponse, ProductionPredictionResponse,
    CashflowPredictionResponse
} from '../types/predictions';

export const predictionsApi = {
    makePrediction: (data: PredictionRequest) =>
        apiClient.post<PredictionResponse>('/predictions', data),

    predictGrowth: (animal_id: number, horizon_jours?: number) =>
        apiClient.get<GrowthPredictionResponse>(`/predictions/growth/${animal_id}`, { params: { horizon_jours } }),

    predictProduction: (espece: string, race?: string, enclos_id?: number, horizon_jours?: number) =>
        apiClient.get<ProductionPredictionResponse>(`/predictions/production/${espece}`, { params: { race, enclos_id, horizon_jours } }),

    predictCashflow: (horizon_jours?: number) =>
        apiClient.get<CashflowPredictionResponse>('/predictions/cashflow', { params: { horizon_jours } }),

    predictHealthRisk: (espece: string, enclos_id?: number) =>
        apiClient.get(`/predictions/health/${espece}`, { params: { enclos_id } }),

    predictCompostMaturity: (compost_id: number) =>
        apiClient.get(`/predictions/compost/${compost_id}`),

    predictOvercrowding: (enclos_id: number, horizon_jours?: number) =>
        apiClient.get(`/predictions/overcrowding/${enclos_id}`, { params: { horizon_jours } }),

    predictWaterQuality: (enclos_id: number, hours_ahead?: number) =>
        apiClient.get(`/predictions/water-quality/${enclos_id}`, { params: { hours_ahead } })
};