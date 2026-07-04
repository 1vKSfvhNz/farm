// lib/api/experimental.ts
import { apiClient } from './client';
import type {
    ReferenceHypothesisCreate, ReferenceHypothesisResponse,
    ExperimentalModeResponse, ConfidenceResponse,
    ReferenceGenerationRequest, ReferenceGenerationResponse
} from '../types/experimental';

export const experimentalApi = {
    getExperimentalStatus: (espece?: string) =>
        apiClient.get<ExperimentalModeResponse>('/experimental/status', { params: { espece } }),

    getConfidence: (espece: string, prediction_type: string) =>
        apiClient.get<ConfidenceResponse>(`/experimental/confidence/${espece}/${prediction_type}`),

    generateReference: (data: ReferenceGenerationRequest) =>
        apiClient.post<ReferenceGenerationResponse>('/experimental/references/generate', data),

    createHypothesis: (data: ReferenceHypothesisCreate) =>
        apiClient.post<ReferenceHypothesisResponse>('/experimental/hypotheses', data),

    getHypotheses: (espece?: string, validee?: boolean) =>
        apiClient.get<ReferenceHypothesisResponse[]>('/experimental/hypotheses', { params: { espece, validee } }),

    validateHypothesis: (hypothesis_id: number) =>
        apiClient.put(`/experimental/hypotheses/${hypothesis_id}/validate`),

    getCollectionStats: () =>
        apiClient.get('/experimental/data/collect-stats')
};