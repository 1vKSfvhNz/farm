// lib/api/vaccination.ts
import { apiClient } from './client';
import type {
    VaccinationCreate, VaccinationUpdate, VaccinationResponse,
    MaladieCreate, MaladieResponse,
    VaccinCreate, VaccinResponse
} from '../types/vaccination';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';

export const vaccinationApi = {
    // Vaccinations
    getVaccinations: (params?: Partial<PaginationParams> & { animal_id?: number; maladie_id?: number; realisee?: boolean }) =>
        apiClient.get<PaginatedResponse<VaccinationResponse>>('/vaccination', { params }),

    getUpcomingVaccinations: (days_ahead?: number) =>
        apiClient.get<{ count: number; vaccinations: VaccinationResponse[] }>('/vaccination/upcoming', { params: { days_ahead } }),

    getOverdueVaccinations: () =>
        apiClient.get<{ count: number; vaccinations: VaccinationResponse[] }>('/vaccination/overdue'),

    getVaccination: (id: number) =>
        apiClient.get<VaccinationResponse>(`/vaccination/${id}`),

    createVaccination: (data: VaccinationCreate) =>
        apiClient.post<VaccinationResponse>('/vaccination', data),

    updateVaccination: (id: number, data: VaccinationUpdate) =>
        apiClient.put<VaccinationResponse>(`/vaccination/${id}`, data),

    realizeVaccination: (id: number, date_realisee?: string) =>
        apiClient.post(`/vaccination/${id}/realize`, null, { params: { date_realisee } }),

    deleteVaccination: (id: number) =>
        apiClient.delete(`/vaccination/${id}`),

    // Maladies
    getMaladies: () =>
        apiClient.get<MaladieResponse[]>('/vaccination/maladies'),

    createMaladie: (data: MaladieCreate) =>
        apiClient.post<MaladieResponse>('/vaccination/maladies', data),

    // Vaccins
    getVaccins: () =>
        apiClient.get<VaccinResponse[]>('/vaccination/vaccins'),

    createVaccin: (data: VaccinCreate) =>
        apiClient.post<VaccinResponse>('/vaccination/vaccins', data)
};