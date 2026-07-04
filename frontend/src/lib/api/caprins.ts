// lib/api/caprins.ts
import { apiClient } from './client';
import type { PeseeCreate, PeseeResponse } from '../types/pesee';
import type { PaginationParams } from '../types/pagination';
import type { CaprinCreate, CaprinResponse, CaprinUpdate } from '$lib/types';
import type { AnimalVenteCreate, Sexe, StatutAnimal } from '../types/animal';

export const caprinsApi = {
    getCaprins: (params?: Partial<PaginationParams> & {
        race?: string;
        sexe?: Sexe[];
        enclos_id?: number;
        production_type?: string;
        statut?: StatutAnimal[];
        search?: string;
    }) =>
        apiClient.get<{ items: CaprinResponse[]; total: number; skip: number; limit: number }>('/caprins', { params }),

    getCaprin: (id: number) =>
        apiClient.get<CaprinResponse>(`/caprins/${id}`),

    createCaprin: (data: CaprinCreate) =>
        apiClient.post<CaprinResponse>('/caprins/create', data),

    updateCaprin: (id: number, data: CaprinUpdate) =>
        apiClient.put<CaprinResponse>(`/caprins/update/${id}`, data),

    getPesees: (id: number) =>
        apiClient.get<PeseeResponse[]>(`/caprins/${id}/pesees`),

    addPesee: (id: number, data: PeseeCreate) =>
        apiClient.post<PeseeResponse>(`/caprins/${id}/pesees`, data),

    getPregnantFemales: (enclos_id?: number) =>
        apiClient.get<{ count: number; females: any[] }>('/caprins/reproduction/chevres', { params: { enclos_id } }),

    getBreedingMales: (enclos_id?: number) =>
        apiClient.get<{ count: number; males: any[] }>('/caprins/reproduction/boucs', { params: { enclos_id } }),

    getKids: (age_days_max?: number, enclos_id?: number) =>
        apiClient.get<{ count: number; kids: any[] }>('/caprins/jeunes/chevreaux', { params: { age_days_max, enclos_id } }),

    recordKidding: (data: { doe_id: number; buck_id?: number; kids_data: any[] }) =>
        apiClient.post('/caprins/reproduction/chevrotage', data),

    getStats: (enclos_id?: number) =>
        apiClient.get('/caprins/stats/global', { params: { enclos_id } }),

    // === NOUVELLES MÉTHODES DE VENTE ===
    enregistrerVente: (id: number, data: AnimalVenteCreate) =>
        apiClient.post<CaprinResponse>(`/caprins/${id}/vente`, data),

    getVentesStats: (params?: { date_debut?: string; date_fin?: string }) =>
        apiClient.get('/caprins/ventes/stats', { params }),

    getCaprinsVendus: (params?: {
        date_debut?: string;
        date_fin?: string;
        client?: string;
        skip?: number;
        limit?: number;
    }) =>
        apiClient.get<CaprinResponse[]>('/caprins/ventes/liste', { params })
};