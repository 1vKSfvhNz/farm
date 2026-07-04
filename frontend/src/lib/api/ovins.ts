// lib/api/ovins.ts
import { apiClient } from './client';
import type { OvinCreate, OvinUpdate, OvinResponse } from '../types/ovin';
import type { PeseeCreate, PeseeResponse } from '../types/pesee';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';
import type { AnimalVenteCreate, Sexe, StatutAnimal } from '../types/animal';

export const ovinsApi = {
    getOvins: (params?: Partial<PaginationParams> & {
        race?: string;
        sexe?: Sexe[];
        enclos_id?: number;
        production_type?: string;
        statut?: StatutAnimal[];
        search?: string;
    }) =>
        apiClient.get<PaginatedResponse<OvinResponse>>('/ovins', { params }),

    getOvin: (id: number) =>
        apiClient.get<OvinResponse>(`/ovins/${id}`),

    createOvin: (data: OvinCreate) =>
        apiClient.post<OvinResponse>('/ovins/create', data),

    updateOvin: (id: number, data: OvinUpdate) =>
        apiClient.put<OvinResponse>(`/ovins/update/${id}`, data),

    getPesees: (id: number) =>
        apiClient.get<PeseeResponse[]>(`/ovins/${id}/pesees`),

    addPesee: (id: number, data: PeseeCreate) =>
        apiClient.post<PeseeResponse>(`/ovins/${id}/pesees`, data),

    getEwesForBreeding: (enclos_id?: number) =>
        apiClient.get<{ count: number; ewes: any[] }>('/ovins/reproduction/brebis', { params: { enclos_id } }),

    getRams: (enclos_id?: number) =>
        apiClient.get<{ count: number; rams: any[] }>('/ovins/reproduction/beliers', { params: { enclos_id } }),

    recordLambing: (data: { doe_id: number; ram_id?: number; lambs_data: any[] }) =>
        apiClient.post('/ovins/reproduction/agnelage', data),

    getWoolProduction: (year?: number) =>
        apiClient.get('/ovins/production/laine', { params: { year } }),

    getStats: (enclos_id?: number) =>
        apiClient.get('/ovins/stats/global', { params: { enclos_id } }),

    // === NOUVELLES MÉTHODES DE VENTE ===
    enregistrerVente: (id: number, data: AnimalVenteCreate) =>
        apiClient.post<OvinResponse>(`/ovins/${id}/vente`, data),

    getVentesStats: (params?: { date_debut?: string; date_fin?: string }) =>
        apiClient.get('/ovins/ventes/stats', { params }),

    getOvinsVendus: (params?: {
        date_debut?: string;
        date_fin?: string;
        client?: string;
        skip?: number;
        limit?: number;
    }) =>
        apiClient.get<OvinResponse[]>('/ovins/ventes/liste', { params })
};