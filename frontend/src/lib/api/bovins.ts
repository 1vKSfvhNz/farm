// lib/api/bovins.ts
import { apiClient } from './client';
import type { BovinCreate, BovinUpdate, BovinResponse } from '../types/bovin';
import type { PeseeCreate, PeseeResponse } from '../types/pesee';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';
import type { AnimalVenteCreate, Sexe, StatutAnimal } from '../types/animal';

export const bovinsApi = {
  getBovins: (params?: Partial<PaginationParams> & {
    race?: string;
    sexe?: Sexe[];
    enclos_id?: string | number;
    production_type?: string;
    statut?: StatutAnimal[];
    search?: string;
  }) => {
    return apiClient.get<PaginatedResponse<BovinResponse>>('/bovins', { params });
  },

  getBovin: (id: number) =>
    apiClient.get<BovinResponse>(`/bovins/${id}`),

  // Version corrigée pour createBovin
  createBovin: async (data: BovinCreate): Promise<BovinResponse> => {
    console.log("📤 [createBovin] Données reçues:", data);
    return apiClient.post<BovinResponse>('/bovins/create', data);
  },

  updateBovin: (id: number, data: BovinUpdate) => {
    console.log(`📝 [updateBovin] ID: ${id}, Données:`, data);
    return apiClient.put<BovinResponse>(`/bovins/update/${id}`, data);
  },

  getPesees: (id: number) =>
    apiClient.get<PeseeResponse[]>(`/bovins/${id}/pesees`),

  addPesee: (id: number, data: PeseeCreate) =>
    apiClient.post<PeseeResponse>(`/bovins/${id}/pesees`, data),

  getGrowth: (id: number) =>
    apiClient.get<{ progression: any[]; animal: string }>(`/bovins/${id}/croissance`),

  getStats: () =>
    apiClient.get('/bovins/stats/global'),

  getLactatingCows: () =>
    apiClient.get<{ count: number; cows: any[] }>('/bovins/lactation/en-cours'),
  
  // === NOUVELLES MÉTHODES DE VENTE ===
  enregistrerVente: (id: number, data: AnimalVenteCreate) =>
    apiClient.post<BovinResponse>(`/bovins/${id}/vente`, data),

  getVentesStats: (params?: { date_debut?: string; date_fin?: string }) =>
    apiClient.get('/bovins/ventes/stats', { params }),

  getBovinsVendus: (params?: {
    date_debut?: string;
    date_fin?: string;
    client?: string;
    skip?: number;
    limit?: number;
  }) =>
    apiClient.get<BovinResponse[]>('/bovins/ventes/liste', { params })
};
