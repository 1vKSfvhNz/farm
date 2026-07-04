// lib/api/enclos.ts
import { apiClient } from './client';
import type { EnclosCreate, EnclosUpdate, EnclosResponse } from '../types/enclos';
import type { PaginatedResponse } from '../types/pagination';

export const enclosApi = {
    getEnclos: (params?: {
        skip?: number;
        limit?: number;
        page?: number;
        enclos_type?: string | string[];
        zone?: string;
    }) => {
        // Créer un objet URLSearchParams pour gérer correctement les tableaux
        const searchParams = new URLSearchParams();
        
        if (params?.page) searchParams.append('page', params.page.toString());
        if (params?.limit) searchParams.append('limit', params.limit.toString());
        if (params?.zone) searchParams.append('zone', params.zone);
        
        // Gérer enclos_type (peut être string ou tableau)
        if (params?.enclos_type) {
            if (Array.isArray(params.enclos_type)) {
                params.enclos_type.forEach(type => {
                    searchParams.append('enclos_type', type);
                });
            } else {
                searchParams.append('enclos_type', params.enclos_type);
            }
        }
        
        return apiClient.get<PaginatedResponse<EnclosResponse>>(`/enclos?${searchParams.toString()}`);
    },

    getEnclosById: (id: number) =>
        apiClient.get<EnclosResponse>(`/enclos/${id}`),

    getEnclosStats: (id: number) =>
        apiClient.get(`/enclos/${id}/stats`),

    createEnclos: (data: EnclosCreate) => {
        apiClient.post<EnclosResponse>('/enclos/create', data);
    },

    updateEnclos: (id: number, data: EnclosUpdate) =>
        apiClient.put<EnclosResponse>(`/enclos/update/${id}`, data),
};