// lib/api/users.ts
import { apiClient } from './client';
import type { PaginatedResponse, PaginationParams } from '../types/pagination';
import type { UserCreate, UserResponse, UserUpdate } from '$lib/types/users';

export const usersApi = {
    getUsers: (params?: Partial<PaginationParams> & { role?: string; is_active?: boolean }) =>
        apiClient.get<PaginatedResponse<UserResponse>>('/users', { params }),

    getUser: (id: number) =>
        apiClient.get<UserResponse>(`/users/${id}`),

    createUser: (data: UserCreate) =>
        apiClient.post<UserResponse>('/users', data),

    updateUser: (id: number, data: UserUpdate) =>
        apiClient.put<UserResponse>(`/users/${id}`, data),

    getUserSessions: (id: number) =>
        apiClient.get<any[]>(`/users/${id}/sessions`),

    getUserActions: (id: number, skip = 0, limit = 100) =>
        apiClient.get<any[]>(`/users/${id}/actions`, { params: { skip, limit } })
};
