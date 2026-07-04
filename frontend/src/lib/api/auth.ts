// lib/api/auth.ts
import { apiClient } from './client';
import type { LoginRequest, TokenResponse, PasswordChangeRequest } from '../types/auth';
import type { UserResponse } from '$lib/types/users';

export const authApi = {
    login: (data: LoginRequest) =>
        apiClient.post<TokenResponse>('/auth/login', data),

    logout: () =>
        apiClient.post('/auth/logout'),

    logoutAll: () =>
        apiClient.post('/auth/logout-all'),

    getCurrentUser: () =>
        apiClient.get<UserResponse>('/auth/me'),

    changePassword: (data: PasswordChangeRequest) =>
        apiClient.post('/auth/change-password', data)
};