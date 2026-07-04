// lib/types/auth.ts
export interface LoginRequest {
    number: string;
    code: string;
}

export interface LoginResponse {
    access_token: string;
    refresh_token: string;
    token_type: string;
    expires_in: number;
    refresh_expires_in: number;
    user_id: number;
    email: string;
    username: string;
    phone: string;
    roles: string[];
}

export interface TokenResponse {
    access_token: string;
    refresh_token: string;
    token_type: string;
    expires_in: number;
    refresh_expires_in: number;
    user_id: number;
    username: string;
    phone: string;
    roles: string[];
}

export interface RefreshTokenRequest {
    refresh_token: string;
}

export interface RefreshTokenResponse {
    access_token: string;
    refresh_token: string;
    token_type: string;
    expires_in: number;
    refresh_expires_in: number;
}

export interface PasswordChangeRequest {
    old_password: string;
    new_password: string;
}
