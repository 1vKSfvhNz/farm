// frontend/src/lib/api/client.ts

import type { TokenResponse } from "$lib/types/auth";

// Configuration
export const API_URL = import.meta.env?.VITE_API_URL || 'http://localhost:8000/api/v1';

// Détection du navigateur
const isBrowser = typeof window !== 'undefined';

export interface RequestConfig extends Omit<RequestInit, 'body'> {
    params?: Record<string, any>;
    headers?: Record<string, string>;
    body?: any;
    responseType?: 'json' | 'blob' | 'text' | 'arrayBuffer';
    skipAuth?: boolean;  // Pour les routes publiques
    skipRefresh?: boolean; // Pour éviter les boucles de refresh
}

class ApiClient {
    private isRefreshing = false;
    private failedQueue: Array<{
        resolve: (value: any) => void;
        reject: (error: any) => void;
        url: string;
        config: RequestConfig;
    }> = [];
    
    private processQueue = (error: any, token: string | null = null) => {
        console.log(`📋 [processQueue] Traitement de ${this.failedQueue.length} requêtes en file`);
        
        this.failedQueue.forEach((prom) => {
            if (error) {
                console.error(`❌ [processQueue] Erreur:`, error.message);
                prom.reject(error);
            } else if (token) {
                console.log(`🔐 [processQueue] Token récupéré: ${token.substring(0, 30)}...`);
                this.request(prom.url, {
                    ...prom.config,
                    headers: {
                        ...prom.config.headers,
                        'Authorization': `Bearer ${token}`
                    },
                    skipRefresh: true
                }).then(prom.resolve).catch(prom.reject);
            }
        });
        this.failedQueue = [];
    };

    private async refreshToken(): Promise<string> {
        console.log('🔄 [refreshToken] Début du rafraîchissement du token');
        const refreshToken = localStorage.getItem('refresh_token');

        if (!refreshToken) {
            console.error('❌ [refreshToken] Aucun refresh token trouvé');
            throw new Error('No refresh token');
        }

        console.log(`🔑 [refreshToken] Refresh token trouvé: ${refreshToken.substring(0, 30)}...`);

        try {
            const response = await fetch(`${API_URL}/auth/refresh`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refresh_token: refreshToken })
            });

            console.log(`📡 [refreshToken] Réponse reçue: ${response.status}`);

            if (!response.ok) {
                if (response.status === 401) {
                    console.error('❌ [refreshToken] Refresh token invalide, déconnexion');
                    this.clearTokensAndRedirect();
                    throw new Error('Session expirée');
                }
                console.error(`❌ [refreshToken] Échec: ${response.status}`);
                throw new Error('Refresh failed');
            }

            const data: TokenResponse = await response.json();
            const newAccessToken = data.access_token;
            const newRefreshToken = data.refresh_token;

            console.log(`✅ [refreshToken] Nouveaux tokens générés: Access: ${newAccessToken.substring(0, 30)}...`);

            localStorage.setItem('access_token', newAccessToken);
            localStorage.setItem('refresh_token', newRefreshToken);
            
            console.log(`💾 [refreshToken] Tokens sauvegardés dans localStorage`);

            return newAccessToken;
        } catch (error) {
            console.error(`❌ [refreshToken] Erreur:`, error);
            this.clearTokensAndRedirect();
            throw error;
        }
    }

    private clearTokensAndRedirect(): void {
        console.log('🧹 [clearTokensAndRedirect] Nettoyage des tokens');
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        localStorage.removeItem('user');

        if (isBrowser && !window.location.pathname.includes('/login')) {
            console.log('🚪 [clearTokensAndRedirect] Redirection vers /login');
            window.location.href = '/login';
        }
    }

    private buildUrl(url: string, params?: Record<string, any>): string {
        const normalizedUrl = url.startsWith('/') ? url : `/${url}`;
        const baseUrl = `${API_URL}${normalizedUrl}`;

        if (!params) return baseUrl;

        const searchParams = new URLSearchParams();
        Object.entries(params).forEach(([key, value]) => {
            if (value !== undefined && value !== null && value !== '') {
                searchParams.append(key, String(value));
            }
        });

        const queryString = searchParams.toString();
        const finalUrl = queryString ? `${baseUrl}?${queryString}` : baseUrl;
        
        console.log(`🔗 [buildUrl] URL construite: ${finalUrl}`);
        return finalUrl;
    }

    private getHeaders(additionalHeaders?: Record<string, string>, isBlob?: boolean): Record<string, string> {
        console.log(`📦 [getHeaders] Construction des headers, isBlob: ${isBlob}`);
        
        const headers: Record<string, string> = {};

        if (!isBlob) {
            headers['Content-Type'] = 'application/json';
        }

        if (isBrowser) {
            const token = localStorage.getItem('access_token');
            
            if (token) {
                console.log(`🔐 [getHeaders] Token trouvé: ${token.substring(0, 30)}...`);
                headers['Authorization'] = `Bearer ${token}`;
            } else {
                console.warn('⚠️ [getHeaders] Aucun token trouvé dans localStorage');
            }
        }

        headers['X-Requested-With'] = 'XMLHttpRequest';
        
        console.log(`✅ [getHeaders] Headers construits:`, Object.keys(headers));

        return { ...headers, ...additionalHeaders };
    }

    private async handleResponse<T>(response: Response, responseType?: string): Promise<T> {
        console.log(`📨 [handleResponse] Status: ${response.status}, Type: ${responseType}`);
        
        if (!response.ok) {
            let errorMessage = `Erreur ${response.status}`;
            let errorDetail = null;

            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorData.message || errorData.error;
                errorMessage = errorDetail || errorMessage;
                console.error(`❌ [handleResponse] Erreur: ${errorMessage}`);
            } catch (e) {
                try {
                    const text = await response.text();
                    if (text) errorMessage = text;
                    console.error(`❌ [handleResponse] Erreur texte: ${errorMessage}`);
                } catch (e2) {
                    console.error(`❌ [handleResponse] Erreur inconnue`);
                }
            }

            const error = new Error(errorMessage);
            (error as any).status = response.status;
            (error as any).detail = errorDetail;
            throw error;
        }

        if (response.status === 204) {
            console.log(`📭 [handleResponse] Pas de contenu (204)`);
            return null as T;
        }

        if (responseType === 'blob') {
            console.log(`📦 [handleResponse] Réponse de type blob`);
            return (await response.blob()) as T;
        }
        if (responseType === 'text') {
            console.log(`📝 [handleResponse] Réponse de type texte`);
            return (await response.text()) as T;
        }
        if (responseType === 'arrayBuffer') {
            console.log(`💾 [handleResponse] Réponse de type arrayBuffer`);
            return (await response.arrayBuffer()) as T;
        }

        const contentType = response.headers.get('content-type');
        if (contentType?.includes('application/json')) {
            console.log(`✅ [handleResponse] Réponse JSON reçue`);
            return await response.json();
        }

        console.warn(`⚠️ [handleResponse] Type de réponse non reconnu`);
        return {} as T;
    }

    async request<T>(url: string, config: RequestConfig = {}): Promise<T> {
        console.log(`🚀 [request] Début requête: ${config.method || 'GET'} ${url}`);
        
        const { params, headers: customHeaders, body, responseType, skipAuth, skipRefresh, ...restConfig } = config;
        const fullUrl = this.buildUrl(url, params);
        const isBlob = responseType === 'blob';
        const headers = this.getHeaders(customHeaders, isBlob);

        if (skipAuth) {
            console.log('🔓 [request] skipAuth=true, suppression du header Authorization');
            delete headers['Authorization'];
        }

        const fetchOptions: RequestInit = {
            ...restConfig,
            headers,
            credentials: 'include'
        };

        if (body !== undefined && !isBlob) {
            fetchOptions.body = JSON.stringify(body);
            console.log(`📤 [request] Body:`, JSON.stringify(body).substring(0, 100));
        } else if (body && isBlob) {
            fetchOptions.body = body;
            console.log(`📤 [request] Body de type blob`);
        }

        const makeRequest = async (token?: string): Promise<Response> => {
            console.log(`🔨 [makeRequest] Exécution de la requête avec token: ${token ? token.substring(0, 30) + '...' : 'non fourni'}`);
            const currentHeaders: Record<string, string> = { ...headers };
            
            if (token && !skipAuth) {
                currentHeaders['Authorization'] = `Bearer ${token}`;
                console.log(`🔐 [makeRequest] Header Authorization ajouté`);
            }
            
            if (skipAuth) {
                delete currentHeaders['Authorization'];
            }

            return fetch(fullUrl, {
                ...fetchOptions,
                headers: currentHeaders
            });
        };

        try {
            let response = await makeRequest();
            console.log(`📡 [request] Réponse initiale: ${response.status}`);

            const isAuthRoute = url.includes('/auth/');
            const isRefreshRoute = url.includes('/auth/refresh');

            if (response.status === 401 && !isAuthRoute && !isRefreshRoute && !skipRefresh && !skipAuth) {
                console.warn(`⚠️ [request] 401 non autorisé, tentative de refresh...`);
                
                if (!this.isRefreshing) {
                    this.isRefreshing = true;
                    console.log(`🔄 [request] Début du processus de refresh`);

                    try {
                        const newToken = await this.refreshToken();
                        this.isRefreshing = false;
                        console.log(`✅ [request] Refresh réussi, nouvelle tentative de requête`);

                        response = await makeRequest(newToken);
                        console.log(`📡 [request] Réponse après refresh: ${response.status}`);

                        if (response.status === 401) {
                            console.error(`❌ [request] Encore 401 après refresh`);
                            throw new Error('Session expirée');
                        }

                        this.processQueue(null, newToken);
                    } catch (refreshError) {
                        console.error(`❌ [request] Échec du refresh`, refreshError);
                        this.isRefreshing = false;
                        this.processQueue(refreshError, null);
                        throw refreshError;
                    }
                } else {
                    console.log(`⏳ [request] Refresh en cours, mise en file d'attente`);
                    return new Promise((resolve, reject) => {
                        this.failedQueue.push({
                            resolve: (value) => resolve(value),
                            reject,
                            url,
                            config
                        });
                    });
                }
            }

            const result = await this.handleResponse<T>(response, responseType);
            console.log(`✅ [request] Requête terminée avec succès`);
            return result;
        } catch (error) {
            console.error(`❌ [request] Erreur:`, error instanceof Error ? error.message : 'Unknown error');
            if (error instanceof Error) {
                throw error;
            }
            throw new Error('Une erreur inattendue est survenue');
        }
    }

    // Méthodes HTTP avec console.log
    get = async <T>(url: string, config?: RequestConfig): Promise<T> => {
        console.log(`📖 [GET] Appel GET vers: ${url}`);
        return this.request<T>(url, { ...config, method: 'GET' });
    };

    post = async <T>(url: string, data?: any, config?: RequestConfig): Promise<T> => {
        console.log(`📝 [POST] Appel POST vers: ${url}`);
        return this.request<T>(url, {
            ...config,
            method: 'POST',
            body: data
        });
    };

    put = async <T>(url: string, data?: any, config?: RequestConfig): Promise<T> => {
        console.log(`🔄 [PUT] Appel PUT vers: ${url}`);
        return this.request<T>(url, {
            ...config,
            method: 'PUT',
            body: data
        });
    };

    delete = async <T>(url: string, config?: RequestConfig): Promise<T> => {
        console.log(`🗑️ [DELETE] Appel DELETE vers: ${url}`);
        return this.request<T>(url, { ...config, method: 'DELETE' });
    };

    patch = async <T>(url: string, data?: any, config?: RequestConfig): Promise<T> => {
        console.log(`🔧 [PATCH] Appel PATCH vers: ${url}`);
        return this.request<T>(url, {
            ...config,
            method: 'PATCH',
            body: data
        });
    };
}

// Instance unique
export const apiClient = new ApiClient();

export async function checkApiHealth(): Promise<boolean> {
    try {
        const response = await fetch(`${API_URL.replace('/api/v1', '')}/health`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });
        return response.ok;
    } catch (error) {
        console.error('API health check failed:', error);
        return false;
    }
}

// Hook pour Svelte/React
export const useApiClient = () => {
    console.log('🎣 [useApiClient] Hook appelé');
    return apiClient;
};

// Fonction utilitaire pour les requêtes sans auth
export const publicRequest = async <T>(url: string, config?: RequestConfig): Promise<T> => {
    console.log(`🌐 [publicRequest] Requête publique vers: ${url}`);
    return apiClient.request<T>(url, { ...config, skipAuth: true });
};

// Fonction pour mettre à jour le token après login
export const setAuthToken = (token: string, refreshToken: string): void => {
    console.log(`💾 [setAuthToken] Sauvegarde des tokens: Access: ${token.substring(0, 30)}...`);
    if (isBrowser) {
        localStorage.setItem('access_token', token);
        localStorage.setItem('refresh_token', refreshToken);
        console.log(`✅ [setAuthToken] Tokens sauvegardés dans localStorage`);
    }
};

// Fonction pour mettre à jour seulement le token d'accès
export const setAccessToken = (token: string): void => {
    console.log(`💾 [setAccessToken] Sauvegarde du token d'accès: ${token.substring(0, 30)}...`);
    if (isBrowser) {
        localStorage.setItem('access_token', token);
        console.log(`✅ [setAccessToken] Token d'accès sauvegardé`);
    }
};

// Fonction pour effacer les tokens
export const clearAuthTokens = (): void => {
    console.log(`🧹 [clearAuthTokens] Suppression des tokens`);
    if (isBrowser) {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        localStorage.removeItem('user');
        console.log(`✅ [clearAuthTokens] Tokens supprimés`);
    }
};

// Fonction pour récupérer le token d'accès
export const getAccessToken = (): string | null => {
    if (isBrowser) {
        const token = localStorage.getItem('access_token');
        console.log(`🔍 [getAccessToken] Token récupéré: ${token ? token.substring(0, 30) + '...' : 'null'}`);
        return token;
    }
    console.warn(`⚠️ [getAccessToken] Pas de navigateur`);
    return null;
};

// Fonction pour récupérer le refresh token
export const getRefreshToken = (): string | null => {
    if (isBrowser) {
        const token = localStorage.getItem('refresh_token');
        console.log(`🔍 [getRefreshToken] Refresh token récupéré: ${token ? token.substring(0, 30) + '...' : 'null'}`);
        return token;
    }
    console.warn(`⚠️ [getRefreshToken] Pas de navigateur`);
    return null;
};

// Fonction pour vérifier si l'utilisateur est authentifié
export const isAuthenticated = (): boolean => {
    if (!isBrowser) return false;
    const token = getAccessToken();
    const isAuth = token !== null && token !== '';
    console.log(`🔐 [isAuthenticated] Authentifié: ${isAuth}`);
    return isAuth;
};

// Fonction pour obtenir les headers d'authentification
export const getAuthHeaders = (): Record<string, string> => {
    const token = getAccessToken();
    const headers: Record<string, string> = {};
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    console.log(`📋 [getAuthHeaders] Headers d'auth:`, Object.keys(headers));
    return headers;
};

// Types exportés
export type { TokenResponse };