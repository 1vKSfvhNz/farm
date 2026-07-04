// frontend/src/lib/stores/auth.ts
import { writable, derived } from 'svelte/store';
import { apiClient, setAuthToken, clearAuthTokens, getAccessToken, getRefreshToken } from '../api/client';
import type { LoginResponse, RefreshTokenResponse } from '../types/auth';
import type { UserResponse } from '$lib/types/users';

const isBrowser = typeof window !== 'undefined';

interface AuthState {
    user: UserResponse | null;
    token: string | null;
    isAuthenticated: boolean;
    isLoading: boolean;
}


// Fonctions utilitaires pour éviter les problèmes de this
const logoutAction = async (set: (state: AuthState) => void, initialState: AuthState) => {
    try {
        const refreshToken = getRefreshToken();
        if (refreshToken) {
            await apiClient.post('/auth/revoke', { refresh_token: refreshToken }, { skipRefresh: true });
        }
        await apiClient.post('/auth/logout');
    } catch (error) {
        console.error('Logout error:', error);
    }

    clearAuthTokens();
    set({ ...initialState, isLoading: false });

    if (isBrowser && !window.location.pathname.includes('/login')) {
        window.location.href = '/login';
    }
};

const refreshUserAction = async (update: (fn: (state: AuthState) => AuthState) => void, logout: () => Promise<void>) => {
    if (isBrowser) {
        try {
            const user = await apiClient.get<UserResponse>('/auth/me');
            update(state => ({ ...state, user }));
            return user;
        } catch (error) {
            console.error('Failed to refresh user:', error);
            if ((error as any)?.status === 401) {
                await logout();
            }
            return null;
        }
    }
    return null;
};

const refreshTokenAction = async (update: (fn: (state: AuthState) => AuthState) => void, logout: () => Promise<void>) => {
    try {
        const refreshToken = getRefreshToken();
        if (!refreshToken) throw new Error('No refresh token');

        const response = await apiClient.post<RefreshTokenResponse>('/auth/refresh', { refresh_token: refreshToken }, { skipRefresh: true });

        setAuthToken(response.access_token, response.refresh_token);

        update(state => ({
            ...state,
            token: response.access_token
        }));

        return response.access_token;
    } catch (error) {
        console.error('Refresh token failed:', error);
        await logout();
        throw error;
    }
};

function createAuthStore() {
    const initialState: AuthState = {
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: true
    };

    const { subscribe, set, update } = writable<AuthState>(initialState);

    const init = async () => {
        if (isBrowser) {
            const token = getAccessToken();
            const refreshToken = getRefreshToken();

            if (token) {
                update(state => ({ ...state, token, isAuthenticated: true }));
                try {
                    // Utiliser publicRequest pour /auth/me car le token est déjà dans les headers
                    const user = await apiClient.get<UserResponse>('/auth/me');
                    update(state => ({
                        ...state,
                        user,
                        isLoading: false,
                        isAuthenticated: true
                    }));
                } catch (error: any) {
                    if (error?.status === 401 && refreshToken) {
                        try {
                            const refreshResponse = await apiClient.post<RefreshTokenResponse>('/auth/refresh', { refresh_token: refreshToken }, { skipRefresh: true });

                            setAuthToken(refreshResponse.access_token, refreshResponse.refresh_token);

                            const user = await apiClient.get<UserResponse>('/auth/me');
                            set({
                                user,
                                token: refreshResponse.access_token,
                                isAuthenticated: true,
                                isLoading: false
                            });
                        } catch (refreshError) {
                            console.error('Refresh failed:', refreshError);
                            clearAuthTokens();
                            set({ ...initialState, isLoading: false });
                        }
                    } else {
                        console.error('Failed to get user:', error);
                        clearAuthTokens();
                        set({ ...initialState, isLoading: false });
                    }
                }
            } else {
                update(state => ({ ...state, isLoading: false }));
            }
        } else {
            update(state => ({ ...state, isLoading: false }));
        }
    };

    // frontend/src/lib/stores/auth.ts - Modifiez la fonction login
    const login = async (userlogin: string, password: string): Promise<boolean> => {
        try {
            const response = await apiClient.post<LoginResponse>('/auth/login', { userlogin, password });

            const accessToken = response.access_token;
            const refreshToken = response.refresh_token;

            if (accessToken && refreshToken) {
                // Utiliser setAuthToken qui stocke avec les bonnes clés
                setAuthToken(accessToken, refreshToken);
                
                // Vérification immédiate que le token est stocké
                console.log('Token stocké:', localStorage.getItem('access_token'));
                console.log('Refresh token stocké:', localStorage.getItem('refresh_token'));

                // Construire l'objet utilisateur
                const user: UserResponse = {
                    id: response.user_id,
                    username: response.username,
                    full_name: response.username,
                    email: response.email || '',
                    phone: response.phone || '',
                    roles: response.roles,
                    is_active: true,
                    created_at: new Date().toISOString(),
                    updated_at: new Date().toISOString()
                };

                set({
                    user,
                    token: accessToken,
                    isAuthenticated: true,
                    isLoading: false
                });

                return true;
            }
            return false;
        } catch (error) {
            console.error('Login failed:', error);
            return false;
        }
    };

    const logout = async (): Promise<void> => {
        await logoutAction(set, initialState);
    };

    const logoutAll = async (): Promise<void> => {
        try {
            await apiClient.post('/auth/revoke-all');
            await apiClient.post('/auth/logout-all');
        } catch (error) {
            console.error('Logout all error:', error);
        }

        clearAuthTokens();
        set({ ...initialState, isLoading: false });

        if (isBrowser && !window.location.pathname.includes('/login')) {
            window.location.href = '/login';
        }
    };

    const refreshUser = async (): Promise<UserResponse | null> => {
        return await refreshUserAction(update, logout);
    };

    const refreshToken = async (): Promise<string | null> => {
        return await refreshTokenAction(update, logout);
    };

    const hasRole = (role: string): boolean => {
        let hasRoleResult = false;
        const unsubscribe = subscribe(state => {
            hasRoleResult = state.user?.roles?.includes(role) ?? false;
        });
        unsubscribe();
        return hasRoleResult;
    };

    const hasAnyRole = (roles: string[]): boolean => {
        let hasRoleResult = false;
        const unsubscribe = subscribe(state => {
            hasRoleResult = roles.some(r => state.user?.roles?.includes(r)) ?? false;
        });
        unsubscribe();
        return hasRoleResult;
    };

    const isSuperAdmin = (): boolean => {
        return hasRole('super_admin');
    };

    const getPermissionsForSpecies = (species: string): { canRead: boolean; canWrite: boolean; canDelete: boolean } => {
        let userRoles: string[] = [];
        const unsubscribe = subscribe(state => {
            userRoles = state.user?.roles ?? [];
        });
        unsubscribe();

        const adminRole = `${species}_admin`;
        const techRole = `${species}_technicien`;
        const observerRole = `${species}_observateur`;

        return {
            canRead: userRoles.includes('super_admin') || userRoles.includes(adminRole) || userRoles.includes(techRole) || userRoles.includes(observerRole),
            canWrite: userRoles.includes('super_admin') || userRoles.includes(adminRole) || userRoles.includes(techRole),
            canDelete: userRoles.includes('super_admin') || userRoles.includes(adminRole)
        };
    };

    return {
        subscribe,
        init,
        login,
        logout,
        logoutAll,
        refreshUser,
        refreshToken,
        hasRole,
        hasAnyRole,
        isSuperAdmin,
        getPermissionsForSpecies
    };
}

// Stores dérivés
export const authStore = createAuthStore();

export const isAuthenticated = derived(authStore, $auth => $auth.isAuthenticated);
export const currentUser = derived(authStore, $auth => $auth.user);
export const userRoles = derived(authStore, $auth => $auth.user?.roles ?? []);
export const isSuperAdmin = derived(authStore, $auth => $auth.user?.roles?.includes('super_admin') ?? false);
export const isVeterinaire = derived(authStore, $auth => $auth.user?.roles?.includes('veterinaire') ?? false);
export const isComptable = derived(authStore, $auth => $auth.user?.roles?.includes('comptable') ?? false);
export const isResponsableEnclos = derived(authStore, $auth => $auth.user?.roles?.includes('responsable_enclos') ?? false);

// Permissions par espèce
export const canReadBovins = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('bovin_admin') ||
        roles.includes('bovin_technicien') ||
        roles.includes('bovin_observateur');
});

export const canWriteBovins = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('bovin_admin') ||
        roles.includes('bovin_technicien');
});

export const canReadOvins = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('ovin_admin') ||
        roles.includes('ovin_technicien') ||
        roles.includes('ovin_observateur');
});

export const canWriteOvins = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('ovin_admin') ||
        roles.includes('ovin_technicien');
});

export const canReadCaprins = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('caprin_admin') ||
        roles.includes('caprin_technicien') ||
        roles.includes('caprin_observateur');
});

export const canWriteCaprins = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('caprin_admin') ||
        roles.includes('caprin_technicien');
});

export const canReadAvicoles = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('avicole_admin') ||
        roles.includes('avicole_technicien') ||
        roles.includes('avicole_observateur');
});

export const canWriteAvicoles = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('avicole_admin') ||
        roles.includes('avicole_technicien');
});

export const canReadPiscicoles = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('piscicole_admin') ||
        roles.includes('piscicole_technicien') ||
        roles.includes('piscicole_observateur');
});

export const canWritePiscicoles = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('piscicole_admin') ||
        roles.includes('piscicole_technicien');
});

export const canReadApiculture = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('apiculture_admin') ||
        roles.includes('apiculture_technicien') ||
        roles.includes('apiculture_observateur');
});

export const canWriteApiculture = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('apiculture_admin') ||
        roles.includes('apiculture_technicien');
});

export const canReadEntomoculture = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('entomoculture_admin') ||
        roles.includes('entomoculture_technicien') ||
        roles.includes('entomoculture_observateur');
});

export const canWriteEntomoculture = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') ||
        roles.includes('entomoculture_admin') ||
        roles.includes('entomoculture_technicien');
});

export const canAccessAccounting = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') || roles.includes('comptable');
});

export const canAccessVeterinaire = derived(authStore, $auth => {
    const roles = $auth.user?.roles ?? [];
    return roles.includes('super_admin') || roles.includes('veterinaire');
});