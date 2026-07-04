// lib/stores/permissions.ts - Version corrigée complète
import { writable } from 'svelte/store';
import type { UserRole, Permission, Espece } from '$lib/types/roles';
import { rolePermissions, getPermissionsFromRoles, isAdmin, isTechnician, isObserver, isVeterinaire, isResponsableEnclos, isSuperAdmin } from '$lib/types/roles';
import { authStore } from './auth';

interface PermissionsState {
    roles: UserRole[];  // Changé: tableau de rôles au lieu d'un seul
    permissions: Permission | null;
}

function createPermissionsStore() {
    const { subscribe, set } = writable<PermissionsState>({
        roles: [],
        permissions: null
    });

    let unsubscribeAuth: (() => void) | null = null;

    function init() {
        unsubscribeAuth = authStore.subscribe(state => {
            if (state.user?.roles && state.user.roles.length > 0) {
                const userRoles = state.user.roles as UserRole[];
                // Fusionner les permissions de tous les rôles
                const permissions = getPermissionsFromRoles(userRoles);
                set({ roles: userRoles, permissions: permissions });
            } else {
                set({ roles: [], permissions: null });
            }
        });
    }

    function cleanup() {
        if (unsubscribeAuth) {
            unsubscribeAuth();
            unsubscribeAuth = null;
        }
    }

    // Helper pour obtenir l'état courant
    let currentState: PermissionsState = { roles: [], permissions: null };
    const unsubscribeStore = subscribe(s => { currentState = s; });

    const store = {
        subscribe,
        init,
        cleanup,

        hasPermission: (permission: keyof Permission): boolean => {
            return currentState.permissions?.[permission] ?? false;
        },

        hasAnyPermission: (permissions: Array<keyof Permission>): boolean => {
            return permissions.some(p => currentState.permissions?.[p] ?? false);
        },

        hasAllPermissions: (permissions: Array<keyof Permission>): boolean => {
            return permissions.every(p => currentState.permissions?.[p] ?? false);
        },

        // Permissions spécifiques enclos
        canViewEnclos: (): boolean => {
            return currentState.permissions?.can_view_enclos ?? false;
        },

        canEditEnclos: (): boolean => {
            return currentState.permissions?.can_edit_enclos ?? false;
        },

        // Vérifications spécifiques par type d'élevage
        canViewEspece: (espece: Espece | string): boolean => {
            const permissionMap: Record<string, keyof Permission> = {
                bovins: 'can_view_bovins',
                ovins: 'can_view_ovins',
                caprins: 'can_view_caprins',
                avicoles: 'can_view_avicoles',
                piscicoles: 'can_view_piscicoles',
                apiculture: 'can_view_apiculture',
                entomoculture: 'can_view_entomoculture'
            };
            const permission = permissionMap[espece];
            return permission ? store.hasPermission(permission) : false;
        },

        canEditEspece: (espece: Espece | string): boolean => {
            const permissionMap: Record<string, keyof Permission> = {
                bovins: 'can_edit_bovins',
                ovins: 'can_edit_ovins',
                caprins: 'can_edit_caprins',
                avicoles: 'can_edit_avicoles',
                piscicoles: 'can_edit_piscicoles',
                apiculture: 'can_edit_apiculture',
                entomoculture: 'can_edit_entomoculture'
            };
            const permission = permissionMap[espece];
            return permission ? store.hasPermission(permission) : false;
        },

        canDeleteEspece: (espece: Espece | string): boolean => {
            const permissionMap: Record<string, keyof Permission> = {
                bovins: 'can_delete_bovins',
                ovins: 'can_delete_ovins',
                caprins: 'can_delete_caprins',
                avicoles: 'can_delete_avicoles',
                piscicoles: 'can_delete_piscicoles',
                apiculture: 'can_delete_apiculture',
                entomoculture: 'can_delete_entomoculture'
            };
            const permission = permissionMap[espece];
            return permission ? store.hasPermission(permission) : false;
        },

        // Récupération des rôles
        getRoles: (): UserRole[] => {
            return currentState.roles;
        },

        getPrimaryRole: (): UserRole | null => {
            return currentState.roles.length > 0 ? currentState.roles[0] : null;
        },

        // Vérifications de rôle (supporte tous les rôles)
        isAdmin: (): boolean => {
            return currentState.roles.some(role => isAdmin(role));
        },

        isSuperAdmin: (): boolean => {
            return currentState.roles.some(role => isSuperAdmin(role));
        },

        isSpeciesAdminFor: (espece: Espece | string): boolean => {
            const speciesRoleMap: Record<string, string> = {
                bovins: 'bovin_admin',
                ovins: 'ovin_admin',
                caprins: 'caprin_admin',
                avicoles: 'avicole_admin',
                piscicoles: 'piscicole_admin',
                apiculture: 'apiculture_admin',
                entomoculture: 'entomoculture_admin'
            };
            const requiredRole = speciesRoleMap[espece];
            return requiredRole ? currentState.roles.includes(requiredRole as UserRole) : false;
        },

        isTechnicianFor: (espece: Espece | string): boolean => {
            const speciesRoleMap: Record<string, string> = {
                bovins: 'bovin_technicien',
                ovins: 'ovin_technicien',
                caprins: 'caprin_technicien',
                avicoles: 'avicole_technicien',
                piscicoles: 'piscicole_technicien',
                apiculture: 'apiculture_technicien',
                entomoculture: 'entomoculture_technicien'
            };
            const requiredRole = speciesRoleMap[espece];
            return requiredRole ? currentState.roles.includes(requiredRole as UserRole) : false;
        },

        isObserverFor: (espece: Espece | string): boolean => {
            const speciesRoleMap: Record<string, string> = {
                bovins: 'bovin_observateur',
                ovins: 'ovin_observateur',
                caprins: 'caprin_observateur',
                avicoles: 'avicole_observateur',
                piscicoles: 'piscicole_observateur',
                apiculture: 'apiculture_observateur',
                entomoculture: 'entomoculture_observateur'
            };
            const requiredRole = speciesRoleMap[espece];
            return requiredRole ? currentState.roles.includes(requiredRole as UserRole) : false;
        },

        isVeterinaire: (): boolean => {
            return currentState.roles.some(role => isVeterinaire(role));
        },

        isResponsableEnclos: (): boolean => {
            return currentState.roles.some(role => isResponsableEnclos(role));
        },

        isComptable: (): boolean => {
            return currentState.roles.includes('responsable_account' as UserRole);
        },

        hasVisionGlobale: (): boolean => {
            return currentState.roles.includes('vision_globale' as UserRole);
        },

        isTechnicien: (): boolean => {
            return currentState.roles.some(role => isTechnician(role));
        },

        isObservateur: (): boolean => {
            return currentState.roles.some(role => isObserver(role));
        },

        // Récupérer la liste des espèces accessibles
        getAccessibleEspeces: (): Espece[] => {
            const especes: Espece[] = ['bovins', 'ovins', 'caprins', 'avicoles', 'piscicoles', 'apiculture', 'entomoculture'];
            return especes.filter(espece => store.canViewEspece(espece));
        },

        // Récupérer les espèces où l'utilisateur a des droits d'édition
        getEditableEspeces: (): Espece[] => {
            const especes: Espece[] = ['bovins', 'ovins', 'caprins', 'avicoles', 'piscicoles', 'apiculture', 'entomoculture'];
            return especes.filter(espece => store.canEditEspece(espece));
        },

        // Vérifier si l'utilisateur a accès à une espèce spécifique avec un niveau minimum
        hasAccessToEspece: (espece: Espece | string, minLevel: 'view' | 'edit' | 'delete' = 'view'): boolean => {
            switch (minLevel) {
                case 'delete':
                    return store.canDeleteEspece(espece);
                case 'edit':
                    return store.canEditEspece(espece);
                default:
                    return store.canViewEspece(espece);
            }
        },

        // Vérification des droits sur une action spécifique
        canPerformAction: (action: string, espece?: Espece | string): boolean => {
            // Actions globales
            if (action === 'export_data') {
                return store.hasPermission('can_export_data');
            }
            if (action === 'view_reports') {
                return store.hasPermission('can_view_reports');
            }
            if (action === 'view_settings') {
                return store.hasPermission('can_view_settings');
            }
            
            // Actions par espèce
            if (espece) {
                if (action === 'view') {
                    return store.canViewEspece(espece);
                }
                if (action === 'edit') {
                    return store.canEditEspece(espece);
                }
                if (action === 'delete') {
                    return store.canDeleteEspece(espece);
                }
            }
            
            return false;
        },

        // Obtenir le niveau de permission maximum pour une espèce
        getMaxPermissionForEspece: (espece: Espece | string): 'none' | 'view' | 'edit' | 'delete' => {
            if (store.canDeleteEspece(espece)) return 'delete';
            if (store.canEditEspece(espece)) return 'edit';
            if (store.canViewEspece(espece)) return 'view';
            return 'none';
        },

        // Vérifier si l'utilisateur peut gérer les utilisateurs
        canManageUsers: (): boolean => {
            return store.hasPermission('can_view_users') && store.hasPermission('can_edit_users');
        }
    };

    return store;
}

export const permissionsStore = createPermissionsStore();