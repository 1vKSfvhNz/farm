// lib/types/roles.ts

// Rôles administratifs
export type SuperAdminRole = 'super_admin';

// Rôles Bovins
export type BovinRole = 'bovin_admin' | 'bovin_technicien' | 'bovin_observateur';

// Rôles Ovins
export type OvinRole = 'ovin_admin' | 'ovin_technicien' | 'ovin_observateur';

// Rôles Caprins
export type CaprinRole = 'caprin_admin' | 'caprin_technicien' | 'caprin_observateur';

// Rôles Avicoles
export type AvicoleRole = 'avicole_admin' | 'avicole_technicien' | 'avicole_observateur';

// Rôles Piscicoles
export type PiscicoleRole = 'piscicole_admin' | 'piscicole_technicien' | 'piscicole_observateur';

// Rôles Apiculture
export type ApicultureRole = 'apiculture_admin' | 'apiculture_technicien' | 'apiculture_observateur';

// Rôles Entomoculture
export type EntomocultureRole = 'entomoculture_admin' | 'entomoculture_technicien' | 'entomoculture_observateur';

// Rôles transverses
export type TransverseRole = 'veterinaire' | 'responsable_enclos' | 'responsable_account' | 'vision_globale';

// Union de tous les rôles (version complète)
export type UserRole = 
    | SuperAdminRole
    | BovinRole
    | OvinRole
    | CaprinRole
    | AvicoleRole
    | PiscicoleRole
    | ApicultureRole
    | EntomocultureRole
    | TransverseRole;

// Type pour l'affichage (avec le rôle 'admin' pour compatibilité)
export type DisplayRole = UserRole | 'admin' | 'technicien' | 'observateur';

// Fonctions utilitaires
export function isAdmin(role: UserRole | string): boolean {
    return role === 'super_admin' || role.endsWith('_admin');
}

export function isSuperAdmin(role: UserRole | string): boolean {
    return role === 'super_admin';
}

export function isTechnician(role: UserRole | string): boolean {
    return role.endsWith('_technicien') || role === 'technicien';
}

export function isObserver(role: UserRole | string): boolean {
    return role.endsWith('_observateur') || role === 'observateur';
}

export function isVeterinaire(role: UserRole | string): boolean {
    return role === 'veterinaire';
}

export function isResponsableEnclos(role: UserRole | string): boolean {
    return role === 'responsable_enclos';
}

export function isComptable(role: UserRole | string): boolean {
    return role === 'responsable_account';
}

export function hasVisionGlobale(role: UserRole | string): boolean {
    return role === 'vision_globale';
}

export function getSpeciesFromRole(role: UserRole | string): string | null {
    if (role.startsWith('bovin_')) return 'bovins';
    if (role.startsWith('ovin_')) return 'ovins';
    if (role.startsWith('caprin_')) return 'caprins';
    if (role.startsWith('avicole_')) return 'avicoles';
    if (role.startsWith('piscicole_')) return 'piscicoles';
    if (role.startsWith('apiculture_')) return 'apiculture';
    if (role.startsWith('entomoculture_')) return 'entomoculture';
    return null;
}

export function getRoleLevel(role: UserRole | string): 'admin' | 'technician' | 'observer' | 'transverse' {
    if (isAdmin(role)) return 'admin';
    if (isTechnician(role)) return 'technician';
    if (isObserver(role)) return 'observer';
    return 'transverse';
}

// Constantes pour l'UI
export const ROLE_LABELS: Record<DisplayRole, string> = {
    // Admin complets
    'super_admin': 'Super Administrateur',
    'admin': 'Administrateur', // Compatibilité
    
    // Bovins
    'bovin_admin': 'Admin Bovins',
    'bovin_technicien': 'Technicien Bovins',
    'bovin_observateur': 'Observateur Bovins',
    
    // Ovins
    'ovin_admin': 'Admin Ovins',
    'ovin_technicien': 'Technicien Ovins',
    'ovin_observateur': 'Observateur Ovins',
    
    // Caprins
    'caprin_admin': 'Admin Caprins',
    'caprin_technicien': 'Technicien Caprins',
    'caprin_observateur': 'Observateur Caprins',
    
    // Avicoles
    'avicole_admin': 'Admin Avicoles',
    'avicole_technicien': 'Technicien Avicoles',
    'avicole_observateur': 'Observateur Avicoles',
    
    // Piscicoles
    'piscicole_admin': 'Admin Piscicoles',
    'piscicole_technicien': 'Technicien Piscicoles',
    'piscicole_observateur': 'Observateur Piscicoles',
    
    // Apiculture
    'apiculture_admin': 'Admin Apiculture',
    'apiculture_technicien': 'Technicien Apiculture',
    'apiculture_observateur': 'Observateur Apiculture',
    
    // Entomoculture
    'entomoculture_admin': 'Admin Entomoculture',
    'entomoculture_technicien': 'Technicien Entomoculture',
    'entomoculture_observateur': 'Observateur Entomoculture',
    
    // Rôles transverses
    'veterinaire': 'Vétérinaire',
    'responsable_enclos': 'Responsable Enclos',
    'responsable_account': 'Comptable',
    'vision_globale': 'Vision Globale',
    
    // Compatibilité
    'technicien': 'Technicien',
    'observateur': 'Observateur',
};

export const ROLE_GROUPS = {
    admin: ['super_admin', 'bovin_admin', 'ovin_admin', 'caprin_admin', 'avicole_admin', 'piscicole_admin', 'apiculture_admin', 'entomoculture_admin'],
    technician: ['bovin_technicien', 'ovin_technicien', 'caprin_technicien', 'avicole_technicien', 'piscicole_technicien', 'apiculture_technicien', 'entomoculture_technicien'],
    observer: ['bovin_observateur', 'ovin_observateur', 'caprin_observateur', 'avicole_observateur', 'piscicole_observateur', 'apiculture_observateur', 'entomoculture_observateur'],
    transverse: ['veterinaire', 'responsable_enclos', 'responsable_account', 'vision_globale'],
};

export type Espece = 'bovins' | 'ovins' | 'caprins' | 'avicoles' | 'piscicoles' | 'apiculture' | 'entomoculture';

export interface Permission {
    // Permissions générales
    can_view_dashboard: boolean;
    can_view_alerts: boolean;
    can_view_reports: boolean;
    can_export_data: boolean;
    can_view_settings: boolean;

    // Permissions par espèce
    can_view_bovins: boolean;
    can_edit_bovins: boolean;
    can_delete_bovins: boolean;

    can_view_ovins: boolean;
    can_edit_ovins: boolean;
    can_delete_ovins: boolean;

    can_view_caprins: boolean;
    can_edit_caprins: boolean;
    can_delete_caprins: boolean;

    can_view_avicoles: boolean;
    can_edit_avicoles: boolean;
    can_delete_avicoles: boolean;

    can_view_piscicoles: boolean;
    can_edit_piscicoles: boolean;
    can_delete_piscicoles: boolean;

    can_view_apiculture: boolean;
    can_edit_apiculture: boolean;
    can_delete_apiculture: boolean;

    can_view_entomoculture: boolean;
    can_edit_entomoculture: boolean;
    can_delete_entomoculture: boolean;

    // Permissions transverses
    can_view_enclos: boolean;
    can_edit_enclos: boolean;
    can_view_accounting: boolean;
    can_edit_accounting: boolean;
    can_view_vaccinations: boolean;
    can_edit_vaccinations: boolean;
    can_view_compost: boolean;
    can_edit_compost: boolean;
    can_view_water_quality: boolean;
    can_edit_water_quality: boolean;
    can_view_bea: boolean;
    can_edit_bea: boolean;
    can_view_predictions: boolean;
    can_view_experimental: boolean;
    can_edit_experimental: boolean;
    can_view_users: boolean;
    can_edit_users: boolean;
    can_view_video: boolean;
}

// Génération dynamique des permissions par rôle
function createPermissionsForRole(role: UserRole | string): Permission {
    const isAdminRole = isAdmin(role);
    const isSuperAdminRole = role === 'super_admin';
    const isTechRole = isTechnician(role);
    const isObsRole = isObserver(role);
    const isVet = role === 'veterinaire';
    const isRespEnclos = role === 'responsable_enclos';
    const isCompta = role === 'responsable_account';
    const hasVision = role === 'vision_globale' || isSuperAdminRole;
    
    const species = getSpeciesFromRole(role);
    const hasFullAccess = isSuperAdminRole || hasVision;
    
    // Permissions de base
    const basePermissions: Permission = {
        can_view_dashboard: true,
        can_view_alerts: true,
        can_view_reports: !isObsRole,
        can_export_data: isAdminRole || hasVision,
        can_view_settings: isSuperAdminRole,
        
        can_view_bovins: hasFullAccess || species === 'bovins' || isVet || isRespEnclos || hasVision,
        can_edit_bovins: hasFullAccess || (species === 'bovins' && (isAdminRole || isTechRole)) || isRespEnclos,
        can_delete_bovins: isSuperAdminRole,
        
        can_view_ovins: hasFullAccess || species === 'ovins' || isVet || isRespEnclos || hasVision,
        can_edit_ovins: hasFullAccess || (species === 'ovins' && (isAdminRole || isTechRole)) || isRespEnclos,
        can_delete_ovins: isSuperAdminRole,
        
        can_view_caprins: hasFullAccess || species === 'caprins' || isVet || isRespEnclos || hasVision,
        can_edit_caprins: hasFullAccess || (species === 'caprins' && (isAdminRole || isTechRole)) || isRespEnclos,
        can_delete_caprins: isSuperAdminRole,
        
        can_view_avicoles: hasFullAccess || species === 'avicoles' || isVet || isRespEnclos || hasVision,
        can_edit_avicoles: hasFullAccess || (species === 'avicoles' && (isAdminRole || isTechRole)) || isRespEnclos,
        can_delete_avicoles: isSuperAdminRole,
        
        can_view_piscicoles: hasFullAccess || species === 'piscicoles' || isVet || isRespEnclos || hasVision,
        can_edit_piscicoles: hasFullAccess || (species === 'piscicoles' && (isAdminRole || isTechRole)) || isRespEnclos,
        can_delete_piscicoles: isSuperAdminRole,
        
        can_view_apiculture: hasFullAccess || species === 'apiculture' || isVet || isRespEnclos || hasVision,
        can_edit_apiculture: hasFullAccess || (species === 'apiculture' && (isAdminRole || isTechRole)) || isRespEnclos,
        can_delete_apiculture: isSuperAdminRole,
        
        can_view_entomoculture: hasFullAccess || species === 'entomoculture' || isVet || isRespEnclos || hasVision,
        can_edit_entomoculture: hasFullAccess || (species === 'entomoculture' && (isAdminRole || isTechRole)) || isRespEnclos,
        can_delete_entomoculture: isSuperAdminRole,
        
        can_view_enclos: true,
        can_edit_enclos: isSuperAdminRole || isRespEnclos || hasVision,
        can_view_accounting: isSuperAdminRole || isCompta || hasVision,
        can_edit_accounting: isSuperAdminRole || isCompta,
        can_view_vaccinations: true,
        can_edit_vaccinations: isSuperAdminRole || isVet || hasVision,
        can_view_compost: true,
        can_edit_compost: isSuperAdminRole || isRespEnclos || hasVision,
        can_view_water_quality: true,
        can_edit_water_quality: isSuperAdminRole || isRespEnclos || hasVision,
        can_view_bea: true,
        can_edit_bea: isSuperAdminRole || isVet || isRespEnclos || hasVision,
        can_view_predictions: true,
        can_view_experimental: isSuperAdminRole || hasVision,
        can_edit_experimental: isSuperAdminRole,
        can_view_users: isSuperAdminRole,
        can_edit_users: isSuperAdminRole,
        can_view_video: !isObsRole,
    };
    
    return basePermissions;
}

// Permissions par rôle (version complète)
export const rolePermissions: Record<UserRole, Permission> = {
    super_admin: createPermissionsForRole('super_admin'),
    bovin_admin: createPermissionsForRole('bovin_admin'),
    bovin_technicien: createPermissionsForRole('bovin_technicien'),
    bovin_observateur: createPermissionsForRole('bovin_observateur'),
    ovin_admin: createPermissionsForRole('ovin_admin'),
    ovin_technicien: createPermissionsForRole('ovin_technicien'),
    ovin_observateur: createPermissionsForRole('ovin_observateur'),
    caprin_admin: createPermissionsForRole('caprin_admin'),
    caprin_technicien: createPermissionsForRole('caprin_technicien'),
    caprin_observateur: createPermissionsForRole('caprin_observateur'),
    avicole_admin: createPermissionsForRole('avicole_admin'),
    avicole_technicien: createPermissionsForRole('avicole_technicien'),
    avicole_observateur: createPermissionsForRole('avicole_observateur'),
    piscicole_admin: createPermissionsForRole('piscicole_admin'),
    piscicole_technicien: createPermissionsForRole('piscicole_technicien'),
    piscicole_observateur: createPermissionsForRole('piscicole_observateur'),
    apiculture_admin: createPermissionsForRole('apiculture_admin'),
    apiculture_technicien: createPermissionsForRole('apiculture_technicien'),
    apiculture_observateur: createPermissionsForRole('apiculture_observateur'),
    entomoculture_admin: createPermissionsForRole('entomoculture_admin'),
    entomoculture_technicien: createPermissionsForRole('entomoculture_technicien'),
    entomoculture_observateur: createPermissionsForRole('entomoculture_observateur'),
    veterinaire: createPermissionsForRole('veterinaire'),
    responsable_enclos: createPermissionsForRole('responsable_enclos'),
    responsable_account: createPermissionsForRole('responsable_account'),
    vision_globale: createPermissionsForRole('vision_globale'),
};

// Fonction pour obtenir les permissions à partir d'un tableau de rôles
export function getPermissionsFromRoles(roles: UserRole[]): Permission {
    const mergedPermissions: Permission = createPermissionsForRole('observateur'); // Permissions minimales par défaut
    
    for (const role of roles) {
        const rolePerms = rolePermissions[role];
        if (rolePerms) {
            // Fusionner les permissions (priorité au true)
            for (const [key, value] of Object.entries(rolePerms)) {
                if (value === true) {
                    (mergedPermissions as any)[key] = true;
                }
            }
        }
    }
    
    return mergedPermissions;
}

// Menu items avec les permissions requises
export interface MenuItem {
    href: string;
    label: string;
    icon: string;
    requiredPermission: keyof Permission | null;
    espece?: Espece;
}

export const menuItems: MenuItem[] = [
    { href: '/', label: 'Tableau de bord', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6', requiredPermission: 'can_view_dashboard' },
    { href: '/bovins', label: 'Bovins', icon: 'M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 012.25-2.25h13.5A2.25 2.25 0 0121 7.5v11.25m-18 0A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75m-18 0v-7.5A2.25 2.25 0 015.25 9h13.5A2.25 2.25 0 0121 11.25v7.5m-9-6h.008v.008H12v-.008zM12 15h.008v.008H12V15zm0 2.25h.008v.008H12v-.008zM9.75 15h.008v.008H9.75V15zm0 2.25h.008v.008H9.75v-.008zM7.5 15h.008v.008H7.5V15zm0 2.25h.008v.008H7.5v-.008zm6.75-4.5h.008v.008h-.008v-.008zm0 2.25h.008v.008h-.008V15zm0 2.25h.008v.008h-.008v-.008zm2.25-4.5h.008v.008H16.5v-.008zm0 2.25h.008v.008H16.5V15z', requiredPermission: 'can_view_bovins', espece: 'bovins' },
    { href: '/ovins', label: 'Ovins', icon: 'M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z', requiredPermission: 'can_view_ovins', espece: 'ovins' },
    { href: '/caprins', label: 'Caprins', icon: 'M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z', requiredPermission: 'can_view_caprins', espece: 'caprins' },
    { href: '/avicoles', label: 'Avicoles', icon: 'M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z', requiredPermission: 'can_view_avicoles', espece: 'avicoles' },
    { href: '/piscicoles', label: 'Piscicoles', icon: 'M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375 7.444 2.25 12 2.25s8.25 1.847 8.25 4.125zm0 5.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125', requiredPermission: 'can_view_piscicoles', espece: 'piscicoles' },
    { href: '/apiculture', label: 'Apiculture', icon: 'M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z', requiredPermission: 'can_view_apiculture', espece: 'apiculture' },
    { href: '/entomoculture', label: 'Entomoculture', icon: 'M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z', requiredPermission: 'can_view_entomoculture', espece: 'entomoculture' },
    { href: '/enclos', label: 'Enclos', icon: 'M3.75 21h16.5M4.5 3h15M5.25 3v18m13.5-18v18M9 6.75h1.5M9 12h1.5M9 17.25h1.5M14.25 6.75h1.5M14.25 12h1.5M14.25 17.25h1.5', requiredPermission: 'can_view_enclos' },
    { href: '/accounting', label: 'Comptabilité', icon: 'M2.25 18.75a60.07 60.07 0 0115.797 2.101c.727.198 1.453-.342 1.453-1.096V18.75M3.75 4.5v.75A.75.75 0 013 6h-.75m0 0v11.25m0 0H21m-1.5 0h.75m0 0v-7.5a.75.75 0 00-.75-.75h-3.75m-6 0h3.75', requiredPermission: 'can_view_accounting' },
    { href: '/vaccinations', label: 'Vaccinations', icon: 'M9 12.75L11.25 15 15 9.75M21 12c0 1.268-.63 2.39-1.593 3.068a3.745 3.745 0 01-1.043 3.296 3.745 3.745 0 01-3.296 1.043A3.745 3.745 0 0112 21c-1.268 0-2.39-.63-3.068-1.593a3.746 3.746 0 01-3.296-1.043 3.746 3.746 0 01-1.043-3.296A3.745 3.745 0 013 12c0-1.268.63-2.39 1.593-3.068a3.745 3.745 0 011.043-3.296 3.746 3.746 0 013.296-1.043A3.746 3.746 0 0112 3c1.268 0 2.39.63 3.068 1.593a3.746 3.746 0 013.296 1.043 3.746 3.746 0 011.043 3.296A3.745 3.745 0 0121 12z', requiredPermission: 'can_view_vaccinations' },
    { href: '/compost', label: 'Compost', icon: 'M12 6v12m-3-2.818l.879.659a3 3 0 002.242.488 3 3 0 002.242-.488l.879-.659M21 12a9 9 0 11-18 0 9 9 0 0118 0z', requiredPermission: 'can_view_compost' },
    { href: '/qualite-eau', label: "Qualité d'eau", icon: 'M4.5 12.75a6 6 0 0111.383-3.057M12 6.75h3.75a.75.75 0 01.75.75v3.75m-4.5-4.5L21 15.75M4.5 12.75L12 21m-7.5-8.25L9 9', requiredPermission: 'can_view_water_quality' },
    { href: '/bien-etre', label: 'Bien-être animal', icon: 'M4.5 12.75a6 6 0 0111.383-3.057M12 6.75h3.75a.75.75 0 01.75.75v3.75m-4.5-4.5L21 15.75M4.5 12.75L12 21m-7.5-8.25L9 9', requiredPermission: 'can_view_bea' },
    { href: '/predictions', label: 'Prédictions', icon: 'M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z', requiredPermission: 'can_view_predictions' },
    { href: '/experimental', label: 'Mode expérimental', icon: 'M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.75 18.75L21 21', requiredPermission: 'can_view_experimental' },
    { href: '/videos', label: 'Vidéos', icon: 'M15 10.5a3 3 0 11-6 0 3 3 0 016 0zM19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z', requiredPermission: 'can_view_video' },
    { href: '/utilisateurs', label: 'Utilisateurs', icon: 'M15 19a3 3 0 11-6 0m6 0a9 9 0 11-6 0m6 0h3m-6 0H6m0-6h.01M12 9h.01M9 9h.01M15 9h.01', requiredPermission: 'can_view_users' },
    { href: '/rapports', label: 'Rapports', icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z', requiredPermission: 'can_view_reports' },
    { href: '/alertes', label: 'Alertes', icon: 'M14.857 17.082a23.848 23.848 0 005.454-1.31A8.967 8.967 0 0118 9.75v-.7V9A6 6 0 006 9v.75a8.967 8.967 0 01-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 01-5.714 0m5.714 0a3 3 0 11-5.714 0', requiredPermission: 'can_view_alerts' },
    { href: '/parametres', label: 'Paramètres', icon: 'M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75', requiredPermission: 'can_view_settings' }
];