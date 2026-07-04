// lib/utils/constants.ts
export const ANIMAL_SPECIES = {
    BOVIN: 'bovin',
    OVIN: 'ovin',
    CAPRIN: 'caprin',
    AVICOLE: 'avicole',
    PISCICOLE: 'piscicole',
    APICULTURE: 'apiculture',
    ENTOMOCULTURE: 'entomoculture'
} as const;

export const ANIMAL_SPECIES_LABELS: Record<string, string> = {
    [ANIMAL_SPECIES.BOVIN]: 'Bovins',
    [ANIMAL_SPECIES.OVIN]: 'Ovins',
    [ANIMAL_SPECIES.CAPRIN]: 'Caprins',
    [ANIMAL_SPECIES.AVICOLE]: 'Avicoles',
    [ANIMAL_SPECIES.PISCICOLE]: 'Piscicoles',
    [ANIMAL_SPECIES.APICULTURE]: 'Apiculture',
    [ANIMAL_SPECIES.ENTOMOCULTURE]: 'Entomoculture'
};

export const ANIMAL_SEX = {
    MALE: 'male',
    FEMELLE: 'femelle',
    HERMAPHRODITE: 'hermaphrodite'
} as const;

export const ANIMAL_SEX_LABELS: Record<string, string> = {
    [ANIMAL_SEX.MALE]: 'Mâle',
    [ANIMAL_SEX.FEMELLE]: 'Femelle',
    [ANIMAL_SEX.HERMAPHRODITE]: 'Hermaphrodite'
};

export const ANIMAL_STATUS = {
    VIVANT: 'vivant',
    VENDU: 'vendu',
    DECEDE: 'decede',
    TRANSFERE: 'transfere'
} as const;

export const ANIMAL_STATUS_LABELS: Record<string, string> = {
    [ANIMAL_STATUS.VIVANT]: 'Vivant',
    [ANIMAL_STATUS.VENDU]: 'Vendu',
    [ANIMAL_STATUS.DECEDE]: 'Décédé',
    [ANIMAL_STATUS.TRANSFERE]: 'Transféré'
};

export const ALERT_LEVELS = {
    INFO: 'info',
    WARNING: 'warning',
    CRITICAL: 'critical'
} as const;

export const ALERT_LEVELS_LABELS: Record<string, string> = {
    [ALERT_LEVELS.INFO]: 'Information',
    [ALERT_LEVELS.WARNING]: 'Attention',
    [ALERT_LEVELS.CRITICAL]: 'Critique'
};

export const ALERT_LEVELS_COLORS: Record<string, string> = {
    [ALERT_LEVELS.INFO]: 'blue',
    [ALERT_LEVELS.WARNING]: 'orange',
    [ALERT_LEVELS.CRITICAL]: 'red'
};

export const USER_ROLES = {
    ADMIN: 'admin',
    VETERINAIRE: 'veterinaire',
    RESPONSABLE_ENCLOS: 'responsable_enclos',
    TECHNICIEN: 'technicien',
    OBSERVATEUR: 'observateur'
} as const;

export const USER_ROLES_LABELS: Record<string, string> = {
    [USER_ROLES.ADMIN]: 'Administrateur',
    [USER_ROLES.VETERINAIRE]: 'Vétérinaire',
    [USER_ROLES.RESPONSABLE_ENCLOS]: 'Responsable d\'enclos',
    [USER_ROLES.TECHNICIEN]: 'Technicien',
    [USER_ROLES.OBSERVATEUR]: 'Observateur'
};

export const ENCLOS_TYPES = {
    ENCLOS: 'enclos',
    BASSIN: 'bassin',
    PATURAGE: 'pâturage',
    CAGE: 'cage',
    BAC: 'bac'
} as const;

export const ENCLOS_TYPES_LABELS: Record<string, string> = {
    [ENCLOS_TYPES.ENCLOS]: 'Enclos',
    [ENCLOS_TYPES.BASSIN]: 'Bassin',
    [ENCLOS_TYPES.PATURAGE]: 'Pâturage',
    [ENCLOS_TYPES.CAGE]: 'Cage',
    [ENCLOS_TYPES.BAC]: 'Bac'
};

export const RUCHES_STATUTS = {
    ACTIVE: 'active',
    ORPHELINE: 'orpheline',
    EN_ESSIMAGE: 'en_essaimage',
    MORTE: 'morte'
} as const;

export const RUCHES_STATUTS_LABELS: Record<string, string> = {
    [RUCHES_STATUTS.ACTIVE]: 'Active',
    [RUCHES_STATUTS.ORPHELINE]: 'Orpheline',
    [RUCHES_STATUTS.EN_ESSIMAGE]: 'En essaimage',
    [RUCHES_STATUTS.MORTE]: 'Morte'
};

export const COMPOST_TYPES = {
    DECRETS_VERTS: 'déchets verts',
    FUMIER: 'fumier',
    MIXTE: 'mixte'
} as const;

export const COMPOST_TYPES_LABELS: Record<string, string> = {
    [COMPOST_TYPES.DECRETS_VERTS]: 'Déchets verts',
    [COMPOST_TYPES.FUMIER]: 'Fumier',
    [COMPOST_TYPES.MIXTE]: 'Mixte'
};