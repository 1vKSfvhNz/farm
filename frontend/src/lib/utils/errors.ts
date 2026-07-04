// lib/utils/errors.ts
export class ApiError extends Error {
    constructor(
        public status: number,
        public message: string,
        public details?: any
    ) {
        super(message);
        this.name = 'ApiError';
    }
}

export class NetworkError extends Error {
    constructor(message: string = 'Erreur réseau') {
        super(message);
        this.name = 'NetworkError';
    }
}

export class ValidationError extends Error {
    constructor(
        public field: string,
        message: string
    ) {
        super(message);
        this.name = 'ValidationError';
    }
}

export const handleApiError = (error: unknown): string => {
    if (error instanceof ApiError) {
        switch (error.status) {
            case 400:
                return error.details?.message || 'Requête invalide';
            case 401:
                return 'Non autorisé. Veuillez vous reconnecter.';
            case 403:
                return 'Accès interdit. Vous n\'avez pas les permissions nécessaires.';
            case 404:
                return 'Ressource non trouvée.';
            case 409:
                return 'Conflit avec les données existantes.';
            case 422:
                return 'Données invalides. Veuillez vérifier votre saisie.';
            case 429:
                return 'Trop de requêtes. Veuillez réessayer plus tard.';
            case 500:
                return 'Erreur serveur. Veuillez réessayer plus tard.';
            default:
                return error.message || 'Une erreur est survenue';
        }
    }

    if (error instanceof NetworkError) {
        return 'Erreur de connexion. Vérifiez votre réseau.';
    }

    if (error instanceof ValidationError) {
        return `Erreur de validation: ${error.message}`;
    }

    if (error instanceof Error) {
        return error.message;
    }

    return 'Une erreur inattendue est survenue';
};

export const showErrorToast = (error: unknown, callback?: (message: string) => void): void => {
    const message = handleApiError(error);
    if (callback) {
        callback(message);
    } else {
        console.error(message);
    }
};