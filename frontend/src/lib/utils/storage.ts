// lib/utils/storage.ts
const isBrowser = typeof window !== 'undefined';

export const storage = {
    get: <T>(key: string, defaultValue: T | null = null): T | null => {
        if (!isBrowser) return defaultValue;
        try {
            const item = localStorage.getItem(key);
            if (item === null) return defaultValue;
            return JSON.parse(item) as T;
        } catch {
            return defaultValue;
        }
    },

    set: <T>(key: string, value: T): void => {
        if (!isBrowser) return;
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (error) {
            console.error(`Failed to save ${key}:`, error);
        }
    },

    remove: (key: string): void => {
        if (!isBrowser) return;
        localStorage.removeItem(key);
    },

    clear: (): void => {
        if (!isBrowser) return;
        localStorage.clear();
    },

    has: (key: string): boolean => {
        if (!isBrowser) return false;
        return localStorage.getItem(key) !== null;
    }
};

// Export direct de localStorage pour les cas où on a besoin des méthodes natives
export const localStorageProxy = {
    getItem: (key: string): string | null => {
        if (!isBrowser) return null;
        return localStorage.getItem(key);
    },
    setItem: (key: string, value: string): void => {
        if (!isBrowser) return;
        localStorage.setItem(key, value);
    },
    removeItem: (key: string): void => {
        if (!isBrowser) return;
        localStorage.removeItem(key);
    },
    clear: (): void => {
        if (!isBrowser) return;
        localStorage.clear();
    }
};

export const sessionStorageProxy = {
    getItem: (key: string): string | null => {
        if (!isBrowser) return null;
        return sessionStorage.getItem(key);
    },
    setItem: (key: string, value: string): void => {
        if (!isBrowser) return;
        sessionStorage.setItem(key, value);
    },
    removeItem: (key: string): void => {
        if (!isBrowser) return;
        sessionStorage.removeItem(key);
    },
    clear: (): void => {
        if (!isBrowser) return;
        sessionStorage.clear();
    }
};