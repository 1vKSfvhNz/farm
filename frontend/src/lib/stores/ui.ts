// lib/stores/ui.ts
import { writable } from 'svelte/store';

interface UIState {
    sidebarOpen: boolean;
    theme: 'light' | 'dark' | 'system';
    notificationsEnabled: boolean;
    soundEnabled: boolean;
    autoRefresh: boolean;
    refreshInterval: number;
    currentPage: string;
    breadcrumbs: Array<{ label: string; href?: string }>;
    modals: {
        id: string;
        isOpen: boolean;
        data?: any;
    }[];
}

function createUIStore() {
    const initialState: UIState = {
        sidebarOpen: true,
        theme: 'system',
        notificationsEnabled: true,
        soundEnabled: false,
        autoRefresh: true,
        refreshInterval: 30000,
        currentPage: '/',
        breadcrumbs: [],
        modals: []
    };

    const { subscribe, set, update } = writable<UIState>(initialState);

    return {
        subscribe,

        toggleSidebar: () => {
            update(state => ({ ...state, sidebarOpen: !state.sidebarOpen }));
        },

        setSidebarOpen: (open: boolean) => {
            update(state => ({ ...state, sidebarOpen: open }));
        },

        setTheme: (theme: UIState['theme']) => {
            update(state => ({ ...state, theme }));
            if (typeof localStorage !== 'undefined') {
                localStorage.setItem('theme', theme);
            }
        },

        setNotificationsEnabled: (enabled: boolean) => {
            update(state => ({ ...state, notificationsEnabled: enabled }));
        },

        setSoundEnabled: (enabled: boolean) => {
            update(state => ({ ...state, soundEnabled: enabled }));
        },

        setAutoRefresh: (enabled: boolean) => {
            update(state => ({ ...state, autoRefresh: enabled }));
        },

        setRefreshInterval: (interval: number) => {
            update(state => ({ ...state, refreshInterval: interval }));
        },

        setCurrentPage: (page: string) => {
            update(state => ({ ...state, currentPage: page }));
        },

        setBreadcrumbs: (breadcrumbs: UIState['breadcrumbs']) => {
            update(state => ({ ...state, breadcrumbs }));
        },

        addBreadcrumb: (crumb: { label: string; href?: string }) => {
            update(state => ({ ...state, breadcrumbs: [...state.breadcrumbs, crumb] }));
        },

        openModal: (id: string, data?: any) => {
            update(state => ({
                ...state,
                modals: [...state.modals, { id, isOpen: true, data }]
            }));
        },

        closeModal: (id: string) => {
            update(state => ({
                ...state,
                modals: state.modals.filter(m => m.id !== id)
            }));
        },

        closeAllModals: () => {
            update(state => ({ ...state, modals: [] }));
        },

        reset: () => {
            set(initialState);
        }
    };
}

export const uiStore = createUIStore();