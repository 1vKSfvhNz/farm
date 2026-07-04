// lib/stores/alerts.ts - Version corrigée (suppression de 'total')
import { writable } from 'svelte/store';
import { alertsApi } from '../api/alerts';
import type { AlertResponse } from '../types/alerts';

interface AlertState {
    alerts: AlertResponse[];
    unreadCount: number;
    criticalCount: number;
    warningCount: number;
    infoCount: number;
    isLoading: boolean;
    filters: {
        niveau?: 'info' | 'warning' | 'critical';
        est_lue?: boolean;
        espece?: string;
        startDate?: string;
        endDate?: string;
    };
    pagination: {
        page: number;
        limit: number;
        total: number;
    };
}

function createAlertStore() {
    const initialState: AlertState = {
        alerts: [],
        unreadCount: 0,
        criticalCount: 0,
        warningCount: 0,
        infoCount: 0,
        isLoading: false,
        filters: {},
        pagination: {
            page: 1,
            limit: 20,
            total: 0
        }
    };

    const { subscribe, set, update } = writable<AlertState>(initialState);

    const updateCounts = (alerts: AlertResponse[]) => {
        const unreadCount = alerts.filter(a => !a.est_lue).length;
        const criticalCount = alerts.filter(a => a.niveau === 'critical').length;
        const warningCount = alerts.filter(a => a.niveau === 'warning').length;
        const infoCount = alerts.filter(a => a.niveau === 'info').length;
        return { unreadCount, criticalCount, warningCount, infoCount };
    };

    // Helper pour obtenir l'état courant
    const getCurrentState = (): AlertState => {
        let currentState: AlertState = initialState;
        const unsubscribe = subscribe(s => { currentState = s; });
        unsubscribe();
        return currentState;
    };

    return {
        subscribe,

        loadAlerts: async () => {
            update(state => ({ ...state, isLoading: true }));
            try {
                const currentState = getCurrentState();

                const response = await alertsApi.getAlerts({
                    skip: (currentState.pagination.page - 1) * currentState.pagination.limit,
                    limit: currentState.pagination.limit,
                    niveau: currentState.filters.niveau,
                    est_lue: currentState.filters.est_lue,
                    espece: currentState.filters.espece
                });

                const counts = updateCounts(response.items);

                update(state => ({
                    ...state,
                    alerts: response.items,
                    unreadCount: counts.unreadCount,
                    criticalCount: counts.criticalCount,
                    warningCount: counts.warningCount,
                    infoCount: counts.infoCount,
                    pagination: {
                        ...state.pagination,
                        total: response.total
                    },
                    isLoading: false
                }));
            } catch (error) {
                console.error('Failed to load alerts:', error);
                update(state => ({ ...state, isLoading: false }));
            }
        },

        loadUnreadCount: async () => {
            try {
                const response = await alertsApi.getUnreadCount();
                update(state => ({ ...state, unreadCount: response.unread_count }));
            } catch (error) {
                console.error('Failed to load unread count:', error);
            }
        },

        markAsRead: async (id: number) => {
            try {
                await alertsApi.markAsRead(id);
                update(state => {
                    const alerts = state.alerts.map(alert =>
                        alert.id === id ? { ...alert, est_lue: true } : alert
                    );
                    const counts = updateCounts(alerts);
                    return {
                        ...state,
                        alerts,
                        unreadCount: counts.unreadCount
                    };
                });
            } catch (error) {
                console.error('Failed to mark alert as read:', error);
            }
        },

        markAllAsRead: async () => {
            try {
                await alertsApi.markAllAsRead();
                update(state => {
                    const alerts = state.alerts.map(alert => ({ ...alert, est_lue: true }));
                    return {
                        ...state,
                        alerts,
                        unreadCount: 0
                    };
                });
            } catch (error) {
                console.error('Failed to mark all alerts as read:', error);
            }
        },

        resolveAlert: async (id: number, resolution_note?: string) => {
            try {
                await alertsApi.resolveAlert(id, resolution_note);
                update(state => ({
                    ...state,
                    alerts: state.alerts.map(alert =>
                        alert.id === id ? { ...alert, est_traitee: true } : alert
                    )
                }));
            } catch (error) {
                console.error('Failed to resolve alert:', error);
            }
        },

        setFilters: (filters: Partial<AlertState['filters']>) => {
            update(state => ({
                ...state,
                filters: { ...state.filters, ...filters },
                pagination: { ...state.pagination, page: 1 }
            }));
        },

        clearFilters: () => {
            update(state => ({
                ...state,
                filters: {},
                pagination: { ...state.pagination, page: 1 }
            }));
        },

        setPage: (page: number) => {
            update(state => ({
                ...state,
                pagination: { ...state.pagination, page }
            }));
        },

        setLimit: (limit: number) => {
            update(state => ({
                ...state,
                pagination: { ...state.pagination, limit, page: 1 }
            }));
        },

        addAlert: (alert: AlertResponse) => {
            update(state => {
                const alerts = [alert, ...state.alerts];
                const counts = updateCounts(alerts);
                return {
                    ...state,
                    alerts,
                    unreadCount: counts.unreadCount,
                    criticalCount: counts.criticalCount,
                    warningCount: counts.warningCount,
                    infoCount: counts.infoCount,
                    pagination: {
                        ...state.pagination,
                        total: state.pagination.total + 1
                    }
                };
            });
        },

        removeAlert: (id: number) => {
            update(state => {
                const alerts = state.alerts.filter(a => a.id !== id);
                const counts = updateCounts(alerts);
                return {
                    ...state,
                    alerts,
                    unreadCount: counts.unreadCount,
                    criticalCount: counts.criticalCount,
                    warningCount: counts.warningCount,
                    infoCount: counts.infoCount,
                    pagination: {
                        ...state.pagination,
                        total: state.pagination.total - 1
                    }
                };
            });
        },

        reset: () => {
            set(initialState);
        }
    };
}

export const alertStore = createAlertStore();