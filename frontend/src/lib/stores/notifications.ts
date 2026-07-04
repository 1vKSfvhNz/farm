// lib/stores/notifications.ts - Version corrigée
import { writable } from 'svelte/store';
import { apiClient } from '../api/client';

interface Notification {
    id: number;
    message: string;
    type: 'info' | 'warning' | 'critical';
    read: boolean;
    created_at: string;
}

interface AlertResponse {
    id: number;
    message: string;
    niveau: string;
    est_lue: boolean;
    created_at: string;
}

function createNotificationStore() {
    const { subscribe, set, update } = writable<{
        notifications: Notification[];
        unreadCount: number;
        isLoading: boolean;
    }>({
        notifications: [],
        unreadCount: 0,
        isLoading: false
    });

    return {
        subscribe,

        loadNotifications: async () => {
            update(state => ({ ...state, isLoading: true }));
            try {
                const response = await apiClient.get<{
                    items: AlertResponse[];
                    total: number;
                }>('/alerts', { params: { limit: 50 } });

                const notifications = response.items.map(alert => ({
                    id: alert.id,
                    message: alert.message,
                    type: alert.niveau as Notification['type'],
                    read: alert.est_lue,
                    created_at: alert.created_at
                }));
                const unreadCount = notifications.filter(n => !n.read).length;
                set({ notifications, unreadCount, isLoading: false });
            } catch (error) {
                console.error('Failed to load notifications:', error);
                update(state => ({ ...state, isLoading: false }));
            }
        },

        markAsRead: async (id: number) => {
            try {
                await apiClient.post(`/alerts/${id}/read`);
                update(state => {
                    const notifications = state.notifications.map(n =>
                        n.id === id ? { ...n, read: true } : n
                    );
                    const unreadCount = notifications.filter(n => !n.read).length;
                    return { ...state, notifications, unreadCount };
                });
            } catch (error) {
                console.error('Failed to mark as read:', error);
            }
        },

        markAllAsRead: async () => {
            try {
                await apiClient.post('/alerts/mark-all-read');
                update(state => ({
                    ...state,
                    notifications: state.notifications.map(n => ({ ...n, read: true })),
                    unreadCount: 0
                }));
            } catch (error) {
                console.error('Failed to mark all as read:', error);
            }
        },

        getUnreadCount: async (): Promise<number> => {
            try {
                const response = await apiClient.get<{ unread_count: number }>('/alerts/unread/count');
                return response.unread_count;
            } catch (error) {
                console.error('Failed to get unread count:', error);
                return 0;
            }
        },

        addNotification: (notification: Notification) => {
            update(state => ({
                ...state,
                notifications: [notification, ...state.notifications],
                unreadCount: state.unreadCount + 1
            }));
        },

        clearNotifications: () => {
            set({ notifications: [], unreadCount: 0, isLoading: false });
        }
    };
}

export const notificationStore = createNotificationStore();