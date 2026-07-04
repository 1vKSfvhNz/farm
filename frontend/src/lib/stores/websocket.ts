// lib/stores/websocket.ts
import { writable } from 'svelte/store';
import { websocketService } from '../api/websocket';
import type { WebSocketConnectionStatus, WebSocketMessage } from '../types/websocket';
import { alertStore } from './alerts';
import { notificationStore } from './notifications';

interface WebSocketStoreState {
    status: WebSocketConnectionStatus;
    lastMessage: WebSocketMessage | null;
    enabled: boolean;
}

function createWebSocketStore() {
    const initialState: WebSocketStoreState = {
        status: {
            connected: false,
            reconnecting: false,
            error: null,
            lastMessage: null
        },
        lastMessage: null,
        enabled: true
    };

    const { subscribe, set, update } = writable<WebSocketStoreState>(initialState);

    // Initialiser les handlers
    const initHandlers = () => {
        // Handler pour les alertes
        websocketService.on('alert', (data) => {
            if (data.alert_id) {
                alertStore.loadUnreadCount();
                alertStore.loadAlerts();
            }
            notificationStore.addNotification({
                id: Date.now(),
                message: data.message,
                type: data.niveau,
                read: false,
                created_at: new Date().toISOString()
            });
        });

        // Handler pour les notifications
        websocketService.on('notification', (data) => {
            notificationStore.addNotification({
                id: Date.now(),
                message: data.message,
                type: data.type || 'info',
                read: false,
                created_at: new Date().toISOString()
            });
        });

        // Handler pour les mises à jour d'animaux
        websocketService.on('animal_update', (data) => {
            // Déclencher un rechargement des données si nécessaire
            console.log('Animal update received:', data);
        });

        // Handler pour les mises à jour d'enclos
        websocketService.on('enclos_update', (data) => {
            console.log('Enclos update received:', data);
        });

        // Handler pour les rappels de vaccination
        websocketService.on('vaccination_reminder', (data) => {
            notificationStore.addNotification({
                id: Date.now(),
                message: `Vaccination à prévoir: ${data.maladie} pour ${data.animal_identification}`,
                type: 'warning',
                read: false,
                created_at: new Date().toISOString()
            });
        });

        // Handler pour les alertes qualité d'eau
        websocketService.on('water_quality_alert', (data) => {
            notificationStore.addNotification({
                id: Date.now(),
                message: `Qualité d'eau: ${data.parametre} (${data.niveau}) dans ${data.enclos_name}`,
                type: data.niveau === 'critical' ? 'critical' : 'warning',
                read: false,
                created_at: new Date().toISOString()
            });
        });
    };

    // Suivre le statut de connexion
    websocketService.onStatusChange((status) => {
        update(state => ({ ...state, status }));
    });

    return {
        subscribe,

        connect: (token?: string) => {
            if (!initialState.enabled) return;
            websocketService.connect(token);
            initHandlers();
        },

        disconnect: () => {
            websocketService.disconnect();
        },

        send: (message: WebSocketMessage) => {
            return websocketService.send(message);
        },

        enable: () => {
            update(state => ({ ...state, enabled: true }));
        },

        disable: () => {
            update(state => ({ ...state, enabled: false }));
            websocketService.disconnect();
        },

        isConnected: () => {
            return websocketService.isConnected();
        },

        reset: () => {
            set(initialState);
        }
    };
}

export const websocketStore = createWebSocketStore();