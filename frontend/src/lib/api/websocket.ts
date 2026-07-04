// lib/api/websocket.ts
import type {
    WebSocketMessage,
    WebSocketEventType,
    WebSocketConnectionStatus
} from '../types/websocket';

type MessageHandler = (data: any, message: WebSocketMessage) => void;

class WebSocketService {
    private ws: WebSocket | null = null;
    private url: string;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 1000;
    private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
    private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
    private heartbeatTimeout: ReturnType<typeof setTimeout> | null = null;
    private messageHandlers: Map<WebSocketEventType, Set<MessageHandler>> = new Map();
    private globalHandlers: Set<MessageHandler> = new Set();
    private statusHandlers: Set<(status: WebSocketConnectionStatus) => void> = new Set();
    private token: string | null = null;

    constructor() {
        const protocol = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = import.meta.env?.VITE_WS_HOST || (typeof window !== 'undefined' ? window.location.host : 'localhost:8000');
        const path = import.meta.env?.VITE_WS_PATH || '/ws';
        this.url = `${protocol}//${host}${path}`;
    }

    private getStatus(): WebSocketConnectionStatus {
        return {
            connected: this.ws?.readyState === WebSocket.OPEN,
            reconnecting: this.reconnectTimeout !== null,
            error: null,
            lastMessage: null
        };
    }

    private notifyStatusChange() {
        const status = this.getStatus();
        this.statusHandlers.forEach(handler => handler(status));
    }

    private startHeartbeat() {
        this.stopHeartbeat();

        this.heartbeatInterval = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'ping',
                    timestamp: new Date().toISOString()
                }));

                this.heartbeatTimeout = setTimeout(() => {
                    console.warn('Heartbeat timeout, closing connection');
                    this.ws?.close();
                }, 10000);
            }
        }, 30000);
    }

    private stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
        if (this.heartbeatTimeout) {
            clearTimeout(this.heartbeatTimeout);
            this.heartbeatTimeout = null;
        }
    }

    private handleMessage(event: MessageEvent) {
        try {
            const message = JSON.parse(event.data) as WebSocketMessage;

            // Répondre au pong
            if (message.type === 'pong') {
                if (this.heartbeatTimeout) {
                    clearTimeout(this.heartbeatTimeout);
                    this.heartbeatTimeout = null;
                }
                return;
            }

            // Appeler les handlers globaux
            this.globalHandlers.forEach(handler => handler(message.data, message));

            // Appeler les handlers spécifiques au type
            const handlers = this.messageHandlers.get(message.type);
            if (handlers) {
                handlers.forEach(handler => handler(message.data, message));
            }
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }

    private handleOpen() {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;

        if (this.token) {
            this.ws?.send(JSON.stringify({
                type: 'auth',
                token: this.token,
                timestamp: new Date().toISOString()
            }));
        }

        this.startHeartbeat();
        this.notifyStatusChange();
    }

    private handleClose(event: CloseEvent) {
        console.log(`WebSocket disconnected: ${event.code} - ${event.reason}`);
        this.stopHeartbeat();
        this.notifyStatusChange();

        if (event.code !== 1000) {
            this.reconnect();
        }
    }

    private handleError(error: Event) {
        console.error('WebSocket error:', error);
        const status = this.getStatus();
        const errorStatus = { ...status, error: 'Connection error', connected: false };
        this.statusHandlers.forEach(handler => handler(errorStatus));
    }

    private reconnect() {
        if (this.reconnectTimeout) return;

        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnection attempts reached');
            const status = this.getStatus();
            const errorStatus = { ...status, error: 'Max reconnection attempts reached', connected: false };
            this.statusHandlers.forEach(handler => handler(errorStatus));
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        const status = this.getStatus();
        const reconnectingStatus = { ...status, reconnecting: true, connected: false };
        this.statusHandlers.forEach(handler => handler(reconnectingStatus));

        this.reconnectTimeout = setTimeout(() => {
            this.reconnectTimeout = null;
            this.connect(this.token || undefined);
        }, delay);
    }

    connect(token?: string): void {
        if (this.ws?.readyState === WebSocket.OPEN) {
            return;
        }

        if (this.reconnectTimeout) {
            return;
        }

        this.token = token || null;

        let wsUrl = this.url;
        if (token) {
            wsUrl += `?token=${encodeURIComponent(token)}`;
        }

        try {
            this.ws = new WebSocket(wsUrl);
            this.ws.onopen = () => this.handleOpen();
            this.ws.onclose = (event) => this.handleClose(event);
            this.ws.onerror = (error) => this.handleError(error);
            this.ws.onmessage = (event) => this.handleMessage(event);
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.reconnect();
        }
    }

    disconnect(): void {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }

        if (this.ws) {
            this.ws.close(1000, 'Normal closure');
            this.ws = null;
        }

        this.stopHeartbeat();
        this.notifyStatusChange();
    }

    send(message: Partial<Omit<WebSocketMessage, 'timestamp'>>): boolean {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                ...message,
                timestamp: new Date().toISOString()
            }));
            return true;
        }
        return false;
    }

    // Alias pour subscribe (compatible avec l'ancienne API 'on')
    on(type: WebSocketEventType, handler: MessageHandler): () => void {
        return this.subscribe(type, handler);
    }

    subscribe(type: WebSocketEventType, handler: MessageHandler): () => void {
        if (!this.messageHandlers.has(type)) {
            this.messageHandlers.set(type, new Set());
        }
        this.messageHandlers.get(type)!.add(handler);

        return () => {
            this.messageHandlers.get(type)?.delete(handler);
        };
    }

    onGlobal(handler: MessageHandler): () => void {
        return this.subscribeGlobal(handler);
    }

    subscribeGlobal(handler: MessageHandler): () => void {
        this.globalHandlers.add(handler);
        return () => {
            this.globalHandlers.delete(handler);
        };
    }

    onStatusChange(handler: (status: WebSocketConnectionStatus) => void): () => void {
        this.statusHandlers.add(handler);
        handler(this.getStatus());
        return () => {
            this.statusHandlers.delete(handler);
        };
    }

    isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }

    getReadyState(): number {
        return this.ws?.readyState ?? WebSocket.CLOSED;
    }

    getReconnectAttempts(): number {
        return this.reconnectAttempts;
    }

    resetReconnectAttempts(): void {
        this.reconnectAttempts = 0;
    }
}

export const websocketService = new WebSocketService();

export const WebSocketReadyState = {
    CONNECTING: 0,
    OPEN: 1,
    CLOSING: 2,
    CLOSED: 3
} as const;