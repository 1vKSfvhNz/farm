<!-- lib/components/video/LiveView.svelte -->
<script lang="ts">
    import { onDestroy } from "svelte";
    import VideoPlayer from "./VideoPlayer.svelte";

    export let streamUrl: string;
    export let cameraName: string;
    export let autoReconnect: boolean = true;

    let isConnected = true;
    let reconnectAttempts = 0;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
    let streamKey = Date.now();

    function reconnect() {
        if (reconnectAttempts >= 5) {
            isConnected = false;
            return;
        }

        reconnectAttempts++;
        reconnectTimeout = setTimeout(() => {
            streamKey = Date.now();
            isConnected = true;
        }, 3000 * reconnectAttempts);
    }

    function handleVideoError() {
        isConnected = false;
        if (autoReconnect) {
            reconnect();
        }
    }

    function handleRetry() {
        reconnectAttempts = 0;
        reconnect();
    }

    onDestroy(() => {
        if (reconnectTimeout) {
            clearTimeout(reconnectTimeout);
        }
    });
</script>

<div class="bg-gray-900 rounded-xl overflow-hidden shadow-lg">
    <div class="relative">
        {#if isConnected}
            <VideoPlayer
                src={streamUrl + (streamKey ? `?_=${streamKey}` : "")}
                controls={true}
                on:error={handleVideoError}
            />
        {:else}
            <div
                class="aspect-video flex flex-col items-center justify-center bg-gray-800"
            >
                <svg
                    class="w-16 h-16 text-gray-600 mb-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                </svg>
                <p class="text-white text-lg font-medium">Connexion perdue</p>
                <p class="text-gray-400 text-sm mt-1">
                    Tentative de reconnexion...
                </p>
                <button
                    on:click={handleRetry}
                    class="mt-4 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
                >
                    Réessayer
                </button>
            </div>
        {/if}

        <div class="absolute top-4 left-4 flex items-center gap-2">
            <span class="w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
            <span
                class="text-white text-xs font-medium bg-black/50 px-2 py-1 rounded"
                >LIVE</span
            >
        </div>

        <div class="absolute bottom-4 left-4 bg-black/50 rounded-lg px-3 py-1">
            <span class="text-white text-sm">{cameraName}</span>
        </div>
    </div>
</div>
