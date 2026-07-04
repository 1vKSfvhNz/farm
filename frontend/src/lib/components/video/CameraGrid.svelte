<!-- lib/components/video/CameraGrid.svelte -->
<script lang="ts">
    import VideoPlayer from "./VideoPlayer.svelte";

    export let cameras: Array<{
        id: number;
        name: string;
        streamUrl: string;
        isActive: boolean;
    }> = [];
    export let columns: number = 4;
    export let showControls: boolean = true;

    let selectedCamera: (typeof cameras)[0] | null = null;
    let isFullscreen = false;

    function getGridClass(): string {
        const cols = Math.min(columns, cameras.length);
        const colClasses: Record<number, string> = {
            1: "grid-cols-1",
            2: "grid-cols-1 md:grid-cols-2",
            3: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
            4: "grid-cols-1 md:grid-cols-2 lg:grid-cols-4",
            5: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5",
            6: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6",
        };
        return colClasses[cols] || "grid-cols-1 md:grid-cols-2 lg:grid-cols-4";
    }

    function toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
            isFullscreen = true;
        } else {
            document.exitFullscreen();
            isFullscreen = false;
        }
    }
</script>

<div class="space-y-4">
    {#if selectedCamera && !isFullscreen}
        <div class="relative">
            <button
                on:click={() => (selectedCamera = null)}
                class="absolute top-2 right-2 z-10 p-1 bg-black/50 rounded-lg text-white hover:bg-black/70 transition-colors"
            >
                <svg
                    class="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M6 18L18 6M6 6l12 12"
                    />
                </svg>
            </button>
            <VideoPlayer src={selectedCamera.streamUrl} controls={true} />
            <h4 class="mt-2 text-sm font-medium text-gray-700">
                {selectedCamera.name}
            </h4>
        </div>
    {/if}

    <div class="grid gap-4 {getGridClass()}">
        {#each cameras as camera}
            {#if !selectedCamera || (selectedCamera && !isFullscreen && selectedCamera.id !== camera.id)}
                <!-- svelte-ignore a11y-no-static-element-interactions -->
                <!-- svelte-ignore a11y-click-events-have-key-events -->
                <div
                    class="relative bg-gray-900 rounded-lg overflow-hidden cursor-pointer group"
                    on:click={() => (selectedCamera = camera)}
                >
                    <div
                        class="aspect-video bg-gray-800 flex items-center justify-center"
                    >
                        {#if camera.isActive}
                            <img
                                src={camera.streamUrl}
                                alt={camera.name}
                                class="w-full h-full object-cover"
                            />
                        {:else}
                            <div class="text-center">
                                <svg
                                    class="w-12 h-12 text-gray-600 mx-auto"
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
                                <p class="text-sm text-gray-500 mt-2">
                                    Caméra hors ligne
                                </p>
                            </div>
                        {/if}
                    </div>
                    <div
                        class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-2 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                        <p class="text-white text-xs font-medium">
                            {camera.name}
                        </p>
                    </div>
                    {#if camera.isActive}
                        <div
                            class="absolute top-2 left-2 flex items-center gap-1"
                        >
                            <span
                                class="w-2 h-2 bg-green-500 rounded-full animate-pulse"
                            ></span>
                            <span class="text-xs text-white">LIVE</span>
                        </div>
                    {/if}
                </div>
            {/if}
        {/each}
    </div>

    {#if showControls && cameras.length > 0 && !selectedCamera}
        <div class="flex justify-end">
            <button
                on:click={toggleFullscreen}
                class="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
            >
                <svg
                    class="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
                    />
                </svg>
            </button>
        </div>
    {/if}
</div>
