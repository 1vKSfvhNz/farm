<!-- lib/components/video/VideoPlayer.svelte -->
<script lang="ts">
    import { onMount } from "svelte";

    export let src: string;
    export let poster: string = "";
    export let autoplay: boolean = false;
    export let controls: boolean = true;
    export let loop: boolean = false;
    export let muted: boolean = false;
    export let width: string = "100%";
    export let height: string = "auto";

    let videoRef: HTMLVideoElement;
    let isPlaying = false;
    let currentTime = 0;
    let duration = 0;
    let volume = 1;

    onMount(() => {
        if (videoRef) {
            videoRef.addEventListener("timeupdate", () => {
                currentTime = videoRef.currentTime;
            });
            videoRef.addEventListener("loadedmetadata", () => {
                duration = videoRef.duration;
            });
        }

        return () => {
            if (videoRef) {
                videoRef.pause();
            }
        };
    });

    function togglePlay() {
        if (isPlaying) {
            videoRef.pause();
        } else {
            videoRef.play();
        }
        isPlaying = !isPlaying;
    }

    function handleSeek(e: Event) {
        const target = e.target as HTMLInputElement;
        videoRef.currentTime = parseFloat(target.value);
    }

    function handleVolumeChange(e: Event) {
        const target = e.target as HTMLInputElement;
        volume = parseFloat(target.value);
        videoRef.volume = volume;
    }

    function formatTime(seconds: number): string {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, "0")}`;
    }
</script>

<div class="bg-black rounded-xl overflow-hidden shadow-lg">
    <video
        bind:this={videoRef}
        {src}
        {poster}
        {autoplay}
        {controls}
        {loop}
        {muted}
        style="width: {width}; height: {height}"
        class="w-full"
    />

    {#if controls}
        <div class="p-3 bg-gray-900 flex items-center gap-4">
            <button
                on:click={togglePlay}
                class="text-white hover:text-primary-400 transition-colors"
            >
                {#if isPlaying}
                    <svg
                        class="w-5 h-5"
                        fill="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                    </svg>
                {:else}
                    <svg
                        class="w-5 h-5"
                        fill="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path d="M8 5v14l11-7z" />
                    </svg>
                {/if}
            </button>

            <span class="text-white text-sm"
                >{formatTime(currentTime)} / {formatTime(duration)}</span
            >

            <input
                type="range"
                min="0"
                max={duration || 0}
                step="0.01"
                value={currentTime}
                on:input={handleSeek}
                class="flex-1 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
            />

            <div class="flex items-center gap-2">
                <svg
                    class="w-4 h-4 text-white"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path d="M3 9v6h4l5 5V4L7 9H3z" />
                </svg>
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={volume}
                    on:input={handleVolumeChange}
                    class="w-20 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                />
            </div>
        </div>
    {/if}
</div>
