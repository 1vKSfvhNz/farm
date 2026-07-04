<!-- lib/components/tables/ExportButton.svelte -->
<script lang="ts">
    export let exportFn: () => Promise<Blob>;
    export let filename: string;
    export let label: string = "Exporter";
    export let variant: "primary" | "secondary" | "outline" = "outline";
    export let size: "sm" | "md" = "sm";

    let isExporting = false;

    const variants = {
        primary: "bg-primary-600 hover:bg-primary-700 text-white",
        secondary: "bg-gray-600 hover:bg-gray-700 text-white",
        outline: "border border-gray-300 hover:bg-gray-50 text-gray-700",
    };

    const sizes = {
        sm: "px-3 py-1.5 text-sm",
        md: "px-4 py-2 text-base",
    };

    async function handleExport() {
        isExporting = true;
        try {
            const blob = await exportFn();
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error("Export failed:", error);
            window.dispatchEvent(
                new CustomEvent("show-toast", {
                    detail: {
                        message: "Erreur lors de l'export",
                        type: "error",
                    },
                }),
            );
        } finally {
            isExporting = false;
        }
    }
</script>

<button
    on:click={handleExport}
    disabled={isExporting}
    class="inline-flex items-center gap-2 rounded-lg font-medium transition-all duration-200 disabled:opacity-50 {variants[
        variant
    ]} {sizes[size]}"
>
    {#if isExporting}
        <svg
            class="animate-spin w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
        >
            <circle
                class="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                stroke-width="4"
            ></circle>
            <path
                class="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            ></path>
        </svg>
    {:else}
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
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
            />
        </svg>
    {/if}
    {label}
</button>
