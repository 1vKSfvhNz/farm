<!-- lib/components/dashboard/ConfidenceIndicator.svelte -->
<script lang="ts">
    export let confidence: number;
    export let size: "sm" | "md" | "lg" = "md";
    export let showLabel: boolean = true;

    let confidenceColor = "";
    let confidenceLabel = "";
    let sizeClass = "";

    $: {
        if (confidence >= 80) {
            confidenceColor = "text-green-600 bg-green-100";
            confidenceLabel = "Haute";
        } else if (confidence >= 60) {
            confidenceColor = "text-yellow-600 bg-yellow-100";
            confidenceLabel = "Moyenne";
        } else {
            confidenceColor = "text-orange-600 bg-orange-100";
            confidenceLabel = "Faible";
        }

        const sizes = {
            sm: "w-16 h-1.5",
            md: "w-24 h-2",
            lg: "w-32 h-2.5",
        };
        sizeClass = sizes[size];
    }
</script>

<div class="flex items-center gap-3">
    <div class="bg-gray-200 rounded-full overflow-hidden {sizeClass}">
        <div
            class="h-full rounded-full transition-all duration-500"
            class:bg-green-500={confidence >= 80}
            class:bg-yellow-500={confidence >= 60 && confidence < 80}
            class:bg-orange-500={confidence < 60}
            style="width: {confidence}%"
        ></div>
    </div>
    {#if showLabel}
        <span
            class="text-xs font-medium px-2 py-0.5 rounded-full {confidenceColor}"
        >
            {confidenceLabel} ({confidence}%)
        </span>
    {/if}
</div>
