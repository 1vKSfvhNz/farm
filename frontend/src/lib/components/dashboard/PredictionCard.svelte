<!-- lib/components/dashboard/PredictionCard.svelte -->
<script lang="ts">
  export let title: string;
  export let value: string | number;
  export let confidence: number;
  export let description: string | null = null;
  export let icon: string;

  export const trend: string | null = null;
  export const trendPositive: boolean = true;

  let confidenceColor = "";
  let confidenceLabel = "";

  $: {
    if (confidence >= 80) {
      confidenceColor = "bg-green-100 text-green-800";
      confidenceLabel = "Haute";
    } else if (confidence >= 60) {
      confidenceColor = "bg-yellow-100 text-yellow-800";
      confidenceLabel = "Moyenne";
    } else {
      confidenceColor = "bg-orange-100 text-orange-800";
      confidenceLabel = "Faible";
    }
  }
</script>

<div
  class="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl shadow-lg overflow-hidden"
>
  <div class="p-5">
    <div class="flex items-start justify-between">
      <div>
        <p class="text-sm font-medium text-gray-400">{title}</p>
        <p class="text-2xl font-bold text-white mt-1">{value}</p>
        {#if description}
          <p class="text-xs text-gray-400 mt-1">{description}</p>
        {/if}
      </div>
      <div
        class="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center"
      >
        <svg
          class="w-5 h-5 text-gray-300"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d={icon}
          />
        </svg>
      </div>
    </div>

    <div class="mt-4 pt-3 border-t border-gray-700">
      <div class="flex items-center justify-between">
        <span class="text-xs text-gray-400">Niveau de confiance</span>
        <div class="flex items-center gap-2">
          <div class="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              class="h-full rounded-full transition-all duration-500"
              class:bg-green-500={confidence >= 80}
              class:bg-yellow-500={confidence >= 60 && confidence < 80}
              class:bg-orange-500={confidence < 60}
              style="width: {confidence}%"
            ></div>
          </div>
          <span class="text-xs px-1.5 py-0.5 rounded {confidenceColor}">
            {confidenceLabel} ({confidence}%)
          </span>
        </div>
      </div>
    </div>
  </div>
</div>
