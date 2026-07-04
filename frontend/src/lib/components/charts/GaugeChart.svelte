<!-- lib/components/charts/GaugeChart.svelte -->
<script lang="ts">
  export let value: number = 0;
  export let min: number = 0;
  export let max: number = 100;
  export let title: string = "";
  export let unit: string = "";
  export let size: "sm" | "md" | "lg" = "md";

  let percentage = 0;
  let color = "";
  let angle = 0;

  const sizes = {
    sm: "w-32 h-32",
    md: "w-48 h-48",
    lg: "w-64 h-64",
  };

  $: {
    percentage = Math.min(
      100,
      Math.max(0, ((value - min) / (max - min)) * 100),
    );

    if (percentage >= 80) {
      color = "#10b981";
    } else if (percentage >= 50) {
      color = "#f59e0b";
    } else {
      color = "#ef4444";
    }

    angle = (percentage / 100) * 180;
  }
</script>

<div
  class="bg-white rounded-xl border border-gray-200 p-4 shadow-sm text-center"
>
  {#if title}
    <h3 class="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
  {/if}

  <div class="flex justify-center">
    <div class="relative {sizes[size]}">
      <svg class="w-full h-full" viewBox="0 0 200 120">
        <!-- Arc de fond -->
        <path
          d="M 30,100 A 70,70 0 0,1 170,100"
          fill="none"
          stroke="#e5e7eb"
          stroke-width="15"
          stroke-linecap="round"
        />
        <!-- Arc de progression -->
        <path
          d="M 30,100 A 70,70 0 0,1 {170 - (angle / 180) * 140},100"
          fill="none"
          stroke={color}
          stroke-width="15"
          stroke-linecap="round"
          stroke-dasharray="220"
          stroke-dashoffset={220 - (percentage / 100) * 220}
          transform="rotate(180, 100, 100)"
        />
        <!-- Aiguille -->
        <line
          x1="100"
          y1="100"
          x2={100 + 60 * Math.cos(((angle - 90) * Math.PI) / 180)}
          y2={100 + 60 * Math.sin(((angle - 90) * Math.PI) / 180)}
          stroke="#374151"
          stroke-width="3"
          stroke-linecap="round"
        />
        <circle cx="100" cy="100" r="8" fill="#374151" />
      </svg>
      <div class="absolute bottom-0 left-0 right-0 text-center">
        <span class="text-2xl font-bold text-gray-800">{value}</span>
        {#if unit}
          <span class="text-sm text-gray-500 ml-1">{unit}</span>
        {/if}
      </div>
    </div>
  </div>
</div>
