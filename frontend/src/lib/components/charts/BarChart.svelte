<!-- lib/components/charts/BarChart.svelte -->
<script lang="ts">
  import { onMount } from "svelte";

  export let labels: string[] = [];
  export let datasets: Array<{
    label: string;
    data: number[];
    color?: string;
  }> = [];
  export let title: string = "";
  export let xAxisLabel: string = "";
  export let yAxisLabel: string = "";
  export let height: number = 300;
  export let horizontal: boolean = false;

  let chartContainer: HTMLCanvasElement;
  let chart: any = null;

  const defaultColors = [
    "#3b82f6",
    "#ef4444",
    "#10b981",
    "#f59e0b",
    "#8b5cf6",
    "#ec4899",
    "#06b6d4",
    "#84cc16",
    "#f97316",
    "#6366f1",
  ];

  onMount(() => {
    async function initChart() {
      const { default: Chart } = await import("chart.js/auto");

      const ctx = chartContainer?.getContext("2d");

      if (!ctx) return;

      chart = new Chart(ctx, {
        type: "bar",
        data: {
          labels,
          datasets: datasets.map((ds, idx) => ({
            label: ds.label,
            data: ds.data,
            backgroundColor:
              ds.color || `${defaultColors[idx % defaultColors.length]}80`,
            borderColor: ds.color || defaultColors[idx % defaultColors.length],
            borderWidth: 1,
            borderRadius: 4,
          })),
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          indexAxis: horizontal ? "y" : "x",
          plugins: {
            legend: {
              position: "top",
            },
            tooltip: {
              mode: "index",
              intersect: false,
            },
          },
          scales: {
            x: {
              title: {
                display: !!xAxisLabel && !horizontal,
                text: xAxisLabel,
              },
            },
            y: {
              title: {
                display: !!yAxisLabel && horizontal,
                text: yAxisLabel,
              },
              beginAtZero: true,
            },
          },
        },
      });
    }

    initChart();

    return () => {
      chart?.destroy();
    };
  });

  $: if (chart && labels.length > 0) {
    chart.data.labels = labels;
    chart.data.datasets = datasets.map((ds, idx) => ({
      ...chart.data.datasets[idx],
      label: ds.label,
      data: ds.data,
      backgroundColor:
        ds.color || defaultColors[idx % defaultColors.length] + "80",
      borderColor: ds.color || defaultColors[idx % defaultColors.length],
    }));
    chart.update();
  }
</script>

<div class="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
  {#if title}
    <h3 class="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
  {/if}
  <div style="height: {height}px">
    <canvas bind:this={chartContainer}></canvas>
  </div>
</div>
