<!-- lib/components/charts/PieChart.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import type { Chart as ChartType } from "chart.js";

    export let labels: string[] = [];
    export let data: number[] = [];
    export let title: string = "";
    export let height: number = 300;
    export let donut: boolean = false;

    let chartContainer: HTMLCanvasElement;
    let chart: ChartType | null = null;

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
                type: donut ? "doughnut" : "pie",
                data: {
                    labels,
                    datasets: [
                        {
                            data,
                            backgroundColor: defaultColors.slice(
                                0,
                                data.length,
                            ),
                            borderWidth: 0,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: "right",
                        },
                        tooltip: {
                            callbacks: {
                                label: (context: any) => {
                                    const label = context.label || "";
                                    const value = context.raw || 0;

                                    const total = context.dataset.data.reduce(
                                        (a: number, b: number) => a + b,
                                        0,
                                    );

                                    const percentage = (
                                        (value / total) *
                                        100
                                    ).toFixed(1);

                                    return `${label}: ${value} (${percentage}%)`;
                                },
                            },
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

        if (chart.data.datasets?.[0]) {
            chart.data.datasets[0].data = data;
        }

        chart.update();
    }
</script>

<div class="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
    {#if title}
        <h3 class="text-lg font-semibold text-gray-900 mb-4">
            {title}
        </h3>
    {/if}

    <div style="height: {height}px">
        <canvas bind:this={chartContainer}></canvas>
    </div>
</div>
