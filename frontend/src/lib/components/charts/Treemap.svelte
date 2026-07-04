<!-- lib/components/charts/Treemap.svelte -->
<script lang="ts">
    import { onMount } from "svelte";

    export let data: Array<{
        name: string;
        value: number;
        parent?: string;
        color?: string;
    }> = [];

    export let title = "";
    export let height = 400;

    let canvas: HTMLCanvasElement;
    let chart: any;

    const colors = [
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
        let destroyed = false;

        async function initChart() {
            const { Chart } = await import("chart.js/auto");
            const { TreemapController, TreemapElement } =
                await import("chartjs-chart-treemap");

            Chart.register(TreemapController, TreemapElement);

            if (destroyed || !canvas) return;

            const total = data.reduce((sum, item) => sum + item.value, 0);

            chart = new Chart(canvas, {
                type: "treemap",
                data: {
                    datasets: [
                        {
                            tree: data.map((item) => ({
                                name: item.name,
                                value: item.value,
                                color: item.color,
                            })),
                            key: "value",
                            groups: ["name"],
                            backgroundColor(ctx: any) {
                                const index = ctx.dataIndex;
                                return (
                                    ctx.raw?._data?.color ??
                                    colors[index % colors.length]
                                );
                            },
                            borderColor: "#fff",
                            borderWidth: 2,
                        } as any,
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,

                    plugins: {
                        title: {
                            display: Boolean(title),
                            text: title,
                        },

                        tooltip: {
                            callbacks: {
                                label(context: any) {
                                    const item = context.raw._data;

                                    const percentage =
                                        total > 0
                                            ? (
                                                  (item.value / total) *
                                                  100
                                              ).toFixed(1)
                                            : "0";

                                    return [
                                        `Valeur : ${item.value.toLocaleString(
                                            "fr-FR",
                                        )} €`,
                                        `${percentage}% du total`,
                                    ];
                                },
                            },
                        },

                        legend: {
                            display: false,
                        },
                    },
                },
            });
        }

        initChart();

        return () => {
            destroyed = true;
            chart?.destroy();
        };
    });
</script>

<div class="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
    <div style="height: {height}px;">
        <canvas bind:this={canvas}></canvas>
    </div>
</div>
