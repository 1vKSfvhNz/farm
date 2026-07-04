<!-- lib/components/charts/WaterQualityChart.svelte -->
<script lang="ts">
    import { onMount } from "svelte";

    export let data: Array<{
        timestamp: string;
        ph: number;
        temperature: number;
        oxygene_dissous: number;
        ammoniac: number;
        nitrites?: number;
        nitrates?: number;
    }> = [];

    export let title = "";
    export let height = 400;

    let canvas: HTMLCanvasElement;
    let chart: any;

    const thresholds = {
        ph: { min: 6.5, max: 8.5 },
        temperature: { min: 10, max: 25 },
        oxygene_dissous: { min: 5 },
        ammoniac: { max: 0.5 },
        nitrites: { max: 0.5 },
        nitrates: { max: 50 },
    };

    onMount(() => {
        let destroyed = false;

        async function initChart() {
            const { Chart } = await import("chart.js/auto");

            if (destroyed || !canvas) return;

            const labels = data.map((d) => {
                const date = new Date(d.timestamp);

                return `${date.toLocaleDateString("fr-FR")} ${date.toLocaleTimeString(
                    "fr-FR",
                    {
                        hour: "2-digit",
                        minute: "2-digit",
                    },
                )}`;
            });

            chart = new Chart(canvas, {
                type: "line",
                data: {
                    labels,
                    datasets: [
                        {
                            label: "pH",
                            data: data.map((d) => d.ph),
                            borderColor: "#3b82f6",
                            backgroundColor: "#3b82f620",
                            tension: 0.4,
                            yAxisID: "y",
                        },
                        {
                            label: "Température (°C)",
                            data: data.map((d) => d.temperature),
                            borderColor: "#f59e0b",
                            backgroundColor: "#f59e0b20",
                            tension: 0.4,
                            yAxisID: "y",
                        },
                        {
                            label: "Oxygène dissous (mg/L)",
                            data: data.map((d) => d.oxygene_dissous),
                            borderColor: "#10b981",
                            backgroundColor: "#10b98120",
                            tension: 0.4,
                            yAxisID: "y1",
                        },
                        {
                            label: "Ammoniac (mg/L)",
                            data: data.map((d) => d.ammoniac),
                            borderColor: "#ef4444",
                            backgroundColor: "#ef444420",
                            tension: 0.4,
                            yAxisID: "y1",
                        },
                        {
                            label: "Nitrites (mg/L)",
                            data: data.map((d) => d.nitrites ?? 0),
                            borderColor: "#8b5cf6",
                            backgroundColor: "#8b5cf620",
                            tension: 0.4,
                            yAxisID: "y1",
                        },
                        {
                            label: "Nitrates (mg/L)",
                            data: data.map((d) => d.nitrates ?? 0),
                            borderColor: "#ec4899",
                            backgroundColor: "#ec489920",
                            tension: 0.4,
                            yAxisID: "y1",
                        },

                        /* Seuils */
                        {
                            label: "pH min",
                            data: labels.map(() => thresholds.ph.min),
                            borderColor: "#dc2626",
                            borderDash: [5, 5],
                            pointRadius: 0,
                            yAxisID: "y",
                        },
                        {
                            label: "pH max",
                            data: labels.map(() => thresholds.ph.max),
                            borderColor: "#dc2626",
                            borderDash: [5, 5],
                            pointRadius: 0,
                            yAxisID: "y",
                        },
                        {
                            label: "O₂ min",
                            data: labels.map(
                                () => thresholds.oxygene_dissous.min,
                            ),
                            borderColor: "#dc2626",
                            borderDash: [5, 5],
                            pointRadius: 0,
                            yAxisID: "y1",
                        },
                        {
                            label: "Ammoniac max",
                            data: labels.map(() => thresholds.ammoniac.max),
                            borderColor: "#dc2626",
                            borderDash: [5, 5],
                            pointRadius: 0,
                            yAxisID: "y1",
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,

                    interaction: {
                        mode: "index",
                        intersect: false,
                    },

                    plugins: {
                        title: {
                            display: Boolean(title),
                            text: title,
                        },

                        legend: {
                            position: "top",
                        },
                    },

                    scales: {
                        y: {
                            type: "linear",
                            position: "left",
                            title: {
                                display: true,
                                text: "pH / Température",
                            },
                        },

                        y1: {
                            type: "linear",
                            position: "right",

                            title: {
                                display: true,
                                text: "Concentration (mg/L)",
                            },

                            grid: {
                                drawOnChartArea: false,
                            },
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
