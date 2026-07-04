<!-- lib/components/dashboard/KpiCard.svelte -->
<script lang="ts">
    export let title: string;
    export let value: string | number;
    export let icon: string;
    export let trend: "up" | "down" | "stable" | null = null;
    export let trendValue: string | null = null;
    export let color: "primary" | "success" | "warning" | "danger" | "info" =
        "primary";

    const colors = {
        primary: {
            bg: "bg-primary-50",
            border: "border-primary-200",
            icon: "text-primary-600",
        },
        success: {
            bg: "bg-green-50",
            border: "border-green-200",
            icon: "text-green-600",
        },
        warning: {
            bg: "bg-yellow-50",
            border: "border-yellow-200",
            icon: "text-yellow-600",
        },
        danger: {
            bg: "bg-red-50",
            border: "border-red-200",
            icon: "text-red-600",
        },
        info: {
            bg: "bg-blue-50",
            border: "border-blue-200",
            icon: "text-blue-600",
        },
    };

    const trendIcons = {
        up: "M13 7h8m0 0v8m0-8l-8 8-4-4-6 6",
        down: "M13 17h8m0 0v-8m0 8l-8-8-4 4-6-6",
        stable: "M5 12h14",
    };

    const trendColors = {
        up: "text-green-600",
        down: "text-red-600",
        stable: "text-yellow-600",
    };
</script>

<div
    class="bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-md transition-all duration-200"
>
    <div class="p-5">
        <div class="flex items-start justify-between">
            <div>
                <p class="text-sm font-medium text-gray-500">{title}</p>
                <p class="text-2xl font-bold text-gray-900 mt-1">{value}</p>
            </div>
            <div
                class="w-10 h-10 rounded-lg {colors[color].bg} border {colors[
                    color
                ].border} flex items-center justify-center"
            >
                <svg
                    class="w-5 h-5 {colors[color].icon}"
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

        {#if trend && trendValue}
            <div class="flex items-center gap-1 mt-3">
                <svg
                    class="w-4 h-4 {trendColors[trend]}"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d={trendIcons[trend]}
                    />
                </svg>
                <span class="text-sm {trendColors[trend]}">{trendValue}</span>
                <span class="text-sm text-gray-400 ml-1">vs mois dernier</span>
            </div>
        {/if}
    </div>
</div>
