<!-- lib/components/ui/AlertBadge.svelte -->
<script lang="ts">
    export let niveau: "info" | "warning" | "critical" = "info";
    export let size: "sm" | "md" = "md";
    export let label: string;

    const config = {
        info: {
            bg: "bg-blue-100",
            text: "text-blue-800",
            dot: "bg-blue-500"
        },
        warning: {
            bg: "bg-yellow-100",
            text: "text-yellow-800",
            dot: "bg-yellow-500"
        },
        critical: {
            bg: "bg-red-100",
            text: "text-red-800",
            dot: "bg-red-500"
        },
    };

    const sizes = {
        sm: "px-2 py-0.5 text-xs",
        md: "px-3 py-1 text-sm",
    };

    const current = config[niveau];
</script>

<span
    class="inline-flex items-center gap-1.5 rounded-full font-medium {sizes[
        size
    ]} {current.bg} {current.text}"
>
    <span class="w-1.5 h-1.5 rounded-full {current.dot}"></span>
    {label}
</span>
