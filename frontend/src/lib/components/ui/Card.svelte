<!-- lib/components/ui/Card.svelte -->
<script lang="ts">
    export let title: string | null = null;
    export let subtitle: string | null = null;
    export let padding: "none" | "sm" | "md" | "lg" = "md";
    export let hoverable: boolean = false;
    export let bordered: boolean = true;
    export let className = "";
    let paddingClass = "";

    $: {
        const paddings = {
            none: "p-0",
            sm: "p-3",
            md: "p-5",
            lg: "p-8",
        };
        paddingClass = paddings[padding];
    }
</script>

<div
    class={`bg-white rounded-xl shadow-sm
    ${bordered ? "border border-gray-200" : ""}
    ${hoverable ? "hover:shadow-md transition-shadow duration-200" : ""}
    ${className}`}
>
    {#if title}
        <div class="border-b border-gray-200 {paddingClass}">
            <h3 class="text-lg font-semibold text-gray-900">{title}</h3>
            {#if subtitle}
                <p class="mt-1 text-sm text-gray-500">{subtitle}</p>
            {/if}
        </div>
    {/if}
    <div class={paddingClass}>
        <slot />
    </div>
    {#if $$slots.footer}
        <div
            class="border-t border-gray-200 {paddingClass} bg-gray-50 rounded-b-xl"
        >
            <slot name="footer" />
        </div>
    {/if}
</div>
