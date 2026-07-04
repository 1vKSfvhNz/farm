<!-- lib/components/ui/Button.svelte -->
<script lang="ts">
    export let variant:
        | "primary"
        | "secondary"
        | "danger"
        | "success"
        | "warning"
        | "outline"
        | "ghost" = "primary";
    export let size: "sm" | "md" | "lg" = "md";
    export let disabled: boolean = false;
    export let loading: boolean = false;
    export let type: "button" | "submit" | "reset" = "button";
    export let fullWidth: boolean = false;
    export let icon: string | null = null;
    export let className: string = "";

    $: {
        const baseClasses =
            "inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed";

        const variants = {
            primary:
                "bg-primary-600 hover:bg-primary-700 text-white focus:ring-primary-500",
            secondary:
                "bg-gray-600 hover:bg-gray-700 text-white focus:ring-gray-500",
            danger: "bg-red-600 hover:bg-red-700 text-white focus:ring-red-500",
            success:
                "bg-green-600 hover:bg-green-700 text-white focus:ring-green-500",
            warning:
                "bg-yellow-500 hover:bg-yellow-600 text-white focus:ring-yellow-400",
            outline:
                "border border-gray-300 bg-white hover:bg-gray-50 text-gray-700 focus:ring-primary-500",
            ghost: "hover:bg-gray-100 text-gray-600 focus:ring-gray-400",
        };

        const sizes = {
            sm: "px-3 py-1.5 text-sm",
            md: "px-4 py-2 text-base",
            lg: "px-6 py-3 text-lg",
        };

        const widthClass = fullWidth ? "w-full" : "";

        className = `${baseClasses} ${variants[variant]} ${sizes[size]} ${widthClass} ${className}`;
    }
</script>

<button {type} {disabled} class={className} on:click on:keydown>
    {#if loading}
        <svg
            class="animate-spin -ml-1 mr-2 h-4 w-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
        >
            <circle
                class="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                stroke-width="4"
            ></circle>
            <path
                class="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            ></path>
        </svg>
    {/if}
    {#if icon && !loading}
        <i class="icon-{icon} -ml-1 mr-2"></i>
    {/if}
    <slot />
</button>
