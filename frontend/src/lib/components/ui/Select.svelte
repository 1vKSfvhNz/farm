<!-- lib/components/ui/Select.svelte -->
<!-- svelte-ignore a11y-label-has-associated-control -->
<script lang="ts">
    export let value: string | number = "";
    export let label: string | null = null;
    export let options: Array<{ value: string | number; label: string }> = [];
    export let placeholder: string = "Sélectionner...";
    export let required: boolean = false;
    export let disabled: boolean = false;
    export let error: string | null = null;
    export let className = "";

    let selectClass = "";

    $: {
        const baseClasses =
            "w-full rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 px-4 py-2 transition-all duration-200";

        const errorClasses = error ? "border-red-500" : "";

        selectClass = `${baseClasses} ${errorClasses} ${className}`;
    }

    function handleChange(e: Event) {
        const newValue = (e.target as HTMLSelectElement).value;
        value = newValue;
        // Émettre un événement personnalisé
        const changeEvent = new CustomEvent('change', { detail: newValue });
        dispatch('change', changeEvent);
    }

    import { createEventDispatcher } from 'svelte';
    const dispatch = createEventDispatcher();
</script>

<div class="mb-4">
    {#if label}
        <label class="block text-sm font-medium text-gray-700 mb-1">
            {label}
            {#if required}
                <span class="text-red-500 ml-1">*</span>
            {/if}
        </label>
    {/if}

    <select
        bind:value
        {disabled}
        {required}
        on:change={handleChange}
        class={selectClass}
    >
        {#if placeholder}
            <option value="" disabled>{placeholder}</option>
        {/if}
        {#each options as option}
            <option value={option.value}>{option.label}</option>
        {/each}
    </select>

    {#if error}
        <p class="mt-1 text-sm text-red-600">{error}</p>
    {/if}
</div>