<!-- lib/components/ui/DatePicker.svelte -->
<script lang="ts">
    export let value: string = "";
    export let label: string | null = null;
    export let placeholder: string = "JJ/MM/AAAA";
    export let required: boolean = false;
    export let disabled: boolean = false;
    export let error: string | null = null;
    export let minDate: string | null = null;
    export let maxDate: string | null = null;

    let inputClass = "";

    $: {
        const baseClasses =
            "w-full rounded-lg border transition-all duration-200 focus:outline-none focus:ring-2";
        const errorClasses = error
            ? "border-red-500 focus:border-red-500 focus:ring-red-100"
            : "border-gray-300 focus:border-primary-500 focus:ring-primary-100";
        inputClass = `${baseClasses} ${errorClasses} px-4 py-2`;
    }

    function handleInput(e: Event) {
        value = (e.target as HTMLInputElement).value;
    }
</script>

<div class="mb-4">
    {#if label}
        <!-- svelte-ignore a11y-label-has-associated-control -->
        <label class="block text-sm font-medium text-gray-700 mb-1">
            {label}
            {#if required}
                <span class="text-red-500 ml-1">*</span>
            {/if}
        </label>
    {/if}

    <input
        type="date"
        {placeholder}
        {disabled}
        {required}
        bind:value
        on:input={handleInput}
        min={minDate || undefined}
        max={maxDate || undefined}
        class={inputClass}
    />

    {#if error}
        <p class="mt-1 text-sm text-red-600">{error}</p>
    {/if}
</div>
