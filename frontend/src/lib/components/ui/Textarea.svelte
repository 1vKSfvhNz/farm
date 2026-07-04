<!-- src/lib/components/ui/Textarea.svelte -->
<script lang="ts">
    export let value: string = "";
    export let label: string = "";
    export let placeholder: string = "";
    export let rows: number = 3;
    export let required: boolean = false;
    export let disabled: boolean = false;
    export let helpText: string = "";
    export let error: string = "";
    export let className: string = "";
    export let id: string = "";
    export let name: string = "";

    // Générer un ID unique si non fourni
    $: textareaId = id || `textarea-${Math.random().toString(36).slice(2, 11)}`;
</script>

<div class="space-y-1.5 {className}">
    {#if label}
        <label for={textareaId} class="block text-sm font-medium text-gray-700">
            {label}
            {#if required}
                <span class="text-red-500 ml-0.5">*</span>
            {/if}
        </label>
    {/if}
    
    <textarea
        {id}
        {name}
        bind:value
        {rows}
        {placeholder}
        {required}
        {disabled}
        class="w-full rounded-lg border {error ? 'border-red-300 focus:border-red-500 focus:ring-red-100' : 'border-gray-300 focus:border-primary-500 focus:ring-primary-100'} 
               focus:ring-2 px-4 py-2 transition-all duration-200
               disabled:bg-gray-100 disabled:cursor-not-allowed
               {error ? 'bg-red-50' : ''}"
        aria-describedby={helpText || error ? `${textareaId}-description` : undefined}
        aria-invalid={!!error}
    />
    
    {#if helpText && !error}
        <p id={`${textareaId}-description`} class="text-xs text-gray-500">
            {helpText}
        </p>
    {/if}
    
    {#if error}
        <p id={`${textareaId}-description`} class="text-xs text-red-600">
            {error}
        </p>
    {/if}
</div>