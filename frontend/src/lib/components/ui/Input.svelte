<!-- lib/components/ui/Input.svelte -->
<script lang="ts">
    export let value: string | number = "";
    export let label: string | null = null;
    export let inputType:
        | "text"
        | "email"
        | "password"
        | "number"
        | "tel"
        | "url" = "text";
    export let placeholder: string = "";
    export let required: boolean = false;
    export let disabled: boolean = false;
    export let error: string | null = null;
    export let hint: string | null = null;
    export let icon: string | null = null;
    export let iconPosition: "left" | "right" = "left";
    export let step: string = "any";
    export let min: number | string | null = null;
    export let max: number | string | null = null;
    export let rows: number = 3;
    export let textarea: boolean = false;

    let inputClass = "";

    $: {
        const baseClasses =
            "w-full rounded-lg border transition-all duration-200 focus:outline-none focus:ring-2";
        const errorClasses = error
            ? "border-red-500 focus:border-red-500 focus:ring-red-100"
            : "border-gray-300 focus:border-primary-500 focus:ring-primary-100";
        const iconPadding = icon
            ? iconPosition === "left"
                ? "pl-10"
                : "pr-10"
            : "";
        inputClass = `${baseClasses} ${errorClasses} ${iconPadding} px-4 py-2`;
    }

    function handleInput(e: Event) {
        value = (e.target as HTMLInputElement).value;
    }

    function handleTextareaInput(e: Event) {
        value = (e.target as HTMLTextAreaElement).value;
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

    <div class="relative">
        {#if icon && iconPosition === "left"}
            <div
                class="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"
            >
                <i class="icon-{icon} text-gray-400"></i>
            </div>
        {/if}

        {#if textarea}
            <textarea
                {placeholder}
                {disabled}
                {required}
                {rows}
                bind:value
                on:input={handleTextareaInput}
                class={inputClass}
            />
        {:else if inputType === "text"}
            <input
                type="text"
                {placeholder}
                {disabled}
                {required}
                bind:value
                on:input={handleInput}
                class={inputClass}
            />
        {:else if inputType === "email"}
            <input
                type="email"
                {placeholder}
                {disabled}
                {required}
                bind:value
                on:input={handleInput}
                class={inputClass}
            />
        {:else if inputType === "password"}
            <input
                type="password"
                {placeholder}
                {disabled}
                {required}
                bind:value
                on:input={handleInput}
                class={inputClass}
            />
        {:else if inputType === "number"}
            <input
                type="number"
                {placeholder}
                {disabled}
                {required}
                {step}
                {min}
                {max}
                bind:value
                on:input={handleInput}
                class={inputClass}
            />
        {:else if inputType === "tel"}
            <input
                type="tel"
                {placeholder}
                {disabled}
                {required}
                bind:value
                on:input={handleInput}
                class={inputClass}
            />
        {:else if inputType === "url"}
            <input
                type="url"
                {placeholder}
                {disabled}
                {required}
                bind:value
                on:input={handleInput}
                class={inputClass}
            />
        {/if}

        {#if icon && iconPosition === "right"}
            <div
                class="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none"
            >
                <i class="icon-{icon} text-gray-400"></i>
            </div>
        {/if}
    </div>

    {#if error}
        <p class="mt-1 text-sm text-red-600">{error}</p>
    {:else if hint}
        <p class="mt-1 text-sm text-gray-500">{hint}</p>
    {/if}
</div>
