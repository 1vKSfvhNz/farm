<!-- lib/components/ui/FloatingLabelInput.svelte - Version avec soulignement animé -->
<script lang="ts">
    export let value: string = "";
    export let label: string = "";
    export let type: "text" | "email" | "password" | "number" | "tel" = "text";
    export let name: string = "";
    export let required: boolean = false;
    export let disabled: boolean = false;
    export let error: string | null = null;
    export let placeholder: string = "";
    export let autocomplete: string = "off";
    export let icon: string | null = null;
    export let min: number | string | null = null;
    export let max: number | string | null = null;
    export let step: string = "any";
    
    let isFocused = false;
    let inputRef: HTMLInputElement;
    let showPassword = false;
    
    $: isFloating = isFocused || value.length > 0;
    $: inputType = type === "password" && showPassword ? "text" : type;
    
    const iconMap: Record<string, string> = {
        user: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>`,
        lock: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>`,
        email: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>`,
        search: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>`
    };
    
    function handleFocus() {
        isFocused = true;
    }
    
    function handleBlur() {
        isFocused = false;
    }
    
    function handleInput(e: Event) {
        value = (e.target as HTMLInputElement).value;
    }
    
    function togglePassword() {
        showPassword = !showPassword;
    }
</script>

<div class="relative mb-8">
    <div class="relative">        
        <input
            bind:this={inputRef}
            type={inputType}
            id={name}
            name={name}
            value={value}
            on:input={handleInput}
            on:focus={handleFocus}
            on:blur={handleBlur}
            disabled={disabled}
            required={required}
            placeholder={isFloating ? placeholder : ""}
            autocomplete={autocomplete}
            min={min || undefined}
            max={max || undefined}
            step={step}
            class="
                w-full pb-2 pt-8 text-gray-900 text-base
                bg-transparent
                border-0
                focus:outline-none focus:ring-0
                transition-all duration-200
                placeholder:text-gray-400 placeholder:text-sm
                ${icon ? 'pl-7' : 'pl-0'}
                ${type === 'password' ? 'pr-10' : ''}
                ${disabled ? 'opacity-60 cursor-not-allowed' : ''}
            "
        />
        
        <!-- Soulignement animé -->
        <div class="absolute bottom-0 left-0 w-full h-0.5 bg-gray-200 transition-all duration-300"></div>
        <div class="absolute bottom-0 left-0 w-0 h-0.5 bg-blue-500 transition-all duration-300"
             class:w-full={isFocused && !error}
             class:bg-red-500={error && isFocused}>
        </div>
        
        <!-- Bouton afficher/masquer mot de passe -->
        {#if type === "password"}
            <button
                type="button"
                on:click={togglePassword}
                class="absolute right-0 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors duration-200 focus:outline-none"
                tabindex="-1"
            >
                {#if showPassword}
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                {:else}
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                    </svg>
                {/if}
            </button>
        {/if}
        
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
        <label
            for={name}
            on:click={() => inputRef?.focus()}
            class="
                absolute left-0 cursor-text
                transition-all duration-200 ease-out
                font-medium select-none
                ${icon ? 'left-7' : 'left-0'}
                ${isFloating 
                    ? 'text-xs text-blue-600 -translate-y-5' 
                    : 'text-gray-500 text-base translate-y-2'
                }
                ${error && isFloating ? 'text-red-600' : ''}
                ${error && !isFloating ? 'text-red-500' : ''}
            "
        >
            {label}
            {#if required}
                <span class="text-red-500 ml-0.5">*</span>
            {/if}
        </label>
    </div>
    
    {#if error}
        <div class="mt-1.5 animate-in slide-in-from-top-1 fade-in duration-200">
            <p class="text-xs text-red-600 flex items-center gap-1.5">
                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {error}
            </p>
        </div>
    {/if}
</div>

<style>
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(-4px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: fadeIn 0.2s ease-out;
    }
    
    .slide-in-from-top-1 {
        animation: slideDown 0.2s ease-out;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>