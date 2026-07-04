<!-- src/routes/auth/login/+page.svelte -->
<script lang="ts">
    import { authStore } from "$stores/auth";
    import { goto } from "$app/navigation";
    import Button from "$components/ui/Button.svelte";
    import Spinner from "$components/ui/Spinner.svelte";
    import Input from "$lib/components/ui/Input.svelte";

    let userlogin = "";
    let password = "";
    let loading = false;
    let error = "";

    async function handleSubmit() {
        if (!userlogin || !password) {
            error = "Veuillez remplir tous les champs";
            return;
        }

        loading = true;
        error = "";

        try {
            const success = await authStore.login(userlogin, password);
            if (success) {
                goto("/");
            } else {
                error = "Identifiant ou mot de passe incorrect";
            }
        } catch (err) {
            console.error("Login error:", err);
            error = "Une erreur est survenue. Veuillez réessayer.";
        } finally {
            loading = false;
        }
    }
</script>

<div class="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4 relative overflow-hidden">
    <!-- Animations de fond -->
    <div class="absolute inset-0 overflow-hidden">
        <div class="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-emerald-500/20 to-teal-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div class="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-blue-500/20 to-cyan-500/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-r from-amber-500/10 to-orange-500/10 rounded-full blur-3xl"></div>
    </div>

    <div class="w-full max-w-md relative z-10 animate-in slide-in-from-bottom-4 duration-500">
        <!-- Carte principale -->
        <div class="bg-white/95 backdrop-blur-sm rounded-3xl shadow-2xl overflow-hidden border border-white/20">
            <!-- En-tête avec illustration -->
            <div class="relative bg-gradient-to-br from-slate-800 to-slate-900 px-8 py-12 text-center overflow-hidden">
                <!-- Éléments décoratifs -->
                <div class="absolute top-0 right-0 w-40 h-40 bg-emerald-400/10 rounded-full -mr-20 -mt-20 blur-2xl"></div>
                <div class="absolute bottom-0 left-0 w-32 h-32 bg-amber-400/10 rounded-full -ml-16 -mb-16 blur-2xl"></div>
                
                <!-- Cercles flottants animés -->
                <div class="absolute top-1/4 right-8 w-2 h-2 bg-emerald-400 rounded-full animate-float"></div>
                <div class="absolute bottom-1/4 left-8 w-1.5 h-1.5 bg-amber-400 rounded-full animate-float delay-300"></div>
                <div class="absolute top-1/3 left-1/4 w-1 h-1 bg-blue-400 rounded-full animate-float delay-700"></div>
                
                <div class="relative z-10">
                    <!-- Logo animé -->
                    <div class="relative inline-block mb-6">
                        <div class="absolute inset-0 bg-gradient-to-r from-emerald-400 to-teal-400 rounded-2xl blur-xl opacity-60 animate-pulse"></div>
                        <div class="relative w-20 h-20 bg-gradient-to-br from-slate-700 to-slate-800 rounded-2xl flex items-center justify-center shadow-xl border border-slate-600 group-hover:scale-110 transition-transform duration-300">
                            <span class="text-4xl animate-bounce-in">🐄</span>
                        </div>
                    </div>
                    
                    <h1 class="text-3xl font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent">
                        Farm Manager
                    </h1>
                    <p class="text-slate-400 text-sm mt-2 flex items-center justify-center gap-2">
                        <span class="inline-block w-1 h-1 bg-emerald-400 rounded-full"></span>
                        Gestion d'élevage intelligent
                        <span class="inline-block w-1 h-1 bg-emerald-400 rounded-full"></span>
                    </p>
                </div>
            </div>

            <!-- Formulaire -->
            <div class="p-8">
                <div class="text-center mb-8">
                    <h2 class="text-2xl font-bold text-slate-800">Bienvenue</h2>
                    <p class="text-slate-500 text-sm mt-2">
                        Connectez-vous pour accéder à votre espace
                    </p>
                </div>

                {#if error}
                    <div class="mb-6 p-4 bg-red-50 border-l-4 border-red-500 rounded-xl animate-in slide-in-from-top-2 fade-in duration-300">
                        <div class="flex items-start gap-3">
                            <svg class="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5 animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <div>
                                <p class="font-medium text-red-700 text-sm">Erreur de connexion</p>
                                <p class="text-red-600 text-sm">{error}</p>
                            </div>
                        </div>
                    </div>
                {/if}

                <form on:submit|preventDefault={handleSubmit}>
                    <Input
                        label="Nom d'utilisateur"
                        bind:value={userlogin}
                        inputType="text"
                        required={true}
                        placeholder="ex: john_doe"
                        icon="user"
                    />

                    <Input
                        label="Mot de passe"
                        bind:value={password}
                        inputType="password"
                        required={true}
                        placeholder="Votre mot de passe"
                        icon="lock"
                    />


                    <!-- Lien mot de passe oublié -->
                    <div class="text-center mb-6">
                        <a href="/auth/forgot-password" class="text-sm text-slate-500 hover:text-slate-700 transition-colors duration-200 inline-flex items-center gap-1 group">
                            <span>Mot de passe oublié ?</span>
                        </a>
                    </div>

                    <Button 
                        on:click={handleSubmit} 
                        disabled={loading} 
                        fullWidth 
                        size="lg" 
                        variant="primary" 
                        className="relative overflow-hidden group bg-gradient-to-r from-slate-700 via-slate-800 to-slate-900 hover:from-slate-800 hover:via-slate-900 hover:to-slate-950 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-[1.02]"
                    >
                        <div class="absolute inset-0 bg-gradient-to-r from-emerald-400/0 via-emerald-400/10 to-emerald-400/0 -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                        
                        {#if loading}
                            <Spinner size="sm" color="white" />
                            <span class="ml-2">Connexion en cours...</span>
                        {:else}
                            <svg class="w-4 h-4 mr-2 transition-transform group-hover:translate-x-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1" />
                            </svg>
                            Se connecter
                        {/if}
                    </Button>
                </form>

                <!-- Version -->
                <div class="text-center mt-8 pt-6 border-t border-slate-100">
                    <p class="text-xs text-slate-400">Farm Manager v2.0 — © 2026</p>
                </div>
            </div>
        </div>
    </div>
</div>

<style>
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes bounce-in {
        0% { transform: scale(0); opacity: 0; }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .animate-float {
        animation: float 3s ease-in-out infinite;
    }
    
    .animate-bounce-in {
        animation: bounce-in 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }
    
    .delay-300 {
        animation-delay: 300ms;
    }
    
    .delay-700 {
        animation-delay: 700ms;
    }
    
    .delay-1000 {
        animation-delay: 1s;
    }
    
    .animate-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    .slide-in-from-bottom-4 {
        animation: slideUp 0.4s ease-out;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>