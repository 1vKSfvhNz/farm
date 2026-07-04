<!-- src/routes/auth/forgot-password/+page.svelte -->
<script lang="ts">
    import { goto } from "$app/navigation";
    import Button from "$components/ui/Button.svelte";
    import Input from "$components/ui/Input.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";
    
    let email = "";
    let loading = false;
    let error = "";
    let success = false;
    let successMessage = "";
    
    async function handleSubmit() {
        if (!email) {
            error = "Veuillez saisir votre adresse email";
            return;
        }
        
        // Validation simple de l'email
        const emailRegex = /^[^\s@]+@([^\s@.,]+\.)+[^\s@.,]{2,}$/;
        if (!emailRegex.test(email)) {
            error = "Veuillez saisir une adresse email valide";
            return;
        }
        
        loading = true;
        error = "";
        
        try {
            const response = await fetch("/api/v1/auth/forgot-password", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ email })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                success = true;
                successMessage = data.message || "Un email de réinitialisation vous a été envoyé si un compte existe avec cette adresse.";
                // Rediriger après 3 secondes
                setTimeout(() => {
                    goto("/auth/login");
                }, 3000);
            } else {
                error = data.detail || "Une erreur est survenue. Veuillez réessayer.";
            }
        } catch (err) {
            console.error("Forgot password error:", err);
            error = "Erreur de connexion. Veuillez vérifier votre connexion internet.";
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
    </div>

    <div class="w-full max-w-md relative z-10 animate-in slide-in-from-bottom-4 duration-500">
        <div class="bg-white/95 backdrop-blur-sm rounded-3xl shadow-2xl overflow-hidden border border-white/20">
            <!-- En-tête -->
            <div class="relative bg-gradient-to-br from-slate-800 to-slate-900 px-8 py-10 text-center overflow-hidden">
                <div class="absolute top-0 right-0 w-40 h-40 bg-emerald-400/10 rounded-full -mr-20 -mt-20 blur-2xl"></div>
                <div class="absolute bottom-0 left-0 w-32 h-32 bg-amber-400/10 rounded-full -ml-16 -mb-16 blur-2xl"></div>
                
                <div class="relative z-10">
                    <button 
                        on:click={() => goto("/auth/login")}
                        class="absolute left-0 top-0 text-slate-400 hover:text-white transition-colors duration-200"
                    >
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                        </svg>
                    </button>
                    
                    <div class="relative inline-block mb-4">
                        <div class="absolute inset-0 bg-gradient-to-r from-emerald-400 to-teal-400 rounded-full blur-xl opacity-60 animate-pulse"></div>
                        <div class="relative w-16 h-16 bg-gradient-to-br from-slate-700 to-slate-800 rounded-full flex items-center justify-center shadow-xl border border-slate-600">
                            <svg class="w-8 h-8 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7.5a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0zM3.75 20.25c0-3.75 3-6.75 6.75-6.75s6.75 3 6.75 6.75" />
                            </svg>
                        </div>
                    </div>
                    
                    <h1 class="text-2xl font-bold text-white">Mot de passe oublié ?</h1>
                    <p class="text-slate-400 text-sm mt-2">
                        Ne vous inquiétez pas, ça arrive à tout le monde
                    </p>
                </div>
            </div>

            <!-- Formulaire -->
            <div class="p-8">
                {#if success}
                    <div class="mb-6 p-4 bg-emerald-50 border-l-4 border-emerald-500 rounded-xl animate-in slide-in-from-top-2 fade-in duration-300">
                        <div class="flex items-start gap-3">
                            <svg class="w-5 h-5 text-emerald-500 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <div>
                                <p class="font-medium text-emerald-700 text-sm">Email envoyé !</p>
                                <p class="text-emerald-600 text-sm mt-1">{successMessage}</p>
                                <p class="text-emerald-600 text-xs mt-2">
                                    Redirection vers la page de connexion dans quelques secondes...
                                </p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="text-center">
                        <div class="inline-block w-8 h-8 border-4 border-emerald-200 border-t-emerald-600 rounded-full animate-spin"></div>
                    </div>
                {:else}
                    {#if error}
                        <div class="mb-6 p-4 bg-red-50 border-l-4 border-red-500 rounded-xl animate-in slide-in-from-top-2 fade-in duration-300">
                            <div class="flex items-start gap-3">
                                <svg class="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <div>
                                    <p class="font-medium text-red-700 text-sm">Erreur</p>
                                    <p class="text-red-600 text-sm">{error}</p>
                                </div>
                            </div>
                        </div>
                    {/if}

                    <div class="text-center mb-6">
                        <p class="text-slate-600 text-sm">
                            Entrez votre adresse email et nous vous enverrons un lien pour réinitialiser votre mot de passe.
                        </p>
                    </div>

                    <form on:submit|preventDefault={handleSubmit}>
                        <Input
                            label="Adresse email"
                            bind:value={email}
                            inputType="email"
                            required={true}
                            placeholder="ex: john@example.com"
                            icon="email"
                        />

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
                                <span class="ml-2">Envoi en cours...</span>
                            {:else}
                                <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                </svg>
                                Envoyer le lien de réinitialisation
                            {/if}
                        </Button>
                    </form>

                    <div class="text-center mt-6">
                        <button 
                            on:click={() => goto("/auth/login")}
                            class="text-sm text-slate-500 hover:text-slate-700 transition-colors duration-200 inline-flex items-center gap-1 group"
                        >
                            <svg class="w-4 h-4 transform group-hover:-translate-x-0.5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                            </svg>
                            Retour à la connexion
                        </button>
                    </div>
                {/if}
            </div>
        </div>
    </div>
</div>

<style>
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
    
    .animate-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    .slide-in-from-bottom-4 {
        animation: slideUp 0.4s ease-out;
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
    
    .delay-1000 {
        animation-delay: 1s;
    }
</style>