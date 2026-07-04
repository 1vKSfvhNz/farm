<!-- src/routes/+layout.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { authStore } from "$stores/auth";
    import { permissionsStore } from "$stores/permissions";
    import { goto } from "$app/navigation";

    import "../app.css";
    
    let isChecking = true;

    onMount(() => {
        // Initialiser les stores
        authStore.init();
        permissionsStore.init();

        // Vérifier l'authentification et rediriger si nécessaire
        const unsubscribe = authStore.subscribe((state) => {
            if (!state.isLoading) {
                isChecking = false;
                if (
                    !state.isAuthenticated &&
                    window.location.pathname !== "/auth/login"
                ) {
                    goto("/auth/login");
                }
            }
        });

        return () => {
            unsubscribe();
            permissionsStore.cleanup();
        };
    });
</script>

{#if isChecking}
    <div
        class="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100"
    >
        <div class="text-center">
            <div
                class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"
            ></div>
            <p class="mt-4 text-gray-600 font-medium">
                Vérification de l'authentification...
            </p>
        </div>
    </div>
{:else}
    <slot />
{/if}