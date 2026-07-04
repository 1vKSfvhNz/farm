<!-- frontend/src/lib/components/pages/LoginPage.svelte -->
<script lang="ts">
    import { authStore } from "../../stores/auth";
    import { notificationsStore } from "../../stores/notifications";

    let phone = "";
    let code = "";
    let loading = false;

    async function handleSubmit() {
        if (!phone || !code) {
            notificationsStore.error("Veuillez remplir tous les champs");
            return;
        }
        loading = true;
        try {
            await authStore.login(phone, code);
            notificationsStore.success("Connexion réussie");
        } catch (err) {
            notificationsStore.error("Numéro ou code incorrect");
        } finally {
            loading = false;
        }
    }
</script>

<div
    class="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 to-secondary-50"
>
    <div class="max-w-md w-full mx-4">
        <div class="bg-white rounded-2xl shadow-xl p-8">
            <div class="text-center mb-8">
                <div
                    class="w-20 h-20 bg-primary-500 rounded-2xl flex items-center justify-center mx-auto shadow-lg"
                >
                    <span class="text-4xl">🌾</span>
                </div>
                <h1 class="mt-4 text-2xl font-bold text-gray-800">
                    Farm Manager
                </h1>
                <p class="text-gray-500">Burkina Faso</p>
            </div>

            <form on:submit|preventDefault={handleSubmit}>
                <div class="mb-4">
                    <input
                        type="tel"
                        placeholder="Numéro de téléphone (+226XXXXXXXX)"
                        bind:value={phone}
                        class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                    />
                </div>
                <div class="mb-6">
                    <input
                        type="password"
                        placeholder="Code secret"
                        bind:value={code}
                        class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                    />
                </div>
                <button
                    type="submit"
                    disabled={loading}
                    class="w-full bg-primary-600 text-white py-3 rounded-lg font-semibold hover:bg-primary-700 transition-colors disabled:opacity-50"
                >
                    {loading ? "Connexion..." : "Se connecter"}
                </button>
            </form>

            <p class="text-center text-xs text-gray-400 mt-6">
                Démo: +22670000000 / admin123
            </p>
        </div>
    </div>
</div>
