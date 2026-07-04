<!-- src/routes/experimental/+page.svelte -->
<!-- svelte-ignore a11y-label-has-associated-control -->
<script lang="ts">
    import { onMount } from "svelte";
    import { experimentalStore } from "$lib/stores/experimental";
    import Button from "$lib/components/ui/Button.svelte";
    import Card from "$lib/components/ui/Card.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";
    import Modal from "$lib/components/ui/Modal.svelte";
    import ExperimentalHypothesisForm from "$lib/components/forms/ExperimentalHypothesisForm.svelte";

    let modeStatus: any = null;
    let hypotheses: any[] = [];
    let collectionStats: any = null;
    let loading = true;
    let showModal = false;
    let selectedEspece = "";

    onMount(async () => {
        await loadData();
    });

    async function loadData() {
        loading = true;
        try {
            await experimentalStore.loadModeStatus();
            await experimentalStore.loadHypotheses();
            await experimentalStore.loadCollectionStats();

            const state = await new Promise<any>((resolve) => {
                const unsubscribe = experimentalStore.subscribe((s) => {
                    unsubscribe();
                    resolve(s);
                });
            });
            modeStatus = state.modeStatus;
            hypotheses = state.hypotheses;
            collectionStats = state.collectionStats;
        } catch (error) {
            console.error("Failed to load experimental data:", error);
        } finally {
            loading = false;
        }
    }

    async function handleGenerateReference() {
        if (!selectedEspece) return;
        try {
            await experimentalStore.generateReference(selectedEspece, true);
            await loadData();
        } catch (error) {
            console.error("Failed to generate reference:", error);
        }
    }

    async function handleValidateHypothesis(id: number) {
        try {
            await experimentalStore.validateHypothesis(id);
            await loadData();
        } catch (error) {
            console.error("Failed to validate hypothesis:", error);
        }
    }

    async function handleSubmitHypothesis(formData: any) {
        try {
            await experimentalStore.createHypothesis(formData);
            showModal = false;
            await loadData();
        } catch (error) {
            console.error("Failed to create hypothesis:", error);
        }
    }

    function getModeBadge(mode: string): string {
        switch (mode) {
            case "complet":
                return "bg-green-100 text-green-800";
            case "hybride":
                return "bg-blue-100 text-blue-800";
            case "experimental":
                return "bg-purple-100 text-purple-800";
            default:
                return "bg-gray-100 text-gray-800";
        }
    }
</script>

<div class="space-y-6">
    <div class="flex justify-between items-center">
        <div>
            <h1 class="text-2xl font-bold text-gray-900">Mode expérimental</h1>
            <p class="text-sm text-gray-500 mt-1">
                Gestion des hypothèses et prédictions avancées
            </p>
        </div>
        <Button on:click={() => (showModal = true)} variant="primary">
            <svg
                class="w-4 h-4 mr-1"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
            >
                <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M12 4v16m8-8H4"
                />
            </svg>
            Nouvelle hypothèse
        </Button>
    </div>

    {#if loading}
        <div class="flex justify-center items-center h-64">
            <Spinner size="lg" />
        </div>
    {:else}
        <!-- Statut du mode expérimental -->
        {#if modeStatus}
            <Card title="Statut du mode expérimental">
                <div class="space-y-4">
                    <div class="flex flex-wrap items-center gap-4">
                        <span
                            class="px-3 py-1 rounded-full text-sm font-medium {getModeBadge(
                                modeStatus.mode,
                            )}"
                        >
                            Mode {modeStatus.mode === "complet"
                                ? "Complet"
                                : modeStatus.mode === "hybride"
                                  ? "Hybride"
                                  : "Expérimental"}
                        </span>
                        <span class="text-sm text-gray-600">
                            {modeStatus.jours_collecte} jours de données collectées
                        </span>
                        <span class="text-sm text-gray-600">
                            Confiance moyenne: {modeStatus.confiance_moyenne}%
                        </span>
                    </div>

                    <div>
                        <h4 class="font-medium text-gray-800 mb-2">
                            Données par espèce
                        </h4>
                        <div class="flex flex-wrap gap-2">
                            {#each Object.entries(modeStatus.nombre_donnees_par_espece || {}) as [espece, count]}
                                <span
                                    class="px-2 py-1 bg-gray-100 rounded-full text-xs"
                                >
                                    {espece}: {count} données
                                </span>
                            {/each}
                        </div>
                    </div>

                    {#if modeStatus.seuils_atteints && modeStatus.seuils_atteints.length > 0}
                        <div class="p-3 bg-green-50 rounded-lg">
                            <p class="text-sm font-medium text-green-800">
                                Seuils atteints
                            </p>
                            <ul
                                class="list-disc list-inside text-sm text-green-700 mt-1"
                            >
                                {#each modeStatus.seuils_atteints as seuil}
                                    <li>{seuil}</li>
                                {/each}
                            </ul>
                        </div>
                    {/if}

                    {#if modeStatus.recommandations && modeStatus.recommandations.length > 0}
                        <div class="p-3 bg-blue-50 rounded-lg">
                            <p class="text-sm font-medium text-blue-800">
                                Recommandations
                            </p>
                            <ul
                                class="list-disc list-inside text-sm text-blue-700 mt-1"
                            >
                                {#each modeStatus.recommandations as rec}
                                    <li>{rec}</li>
                                {/each}
                            </ul>
                        </div>
                    {/if}
                </div>
            </Card>
        {/if}

        <!-- Statistiques de collecte -->
        {#if collectionStats}
            <Card title="Statistiques de collecte">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <p class="text-3xl font-bold text-primary-600">
                            {collectionStats.totalDonnees}
                        </p>
                        <p class="text-sm text-gray-500">
                            Données totales collectées
                        </p>
                    </div>
                    <div>
                        <h4 class="font-medium text-gray-800 mb-2">
                            Par espèce
                        </h4>
                        <div class="space-y-1">
                            {#each Object.entries(collectionStats.parEspece || {}) as [espece, count]}
                                <div class="flex justify-between text-sm">
                                    <span>{espece}</span>
                                    <span class="font-medium">{count}</span>
                                </div>
                            {/each}
                        </div>
                    </div>
                </div>
            </Card>
        {/if}

        <!-- Génération de référence -->
        <Card title="Générer une référence">
            <div class="flex flex-wrap gap-4 items-end">
                <div class="flex-1 min-w-[200px]">
                    <label class="block text-sm font-medium text-gray-700 mb-1"
                        >Espèce</label
                    >
                    <select
                        bind:value={selectedEspece}
                        class="w-full rounded-lg border border-gray-300 px-3 py-2"
                    >
                        <option value="">Sélectionner une espèce</option>
                        <option value="bovin">Bovin</option>
                        <option value="ovin">Ovin</option>
                        <option value="caprin">Caprin</option>
                        <option value="avicole">Avicole</option>
                    </select>
                </div>
                <Button
                    on:click={handleGenerateReference}
                    variant="primary"
                    disabled={!selectedEspece}
                >
                    Générer les références
                </Button>
            </div>
        </Card>

        <!-- Hypothèses -->
        <Card title="Hypothèses de référence">
            {#if hypotheses.length === 0}
                <p class="text-center text-gray-500 py-8">
                    Aucune hypothèse enregistrée
                </p>
            {:else}
                <div class="space-y-3">
                    {#each hypotheses as h}
                        <div class="p-4 border border-gray-200 rounded-lg">
                            <div class="flex justify-between items-start">
                                <div>
                                    <div class="flex items-center gap-2 mb-2">
                                        <span
                                            class="px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100"
                                        >
                                            {h.espece}
                                        </span>
                                        <span
                                            class="text-sm font-medium text-gray-900"
                                            >{h.parametre}</span
                                        >
                                        {#if h.validee}
                                            <span
                                                class="px-2 py-0.5 rounded-full text-xs bg-green-100 text-green-800"
                                                >Validée</span
                                            >
                                        {:else}
                                            <span
                                                class="px-2 py-0.5 rounded-full text-xs bg-yellow-100 text-yellow-800"
                                                >En attente</span
                                            >
                                        {/if}
                                    </div>
                                    <p class="text-2xl font-bold text-gray-900">
                                        {h.valeur_estimee}
                                    </p>
                                    <p class="text-sm text-gray-500">
                                        {h.unite || "unité"}
                                    </p>
                                    {#if h.race}
                                        <p class="text-xs text-gray-400 mt-1">
                                            Race: {h.race}
                                        </p>
                                    {/if}
                                </div>
                                {#if !h.validee}
                                    <Button
                                        on:click={() =>
                                            handleValidateHypothesis(h.id)}
                                        size="sm"
                                        variant="primary"
                                    >
                                        Valider
                                    </Button>
                                {/if}
                            </div>
                        </div>
                    {/each}
                </div>
            {/if}
        </Card>
    {/if}

    <Modal
        open={showModal}
        title="Nouvelle hypothèse"
        on:close={() => (showModal = false)}
        size="lg"
    >
        <ExperimentalHypothesisForm
            formData={{
                espece: "",
                parametre: "",
                valeur_estimee: 0,
            }}
            loading={false}
            on:submit={handleSubmitHypothesis}
            on:cancel={() => (showModal = false)}
        />
    </Modal>
</div>
