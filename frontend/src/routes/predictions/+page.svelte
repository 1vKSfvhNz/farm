<!-- src/routes/predictions/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { predictionsApi } from "$lib/api/predictions";
    import { bovinsApi } from "$lib/api/bovins";
    import { enclosApi } from "$lib/api/enclos";
    import ConfidenceIndicator from "$lib/components/dashboard/ConfidenceIndicator.svelte";
    import Select from "$lib/components/ui/Select.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";

    let especes = [
        "bovin",
        "ovin",
        "caprin",
        "avicole",
        "piscicole",
        "apiculture",
        "entomoculture",
    ];
    let especeOptions = especes.map((e) => ({
        value: e,
        label: e.charAt(0).toUpperCase() + e.slice(1),
    }));

    let selectedEspece = "bovin";
    let selectedAnimalId = 0;
    let selectedEnclosId = 0;
    let horizonJours = 30;

    let animaux: any[] = [];
    let enclosList: any[] = [];
    let loadingAnimaux = false;
    let loadingPrediction = false;
    let prediction: any = null;

    onMount(async () => {
        await loadAnimaux();
        await loadEnclos();
    });

    async function loadAnimaux() {
        loadingAnimaux = true;
        try {
            const response = await bovinsApi.getBovins({ limit: 1000 });
            animaux = response.items;
        } catch (error) {
            console.error("Failed to load animaux:", error);
        } finally {
            loadingAnimaux = false;
        }
    }

    async function loadEnclos() {
        try {
            const response = await enclosApi.getEnclos({ limit: 100 });
            enclosList = response.items;
        } catch (error) {
            console.error("Failed to load enclos:", error);
        }
    }

    async function handlePredictGrowth() {
        if (!selectedAnimalId) return;
        loadingPrediction = true;
        try {
            prediction = await predictionsApi.predictGrowth(
                selectedAnimalId,
                horizonJours,
            );
        } catch (error) {
            console.error("Failed to predict growth:", error);
        } finally {
            loadingPrediction = false;
        }
    }

    async function handlePredictProduction() {
        loadingPrediction = true;
        try {
            prediction = await predictionsApi.predictProduction(
                selectedEspece,
                undefined,
                selectedEnclosId || undefined,
                horizonJours,
            );
        } catch (error) {
            console.error("Failed to predict production:", error);
        } finally {
            loadingPrediction = false;
        }
    }

    async function handlePredictCashflow() {
        loadingPrediction = true;
        try {
            prediction = await predictionsApi.predictCashflow(horizonJours);
        } catch (error) {
            console.error("Failed to predict cashflow:", error);
        } finally {
            loadingPrediction = false;
        }
    }
</script>

<div class="space-y-6">
    <h1 class="text-2xl font-bold text-gray-900">Prédictions</h1>

    <!-- Type de prédiction -->
    <div class="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">
            Choisissez une prédiction
        </h2>
        <div class="flex flex-wrap gap-3">
            <button
                on:click={handlePredictGrowth}
                class="px-4 py-2 rounded-lg bg-primary-600 text-white hover:bg-primary-700 transition-colors"
            >
                Croissance animale
            </button>
            <button
                on:click={handlePredictProduction}
                class="px-4 py-2 rounded-lg bg-green-600 text-white hover:bg-green-700 transition-colors"
            >
                Production
            </button>
            <button
                on:click={handlePredictCashflow}
                class="px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors"
            >
                Trésorerie
            </button>
        </div>
    </div>

    <!-- Paramètres -->
    <div class="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">Paramètres</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Select
                label="Espèce"
                bind:value={selectedEspece}
                options={especeOptions}
                on:change={() => loadAnimaux()}
            />

            {#if loadingAnimaux}
                <div class="flex items-center justify-center pt-6">
                    <Spinner size="sm" />
                </div>
            {:else}
                <Select
                    label="Animal"
                    bind:value={selectedAnimalId}
                    options={[
                        { value: 0, label: "Sélectionner..." },
                        ...animaux.map((a) => ({
                            value: a.id,
                            label: a.identification,
                        })),
                    ]}
                />
            {/if}

            <Select
                label="Enclos"
                bind:value={selectedEnclosId}
                options={[
                    { value: 0, label: "Tous" },
                    ...enclosList.map((e) => ({ value: e.id, label: e.name })),
                ]}
            />

            <Select
                label="Horizon (jours)"
                bind:value={horizonJours}
                options={[
                    { value: 30, label: "30 jours" },
                    { value: 60, label: "60 jours" },
                    { value: 90, label: "90 jours" },
                    { value: 180, label: "180 jours" },
                    { value: 365, label: "1 an" },
                ]}
            />
        </div>
    </div>

    <!-- Résultats -->
    {#if loadingPrediction}
        <div class="flex justify-center items-center py-12">
            <Spinner size="lg" />
            <p class="ml-3 text-gray-600">
                Calcul de la prédiction en cours...
            </p>
        </div>
    {:else if prediction}
        <div class="space-y-4">
            <!-- Confiance -->
            <div
                class="bg-white rounded-xl border border-gray-200 shadow-sm p-4"
            >
                <div class="flex items-center justify-between">
                    <span class="text-sm text-gray-600"
                        >Niveau de confiance</span
                    >
                    <ConfidenceIndicator
                        confidence={prediction.confidence || 75}
                        size="md"
                    />
                </div>
            </div>

            <!-- Prédiction de croissance -->
            {#if prediction.poids_prevu_jours}
                <div
                    class="bg-white rounded-xl border border-gray-200 shadow-sm p-6"
                >
                    <h3 class="text-lg font-semibold text-gray-900 mb-4">
                        Prédiction de croissance
                    </h3>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <div class="p-4 bg-gray-50 rounded-lg text-center">
                            <p class="text-sm text-gray-500">Poids actuel</p>
                            <p class="text-2xl font-bold text-gray-900">
                                {prediction.poids_actuel_kg} kg
                            </p>
                        </div>
                        <div class="p-4 bg-gray-50 rounded-lg text-center">
                            <p class="text-sm text-gray-500">Âge actuel</p>
                            <p class="text-2xl font-bold text-gray-900">
                                {prediction.age_actuel_jours} jours
                            </p>
                        </div>
                        {#if prediction.date_atteinte_poids_vente}
                            <div class="p-4 bg-green-50 rounded-lg text-center">
                                <p class="text-sm text-green-600">
                                    Date atteinte poids vente
                                </p>
                                <p class="text-xl font-bold text-green-700">
                                    {new Date(
                                        prediction.date_atteinte_poids_vente,
                                    ).toLocaleDateString("fr-FR")}
                                </p>
                            </div>
                        {/if}
                    </div>

                    <h4 class="font-medium text-gray-800 mb-3">
                        Prédiction par période
                    </h4>
                    <div class="overflow-x-auto">
                        <table class="w-full text-sm">
                            <thead class="bg-gray-50">
                                <tr>
                                    <th class="px-4 py-2 text-left"
                                        >Période (jours)</th
                                    >
                                    <th class="px-4 py-2 text-left"
                                        >Poids min (kg)</th
                                    >
                                    <th class="px-4 py-2 text-left"
                                        >Poids max (kg)</th
                                    >
                                    <th class="px-4 py-2 text-left"
                                        >Poids moyen (kg)</th
                                    >
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-gray-100">
                                {#each prediction.poids_prevu_jours as p}
                                    <tr>
                                        <td class="px-4 py-2">{p.jour} jours</td
                                        >
                                        <td class="px-4 py-2">{p.poids_min}</td>
                                        <td class="px-4 py-2">{p.poids_max}</td>
                                        <td class="px-4 py-2 font-medium"
                                            >{p.poids_moyen}</td
                                        >
                                    </tr>
                                {/each}
                            </tbody>
                        </table>
                    </div>

                    {#if prediction.recommandations && prediction.recommandations.length > 0}
                        <div class="mt-4 p-4 bg-blue-50 rounded-lg">
                            <p class="text-sm font-medium text-blue-800">
                                Recommandations
                            </p>
                            <ul
                                class="list-disc list-inside text-sm text-blue-700 mt-2"
                            >
                                {#each prediction.recommandations as rec}
                                    <li>{rec}</li>
                                {/each}
                            </ul>
                        </div>
                    {/if}

                    {#if prediction.retard_croissance_detecte}
                        <div class="mt-4 p-4 bg-yellow-50 rounded-lg">
                            <p class="text-sm font-medium text-yellow-800">
                                ⚠️ Retard de croissance détecté
                            </p>
                        </div>
                    {/if}
                </div>
            {/if}

            <!-- Prédiction de production -->
            {#if prediction.production_prevue_30j}
                <div
                    class="bg-white rounded-xl border border-gray-200 shadow-sm p-6"
                >
                    <h3 class="text-lg font-semibold text-gray-900 mb-4">
                        Prédiction de production
                    </h3>
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div class="p-4 bg-gray-50 rounded-lg text-center">
                            <p class="text-sm text-gray-500">
                                Production actuelle
                            </p>
                            <p class="text-xl font-bold text-gray-900">
                                {prediction.production_quotidienne_actuelle}
                                {prediction.type_production === "lait"
                                    ? "L/j"
                                    : "unités/j"}
                            </p>
                        </div>
                        <div class="p-4 bg-gray-50 rounded-lg text-center">
                            <p class="text-sm text-gray-500">Prévue à 15j</p>
                            <p class="text-xl font-bold text-gray-900">
                                {prediction.production_prevue_15j}
                            </p>
                        </div>
                        <div class="p-4 bg-gray-50 rounded-lg text-center">
                            <p class="text-sm text-gray-500">Prévue à 30j</p>
                            <p class="text-xl font-bold text-gray-900">
                                {prediction.production_prevue_30j}
                            </p>
                        </div>
                        <div class="p-4 bg-gray-50 rounded-lg text-center">
                            <p class="text-sm text-gray-500">Prévue à 90j</p>
                            <p class="text-xl font-bold text-gray-900">
                                {prediction.production_prevue_90j}
                            </p>
                        </div>
                    </div>

                    {#if prediction.recommandations && prediction.recommandations.length > 0}
                        <div class="mt-4 p-4 bg-blue-50 rounded-lg">
                            <p class="text-sm font-medium text-blue-800">
                                Recommandations
                            </p>
                            <ul
                                class="list-disc list-inside text-sm text-blue-700 mt-2"
                            >
                                {#each prediction.recommandations as rec}
                                    <li>{rec}</li>
                                {/each}
                            </ul>
                        </div>
                    {/if}
                </div>
            {/if}

            <!-- Prédiction de trésorerie -->
            {#if prediction.tresorerie_prevue_30j !== undefined}
                <div
                    class="bg-white rounded-xl border border-gray-200 shadow-sm p-6"
                >
                    <h3 class="text-lg font-semibold text-gray-900 mb-4">
                        Prédiction de trésorerie
                    </h3>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <div class="p-4 bg-gray-50 rounded-lg text-center">
                            <p class="text-sm text-gray-500">
                                Trésorerie actuelle
                            </p>
                            <p class="text-2xl font-bold text-gray-900">
                                {prediction.tresorerie_actuelle.toLocaleString(
                                    "fr-FR",
                                )} €
                            </p>
                        </div>
                        <div class="p-4 bg-gray-50 rounded-lg text-center">
                            <p class="text-sm text-gray-500">
                                Entrées prévues (30j)
                            </p>
                            <p class="text-xl font-bold text-green-600">
                                +{prediction.entrees_prevues_30j.toLocaleString(
                                    "fr-FR",
                                )} €
                            </p>
                        </div>
                        <div class="p-4 bg-gray-50 rounded-lg text-center">
                            <p class="text-sm text-gray-500">
                                Sorties prévues (30j)
                            </p>
                            <p class="text-xl font-bold text-red-600">
                                -{prediction.sorties_prevues_30j.toLocaleString(
                                    "fr-FR",
                                )} €
                            </p>
                        </div>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div class="p-4 bg-blue-50 rounded-lg text-center">
                            <p class="text-sm text-blue-600">
                                Trésorerie prévue à 30j
                            </p>
                            <p class="text-xl font-bold text-blue-700">
                                {prediction.tresorerie_prevue_30j.toLocaleString(
                                    "fr-FR",
                                )} €
                            </p>
                        </div>
                        <div class="p-4 bg-blue-50 rounded-lg text-center">
                            <p class="text-sm text-blue-600">
                                Trésorerie prévue à 60j
                            </p>
                            <p class="text-xl font-bold text-blue-700">
                                {prediction.tresorerie_prevue_60j.toLocaleString(
                                    "fr-FR",
                                )} €
                            </p>
                        </div>
                        <div class="p-4 bg-blue-50 rounded-lg text-center">
                            <p class="text-sm text-blue-600">
                                Trésorerie prévue à 90j
                            </p>
                            <p class="text-xl font-bold text-blue-700">
                                {prediction.tresorerie_prevue_90j.toLocaleString(
                                    "fr-FR",
                                )} €
                            </p>
                        </div>
                    </div>

                    {#if prediction.seuil_alerte_atteint}
                        <div class="mt-4 p-4 bg-red-50 rounded-lg">
                            <p class="text-sm font-medium text-red-800">
                                ⚠️ Alerte : Seuil de trésorerie critique atteint
                                dans les 90 jours
                            </p>
                        </div>
                    {/if}

                    {#if prediction.recommandations && prediction.recommandations.length > 0}
                        <div class="mt-4 p-4 bg-blue-50 rounded-lg">
                            <p class="text-sm font-medium text-blue-800">
                                Recommandations
                            </p>
                            <ul
                                class="list-disc list-inside text-sm text-blue-700 mt-2"
                            >
                                {#each prediction.recommandations as rec}
                                    <li>{rec}</li>
                                {/each}
                            </ul>
                        </div>
                    {/if}
                </div>
            {/if}
        </div>
    {:else}
        <div
            class="bg-white rounded-xl border border-gray-200 shadow-sm p-12 text-center"
        >
            <svg
                class="w-16 h-16 text-gray-300 mx-auto mb-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
            >
                <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"
                />
            </svg>
            <p class="text-gray-500">
                Sélectionnez un type de prédiction et les paramètres pour voir
                les résultats
            </p>
        </div>
    {/if}
</div>
