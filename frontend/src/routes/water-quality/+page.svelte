<!-- src/routes/qualite-eau/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { waterQualityApi } from "$lib/api/water_quality";
    import { enclosApi } from "$lib/api/enclos";
    import { permissionsStore } from "$lib/stores/permissions";
    import WaterQualityChart from "$lib/components/charts/WaterQualityChart.svelte";
    import Button from "$lib/components/ui/Button.svelte";
    import Modal from "$lib/components/ui/Modal.svelte";
    import WaterQualityForm from "$lib/components/forms/WaterQualityForm.svelte";
    import Select from "$lib/components/ui/Select.svelte";
    import Card from "$lib/components/ui/Card.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";

    let measurements: any[] = [];
    let enclosList: any[] = [];
    let selectedEnclosId: number = 0;
    let loading = true;
    let showModal = false;
    let showAlertes = false;
    let alertes: any[] = [];
    let stats = {
        ph_moyen: 0,
        temperature_moyenne: 0,
        oxygene_moyen: 0,
        alertes_count: 0,
    };

    const canEdit = permissionsStore.hasPermission("can_edit_water_quality");

    onMount(async () => {
        await loadEnclos();
    });

    async function loadEnclos() {
        try {
            const response = await enclosApi.getEnclos({ limit: 100 });
            // Filtrer uniquement les bassins
            enclosList = response.items.filter((e: any) => e.type === "bassin");
            if (enclosList.length > 0) {
                selectedEnclosId = enclosList[0].id;
                await loadData();
                await loadStats();
            } else {
                loading = false;
            }
        } catch (error) {
            console.error("Failed to load enclos:", error);
            loading = false;
        }
    }

    async function loadData() {
        if (!selectedEnclosId) return;
        loading = true;
        try {
            const [measurementsRes, alertesRes] = await Promise.all([
                waterQualityApi.getMeasurements(selectedEnclosId, {
                    limit: 50,
                }),
                waterQualityApi.getAlerts({
                    enclos_id: selectedEnclosId,
                    limit: 20,
                }),
            ]);
            measurements = measurementsRes.items;
            alertes = alertesRes.items;
        } catch (error) {
            console.error("Failed to load water quality data:", error);
        } finally {
            loading = false;
        }
    }

    async function loadStats() {
        if (!selectedEnclosId) return;
        try {
            const allMeasurements = await waterQualityApi.getMeasurements(
                selectedEnclosId,
                { limit: 1000 },
            );
            const items = allMeasurements.items;

            if (items.length > 0) {
                const phSum = items.reduce((sum, m) => sum + (m.ph || 0), 0);
                const tempSum = items.reduce(
                    (sum, m) => sum + (m.temperature || 0),
                    0,
                );
                const oxySum = items.reduce(
                    (sum, m) => sum + (m.oxygene_dissous || 0),
                    0,
                );

                stats = {
                    ph_moyen: parseFloat((phSum / items.length).toFixed(1)),
                    temperature_moyenne: parseFloat(
                        (tempSum / items.length).toFixed(1),
                    ),
                    oxygene_moyen: parseFloat(
                        (oxySum / items.length).toFixed(1),
                    ),
                    alertes_count: alertes.filter((a) => !a.traitee).length,
                };
            }
        } catch (error) {
            console.error("Failed to load stats:", error);
        }
    }

    async function handleEnclosChange() {
        await loadData();
        await loadStats();
    }

    function handleAdd() {
        showModal = true;
    }

    async function handleSubmit(formData: any) {
        try {
            await waterQualityApi.createMeasurement(formData);
            showModal = false;
            await loadData();
            await loadStats();
        } catch (error) {
            console.error("Failed to add measurement:", error);
        }
    }

    async function handleResolveAlert(alertId: number) {
        try {
            await waterQualityApi.resolveAlert(alertId);
            await loadData();
            await loadStats();
        } catch (error) {
            console.error("Failed to resolve alert:", error);
        }
    }

    function toggleAlertes() {
        showAlertes = !showAlertes;
    }

    function getParamStatus(param: string, value: number): string {
        const thresholds: Record<string, { min?: number; max?: number }> = {
            ph: { min: 6.5, max: 8.5 },
            temperature: { min: 10, max: 25 },
            oxygene_dissous: { min: 5 },
            ammoniac: { max: 0.5 },
            nitrites: { max: 0.5 },
            nitrates: { max: 50 },
        };

        const threshold = thresholds[param];
        if (!threshold) return "text-gray-600";

        if (threshold.min !== undefined && value < threshold.min)
            return "text-red-600";
        if (threshold.max !== undefined && value > threshold.max)
            return "text-red-600";
        return "text-green-600";
    }
</script>

<div class="space-y-6">
    <div class="flex justify-between items-center">
        <div>
            <h1 class="text-2xl font-bold text-gray-900">Qualité de l'eau</h1>
            <p class="text-sm text-gray-500 mt-1">
                Surveillance des paramètres des bassins
            </p>
        </div>
        {#if canEdit}
            <Button on:click={handleAdd} variant="primary">
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
                Nouvelle mesure
            </Button>
        {/if}
    </div>

    <!-- Filtre par bassin -->
    <div class="bg-white rounded-xl border border-gray-200 p-4">
        <div class="flex flex-wrap items-center justify-between gap-4">
            <Select
                label="Bassin"
                bind:value={selectedEnclosId}
                options={enclosList.map((e) => ({
                    value: e.id,
                    label: e.name,
                }))}
                on:change={handleEnclosChange}
                className="w-64"
            />
            <div class="flex gap-2">
                <Button on:click={toggleAlertes} variant="outline" size="sm">
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
                            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                        />
                    </svg>
                    Alertes ({stats.alertes_count})
                </Button>
            </div>
        </div>
    </div>

    {#if loading}
        <div class="flex justify-center items-center h-64">
            <Spinner size="lg" />
        </div>
    {:else if enclosList.length === 0}
        <div
            class="bg-white rounded-xl border border-gray-200 p-12 text-center"
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
                    d="M4.5 12.75a6 6 0 0111.383-3.057M12 6.75h3.75a.75.75 0 01.75.75v3.75m-4.5-4.5L21 15.75M4.5 12.75L12 21m-7.5-8.25L9 9"
                />
            </svg>
            <p class="text-gray-500">Aucun bassin configuré</p>
            <p class="text-sm text-gray-400 mt-1">
                Ajoutez un bassin dans la gestion des enclos
            </p>
            <Button
                on:click={() => (window.location.href = "/enclos")}
                variant="primary"
                className="mt-4"
            >
                Gérer les enclos
            </Button>
        </div>
    {:else if measurements.length === 0}
        <div
            class="bg-white rounded-xl border border-gray-200 p-12 text-center"
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
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
            </svg>
            <p class="text-gray-500">Aucune mesure de qualité d'eau</p>
            <p class="text-sm text-gray-400 mt-1">
                Ajoutez votre première mesure pour commencer le suivi
            </p>
            {#if canEdit}
                <Button on:click={handleAdd} variant="primary" className="mt-4">
                    Nouvelle mesure
                </Button>
            {/if}
        </div>
    {:else}
        <!-- Statistiques rapides -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div
                class="bg-white rounded-xl border border-gray-200 p-4 text-center"
            >
                <p class="text-sm text-gray-500">pH moyen</p>
                <p
                    class="text-2xl font-bold {getParamStatus(
                        'ph',
                        stats.ph_moyen,
                    )}"
                >
                    {stats.ph_moyen}
                </p>
                <p class="text-xs text-gray-400">Normale: 6.5 - 8.5</p>
            </div>
            <div
                class="bg-white rounded-xl border border-gray-200 p-4 text-center"
            >
                <p class="text-sm text-gray-500">Température moyenne</p>
                <p
                    class="text-2xl font-bold {getParamStatus(
                        'temperature',
                        stats.temperature_moyenne,
                    )}"
                >
                    {stats.temperature_moyenne}°C
                </p>
                <p class="text-xs text-gray-400">Normale: 10 - 25°C</p>
            </div>
            <div
                class="bg-white rounded-xl border border-gray-200 p-4 text-center"
            >
                <p class="text-sm text-gray-500">Oxygène moyen</p>
                <p
                    class="text-2xl font-bold {getParamStatus(
                        'oxygene_dissous',
                        stats.oxygene_moyen,
                    )}"
                >
                    {stats.oxygene_moyen} mg/L
                </p>
                <p class="text-xs text-gray-400">Minimum: 5 mg/L</p>
            </div>
            <div
                class="bg-white rounded-xl border border-gray-200 p-4 text-center"
            >
                <p class="text-sm text-gray-500">Alertes actives</p>
                <p class="text-2xl font-bold text-red-600">
                    {stats.alertes_count}
                </p>
                <p class="text-xs text-gray-400">Nécessite attention</p>
            </div>
        </div>

        <!-- Graphique d'évolution -->
        <WaterQualityChart
            data={measurements}
            title="Évolution des paramètres"
            height={450}
        />

        <!-- Dernières mesures -->
        <Card title="Dernières mesures">
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-4 py-3 text-left">Date</th>
                            <th class="px-4 py-3 text-left">pH</th>
                            <th class="px-4 py-3 text-left">Température</th>
                            <th class="px-4 py-3 text-left">Oxygène</th>
                            <th class="px-4 py-3 text-left">Ammoniac</th>
                            <th class="px-4 py-3 text-left">Nitrites</th>
                            <th class="px-4 py-3 text-left">Nitrates</th>
                            <th class="px-4 py-3 text-left">Source</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-100">
                        {#each measurements.slice(0, 10) as m}
                            <table class="hover:bg-gray-50">
                                <td class="px-4 py-3"
                                    >{new Date(m.timestamp).toLocaleString(
                                        "fr-FR",
                                    )}</td
                                >
                                <td class="px-4 py-3">
                                    <span class={getParamStatus("ph", m.ph)}>
                                        {m.ph || "-"}
                                    </span>
                                </td>
                                <td class="px-4 py-3">
                                    <span
                                        class={getParamStatus(
                                            "temperature",
                                            m.temperature,
                                        )}
                                    >
                                        {m.temperature
                                            ? `${m.temperature}°C`
                                            : "-"}
                                    </span>
                                </td>
                                <td class="px-4 py-3">
                                    <span
                                        class={getParamStatus(
                                            "oxygene_dissous",
                                            m.oxygene_dissous,
                                        )}
                                    >
                                        {m.oxygene_dissous
                                            ? `${m.oxygene_dissous} mg/L`
                                            : "-"}
                                    </span>
                                </td>
                                <td class="px-4 py-3">
                                    <span
                                        class={getParamStatus(
                                            "ammoniac",
                                            m.ammoniac,
                                        )}
                                    >
                                        {m.ammoniac
                                            ? `${m.ammoniac} mg/L`
                                            : "-"}
                                    </span>
                                </td>
                                <td class="px-4 py-3">
                                    <span
                                        class={getParamStatus(
                                            "nitrites",
                                            m.nitrites,
                                        )}
                                    >
                                        {m.nitrites
                                            ? `${m.nitrites} mg/L`
                                            : "-"}
                                    </span>
                                </td>
                                <td class="px-4 py-3">
                                    <span
                                        class={getParamStatus(
                                            "nitrates",
                                            m.nitrates,
                                        )}
                                    >
                                        {m.nitrates
                                            ? `${m.nitrates} mg/L`
                                            : "-"}
                                    </span>
                                </td>
                                <td class="px-4 py-3">{m.source || "-"}</td>
                            </table>
                        {/each}
                    </tbody>
                </table>
            </div>
            {#if measurements.length > 10}
                <div class="mt-4 text-center text-sm text-gray-500">
                    + {measurements.length - 10} mesures supplémentaires
                </div>
            {/if}
        </Card>
    {/if}

    <!-- Modal ajout mesure -->
    <Modal
        open={showModal}
        title="Ajouter une mesure"
        on:close={() => (showModal = false)}
        size="lg"
    >
        <WaterQualityForm
            formData={{
                enclos_id: selectedEnclosId,
                timestamp: new Date().toISOString().slice(0, 16),
            }}
            enclosOptions={enclosList.map((e) => ({
                value: e.id,
                label: e.name,
            }))}
            loading={false}
            on:submit={handleSubmit}
            on:cancel={() => (showModal = false)}
        />
    </Modal>

    <!-- Modal alertes -->
    <Modal
        open={showAlertes}
        title="Alertes qualité d'eau"
        on:close={() => (showAlertes = false)}
        size="lg"
    >
        <div class="space-y-3">
            {#if alertes.length === 0}
                <p class="text-center text-gray-500 py-8">
                    Aucune alerte active
                </p>
            {:else}
                {#each alertes as alerte}
                    {#if !alerte.traitee}
                        <div
                            class="p-4 rounded-lg bg-yellow-50 border border-yellow-200"
                        >
                            <div class="flex justify-between items-start">
                                <div class="flex-1">
                                    <div class="flex items-center gap-2">
                                        <span
                                            class="px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-200 text-yellow-800"
                                        >
                                            {alerte.parametre}
                                        </span>
                                        <span class="text-xs text-yellow-600">
                                            {new Date(
                                                alerte.created_at,
                                            ).toLocaleString("fr-FR")}
                                        </span>
                                    </div>
                                    <p class="text-sm text-yellow-800 mt-2">
                                        {alerte.message}
                                    </p>
                                    <p class="text-xs text-yellow-700 mt-1">
                                        Valeur: {alerte.valeur} | Seuil: {alerte.seuil}
                                    </p>
                                </div>
                                <button
                                    on:click={() =>
                                        handleResolveAlert(alerte.id)}
                                    class="text-xs text-yellow-700 hover:text-yellow-900 bg-yellow-100 px-2 py-1 rounded"
                                >
                                    Traiter
                                </button>
                            </div>
                        </div>
                    {/if}
                {/each}

                <!-- Alertes traitées -->
                {#if alertes.some((a) => a.traitee)}
                    <div class="mt-4 pt-4 border-t border-gray-200">
                        <p class="text-sm font-medium text-gray-700 mb-2">
                            Alertes traitées
                        </p>
                        {#each alertes.filter((a) => a.traitee) as alerte}
                            <div
                                class="p-3 rounded-lg bg-gray-50 border border-gray-200 mb-2"
                            >
                                <div class="flex justify-between items-start">
                                    <div>
                                        <span
                                            class="px-2 py-0.5 rounded-full text-xs font-medium bg-gray-200 text-gray-600"
                                        >
                                            {alerte.parametre}
                                        </span>
                                        <p class="text-sm text-gray-600 mt-1">
                                            {alerte.message}
                                        </p>
                                    </div>
                                    <span class="text-xs text-green-600"
                                        >✓ Traitée</span
                                    >
                                </div>
                            </div>
                        {/each}
                    </div>
                {/if}
            {/if}
        </div>
    </Modal>
</div>
