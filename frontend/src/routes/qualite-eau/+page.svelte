<!-- src/routes/qualite-eau/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { waterQualityApi } from "$lib/api/water_quality";
    import { enclosApi } from "$lib/api/enclos";
    import WaterQualityChart from "$lib/components/charts/WaterQualityChart.svelte";
    import Button from "$lib/components/ui/Button.svelte";
    import Modal from "$lib/components/ui/Modal.svelte";
    import WaterQualityForm from "$lib/components/forms/WaterQualityForm.svelte";
    import Select from "$lib/components/ui/Select.svelte";

    let measurements: any[] = [];
    let enclosList: any[] = [];
    let selectedEnclosId: number = 0;
    let loading = true;
    let showModal = false;
    let showAlertes = false;
    let alertes: any[] = [];

    onMount(async () => {
        await loadEnclos();
    });

    async function loadEnclos() {
        try {
            const response = await enclosApi.getEnclos({ limit: 100 });
            enclosList = response.items.filter((e: any) => e.type === "bassin");
            if (enclosList.length > 0) {
                selectedEnclosId = enclosList[0].id;
                await loadData();
            }
        } catch (error) {
            console.error("Failed to load enclos:", error);
        } finally {
            loading = false;
        }
    }

    async function loadData() {
        if (!selectedEnclosId) return;
        loading = true;
        try {
            const [measurementsRes, alertesRes] = await Promise.all([
                waterQualityApi.getMeasurements(selectedEnclosId, {
                    limit: 30,
                }),
                waterQualityApi.getAlerts({
                    enclos_id: selectedEnclosId,
                    limit: 10,
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

    async function handleEnclosChange() {
        await loadData();
    }

    function handleAdd() {
        showModal = true;
    }

    async function handleSubmit(formData: any) {
        try {
            await waterQualityApi.createMeasurement(formData);
            showModal = false;
            await loadData();
        } catch (error) {
            console.error("Failed to add measurement:", error);
        }
    }

    function toggleAlertes() {
        showAlertes = !showAlertes;
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
    </div>

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
                    Alertes ({alertes.filter((a) => !a.traitee).length})
                </Button>
            </div>
        </div>
    </div>

    {#if loading}
        <div class="flex justify-center items-center h-64">
            <div
                class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"
            ></div>
        </div>
    {:else if measurements.length > 0}
        <WaterQualityChart
            data={measurements}
            title="Évolution des paramètres"
            height={450}
        />

        <!-- Dernières mesures -->
        <div class="bg-white rounded-xl border border-gray-200 shadow-sm">
            <div class="p-4 border-b border-gray-200">
                <h3 class="font-semibold text-gray-900">Dernières mesures</h3>
            </div>
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
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-100">
                        {#each measurements.slice(0, 10) as m}
                            <tr class="hover:bg-gray-50">
                                <td class="px-4 py-3"
                                    >{new Date(m.timestamp).toLocaleString(
                                        "fr-FR",
                                    )}</td
                                >
                                <td class="px-4 py-3">
                                    <span
                                        class={m.ph &&
                                        (m.ph < 6.5 || m.ph > 8.5)
                                            ? "text-red-600 font-medium"
                                            : ""}
                                    >
                                        {m.ph || "-"}
                                    </span>
                                </td>
                                <td class="px-4 py-3">
                                    <span
                                        class={m.temperature &&
                                        (m.temperature < 10 ||
                                            m.temperature > 25)
                                            ? "text-red-600 font-medium"
                                            : ""}
                                    >
                                        {m.temperature
                                            ? `${m.temperature}°C`
                                            : "-"}
                                    </span>
                                </td>
                                <td class="px-4 py-3">
                                    <span
                                        class={m.oxygene_dissous &&
                                        m.oxygene_dissous < 5
                                            ? "text-red-600 font-medium"
                                            : ""}
                                    >
                                        {m.oxygene_dissous
                                            ? `${m.oxygene_dissous} mg/L`
                                            : "-"}
                                    </span>
                                </td>
                                <td class="px-4 py-3">
                                    <span
                                        class={m.ammoniac && m.ammoniac > 0.5
                                            ? "text-red-600 font-medium"
                                            : ""}
                                    >
                                        {m.ammoniac
                                            ? `${m.ammoniac} mg/L`
                                            : "-"}
                                    </span>
                                </td>
                                <td class="px-4 py-3">
                                    <span
                                        class={m.nitrites && m.nitrites > 0.5
                                            ? "text-red-600 font-medium"
                                            : ""}
                                    >
                                        {m.nitrites
                                            ? `${m.nitrites} mg/L`
                                            : "-"}
                                    </span>
                                </td>
                                <td class="px-4 py-3">
                                    <span
                                        class={m.nitrates && m.nitrates > 50
                                            ? "text-red-600 font-medium"
                                            : ""}
                                    >
                                        {m.nitrates
                                            ? `${m.nitrates} mg/L`
                                            : "-"}
                                    </span>
                                </td>
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        </div>
    {:else}
        <div
            class="bg-white rounded-xl border border-gray-200 p-8 text-center text-gray-500"
        >
            Aucune mesure de qualité d'eau pour ce bassin
        </div>
    {/if}

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

    <Modal
        open={showAlertes}
        title="Alertes qualité d'eau"
        on:close={() => (showAlertes = false)}
        size="lg"
    >
        <div class="space-y-3">
            {#if alertes.length === 0}
                <p class="text-center text-gray-500 py-4">Aucune alerte</p>
            {:else}
                {#each alertes as alerte}
                    <div
                        class="p-3 rounded-lg bg-yellow-50 border border-yellow-200"
                    >
                        <div class="flex justify-between items-start">
                            <div>
                                <p class="text-sm font-medium text-yellow-800">
                                    {alerte.parametre}
                                </p>
                                <p class="text-sm text-yellow-700 mt-1">
                                    {alerte.message}
                                </p>
                                <p class="text-xs text-yellow-600 mt-2">
                                    Valeur: {alerte.valeur} | Seuil: {alerte.seuil}
                                </p>
                            </div>
                            {#if !alerte.traitee}
                                <button
                                    on:click={() =>
                                        waterQualityApi.resolveAlert(alerte.id)}
                                    class="text-xs text-yellow-700 hover:text-yellow-900"
                                >
                                    Traiter
                                </button>
                            {/if}
                        </div>
                    </div>
                {/each}
            {/if}
        </div>
    </Modal>
</div>
