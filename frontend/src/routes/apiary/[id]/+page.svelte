<!-- src/routes/apiary/[id]/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { apiaryApi } from "$lib/api/apiary"
    import { permissionsStore } from "$lib/stores/permissions";
    import Card from "$lib/components/ui/Card.svelte";
    import Button from "$lib/components/ui/Button.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";
    import BarChart from "$lib/components/charts/BarChart.svelte";
    import Modal from "$lib/components/ui/Modal.svelte";
    import RecolteMielForm from "$lib/components/forms/RecolteMielForm.svelte";

    let id: number;
    let ruche: any = null;
    let recoltes: any[] = [];
    let loading = true;
    let showRecolteModal = false;

    const canEdit = permissionsStore.canEditEspece("apiculture");

    onMount(async () => {
        const path = window.location.pathname;
        const match = path.match(/\/apiculture\/(\d+)/);
        if (match) {
            id = parseInt(match[1]);
            await loadData();
        }
    });

    async function loadData() {
        loading = true;
        try {
            const [rucheData, recoltesData] = await Promise.all([
                apiaryApi.getRuche(id),
                apiaryApi.getRecoltes(id, { limit: 100 }),
            ]);
            ruche = rucheData;
            recoltes = recoltesData.items;
        } catch (error) {
            console.error("Failed to load data:", error);
        } finally {
            loading = false;
        }
    }

    async function handleAddRecolte(formData: any) {
        try {
            await apiaryApi.addRecolte(id, formData);
            showRecolteModal = false;
            await loadData();
        } catch (error) {
            console.error("Failed to add recolte:", error);
        }
    }

    const chartLabels = recoltes.map((r) =>
        new Date(r.date_recolte).toLocaleDateString("fr-FR"),
    );
    const chartData = recoltes.map((r) => r.poids_kg);

    function getStatutLabel(statut: string): string {
        const labels: Record<string, string> = {
            active: "Active",
            orpheline: "Orpheline",
            en_essaimage: "En essaimage",
            morte: "Morte",
        };
        return labels[statut] || statut;
    }

    function getStatutColor(statut: string): string {
        const colors: Record<string, string> = {
            active: "text-green-600",
            orpheline: "text-red-600",
            en_essaimage: "text-yellow-600",
            morte: "text-gray-600",
        };
        return colors[statut] || "text-gray-600";
    }
</script>

<div class="space-y-6">
    {#if loading}
        <div class="flex justify-center items-center h-64">
            <Spinner size="lg" />
        </div>
    {:else if !ruche}
        <div class="text-center py-12">
            <p class="text-gray-500">Ruche non trouvée</p>
            <Button
                on:click={() => (window.location.href = "/apiculture")}
                variant="primary"
                className="mt-4"
            >
                Retour à la liste
            </Button>
        </div>
    {:else}
        <div class="flex justify-between items-center">
            <div>
                <h1 class="text-2xl font-bold text-gray-900">
                    {ruche.identification}
                </h1>
                <p class="text-sm text-gray-500 mt-1">
                    Emplacement: {ruche.emplacement || "Non spécifié"}
                </p>
            </div>
            <div class="flex gap-2">
                {#if canEdit}
                    <Button
                        on:click={() =>
                            (window.location.href = `/apiculture/${id}/edit`)}
                        variant="outline"
                    >
                        Modifier
                    </Button>
                {/if}
                <Button
                    on:click={() => (window.location.href = "/apiculture")}
                    variant="outline"
                >
                    Retour
                </Button>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card title="Informations générales">
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Statut:</span>
                        <span class="font-medium {getStatutColor(ruche.statut)}"
                            >{getStatutLabel(ruche.statut)}</span
                        >
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Race:</span>
                        <span class="font-medium">{ruche.race || "-"}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Date installation:</span>
                        <span class="font-medium"
                            >{new Date(
                                ruche.date_installation,
                            ).toLocaleDateString("fr-FR")}</span
                        >
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Nombre de cadres:</span>
                        <span class="font-medium"
                            >{ruche.nombre_cadres || "-"}</span
                        >
                    </div>
                </div>
            </Card>

            <Card title="Production">
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Production totale:</span>
                        <span class="font-medium"
                            >{recoltes.reduce((sum, r) => sum + r.poids_kg, 0)} kg</span
                        >
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Nombre de récoltes:</span>
                        <span class="font-medium">{recoltes.length}</span>
                    </div>
                    {#if recoltes.length > 0}
                        <div class="flex justify-between">
                            <span class="text-gray-500"
                                >Moyenne par récolte:</span
                            >
                            <span class="font-medium"
                                >{(
                                    recoltes.reduce(
                                        (sum, r) => sum + r.poids_kg,
                                        0,
                                    ) / recoltes.length
                                ).toFixed(1)} kg</span
                            >
                        </div>
                    {/if}
                </div>
            </Card>

            <Card title="Actions">
                <div class="space-y-3">
                    {#if canEdit}
                        <Button
                            on:click={() => (showRecolteModal = true)}
                            variant="primary"
                            size="sm"
                            fullWidth
                        >
                            Ajouter une récolte
                        </Button>
                    {/if}
                    {#if ruche.statut === "orpheline"}
                        <div
                            class="p-3 bg-yellow-50 rounded-lg text-sm text-yellow-800"
                        >
                            ⚠️ Cette ruche est orpheline. Une intervention est
                            nécessaire.
                        </div>
                    {/if}
                </div>
            </Card>
        </div>

        {#if recoltes.length > 0}
            <Card title="Production de miel">
                <BarChart
                    labels={chartLabels}
                    datasets={[
                        {
                            label: "Production (kg)",
                            data: chartData,
                            color: "#f59e0b",
                        },
                    ]}
                    yAxisLabel="Quantité (kg)"
                    height={350}
                />
            </Card>

            <Card title="Historique des récoltes">
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-4 py-2 text-left">Date</th>
                                <th class="px-4 py-2 text-left">Poids (kg)</th>
                                <th class="px-4 py-2 text-left">Qualité</th>
                                <th class="px-4 py-2 text-left">Taux d'eau</th>
                                <th class="px-4 py-2 text-left">Notes</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-100">
                            {#each recoltes.slice().reverse() as recolte}
                                <tr class="hover:bg-gray-50">
                                    <td class="px-4 py-2"
                                        >{new Date(
                                            recolte.date_recolte,
                                        ).toLocaleDateString("fr-FR")}</td
                                    >
                                    <td class="px-4 py-2 font-medium"
                                        >{recolte.poids_kg} kg</td
                                    >
                                    <td class="px-4 py-2"
                                        >{recolte.qualite || "-"}</td
                                    >
                                    <td class="px-4 py-2"
                                        >{recolte.taux_eau
                                            ? `${recolte.taux_eau}%`
                                            : "-"}</td
                                    >
                                    <td class="px-4 py-2"
                                        >{recolte.notes || "-"}</td
                                    >
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                </div>
            </Card>
        {/if}

        {#if ruche.notes}
            <Card title="Notes">
                <p class="text-gray-600 whitespace-pre-wrap">{ruche.notes}</p>
            </Card>
        {/if}
    {/if}

    <Modal
        open={showRecolteModal}
        title="Ajouter une récolte"
        on:close={() => (showRecolteModal = false)}
        size="md"
    >
        <RecolteMielForm
            formData={{
                ruche_id: id,
                date_recolte: new Date().toISOString().split("T")[0],
                poids_kg: 0,
            }}
            rucheOptions={[{ value: id, label: ruche?.identification || "" }]}
            loading={false}
            on:submit={handleAddRecolte}
            on:cancel={() => (showRecolteModal = false)}
        />
    </Modal>
</div>
