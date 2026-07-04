<!-- src/routes/piscicoles/[id]/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { piscicolesApi } from "$lib/api/piscicoles";
    import { peseeApi } from "$lib/api/pesee";
    import { permissionsStore } from "$lib/stores/permissions";
    import Card from "$lib/components/ui/Card.svelte";
    import Button from "$lib/components/ui/Button.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";
    import LineChart from "$lib/components/charts/LineChart.svelte";
    import Modal from "$lib/components/ui/Modal.svelte";
    import PeseeForm from "$lib/components/forms/PeseeForm.svelte";

    let id: number;
    let piscicole: any = null;
    let pesees: any[] = [];
    let loading = true;
    let showPeseeModal = false;

    const canEdit = permissionsStore.canEditEspece("piscicoles");

    onMount(async () => {
        const path = window.location.pathname;
        const match = path.match(/\/piscicoles\/(\d+)/);
        if (match) {
            id = parseInt(match[1]);
            await loadData();
        }
    });

    async function loadData() {
        loading = true;
        try {
            const [piscicoleData, peseesData] = await Promise.all([
                piscicolesApi.getPiscicole(id),
                peseeApi.getPesees({ animal_id: id, limit: 100 }),
            ]);
            piscicole = piscicoleData;
            pesees = peseesData.items;
        } catch (error) {
            console.error("Failed to load data:", error);
        } finally {
            loading = false;
        }
    }

    async function handleAddPesee(formData: any) {
        try {
            await peseeApi.createPesee({ ...formData, animal_id: id });
            showPeseeModal = false;
            await loadData();
        } catch (error) {
            console.error("Failed to add pesee:", error);
        }
    }

    const chartLabels = pesees.map((p) =>
        new Date(p.date_pesee).toLocaleDateString("fr-FR"),
    );
    const chartData = pesees.map((p) => p.poids);
</script>

<div class="space-y-6">
    {#if loading}
        <div class="flex justify-center items-center h-64">
            <Spinner size="lg" />
        </div>
    {:else if !piscicole}
        <div class="text-center py-12">
            <p class="text-gray-500">Piscicole non trouvé</p>
            <Button
                on:click={() => (window.location.href = "/piscicoles")}
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
                    {piscicole.identification}
                </h1>
                <p class="text-sm text-gray-500 mt-1">
                    Espèce: {piscicole.race}
                </p>
            </div>
            <div class="flex gap-2">
                {#if canEdit}
                    <Button
                        on:click={() =>
                            (window.location.href = `/piscicoles/${id}/edit`)}
                        variant="outline"
                    >
                        Modifier
                    </Button>
                {/if}
                <Button
                    on:click={() => (window.location.href = "/piscicoles")}
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
                        <span class="font-medium">
                            {#if piscicole.statut === "vivant"}
                                <span class="text-green-600">Vivant</span>
                            {:else if piscicole.statut === "vendu"}
                                <span class="text-blue-600">Vendu</span>
                            {:else}
                                <span class="text-red-600">Décédé</span>
                            {/if}
                        </span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Âge:</span>
                        <span class="font-medium"
                            >{piscicole.age_jours || 0} jours</span
                        >
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Date arrivée:</span>
                        <span class="font-medium"
                            >{new Date(
                                piscicole.date_arrivee,
                            ).toLocaleDateString("fr-FR")}</span
                        >
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Taille moyenne:</span>
                        <span class="font-medium"
                            >{piscicole.taille_moyenne
                                ? `${piscicole.taille_moyenne} cm`
                                : "-"}</span
                        >
                    </div>
                    {#if piscicole.prix_achat}
                        <div class="flex justify-between">
                            <span class="text-gray-500">Prix d'achat:</span>
                            <span class="font-medium"
                                >{piscicole.prix_achat.toLocaleString("fr-FR")} €</span
                            >
                        </div>
                    {/if}
                </div>
            </Card>

            <Card title="Production">
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Production viande:</span>
                        <span class="font-medium"
                            >{piscicole.production_viande
                                ? "✅ Oui"
                                : "❌ Non"}</span
                        >
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Reproduction:</span>
                        <span class="font-medium"
                            >{piscicole.production_reproduction
                                ? "✅ Oui"
                                : "❌ Non"}</span
                        >
                    </div>
                </div>
            </Card>

            <Card title="Dernière pesée">
                <div class="space-y-2 text-sm">
                    {#if pesees.length > 0}
                        {@const dernierePesee = pesees[pesees.length - 1]}
                        <div class="flex justify-between">
                            <span class="text-gray-500">Poids:</span>
                            <span class="font-medium text-lg"
                                >{dernierePesee.poids} g</span
                            >
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">Date:</span>
                            <span class="font-medium"
                                >{new Date(
                                    dernierePesee.date_pesee,
                                ).toLocaleDateString("fr-FR")}</span
                            >
                        </div>
                    {:else}
                        <p class="text-gray-500 text-center">
                            Aucune pesée enregistrée
                        </p>
                    {/if}
                    {#if canEdit}
                        <div class="mt-4">
                            <Button
                                on:click={() => (showPeseeModal = true)}
                                variant="primary"
                                size="sm"
                                fullWidth
                            >
                                Ajouter une pesée
                            </Button>
                        </div>
                    {/if}
                </div>
            </Card>
        </div>

        {#if pesees.length > 1}
            <Card title="Courbe de croissance">
                <LineChart
                    labels={chartLabels}
                    datasets={[
                        {
                            label: "Poids (g)",
                            data: chartData,
                            color: "#3b82f6",
                        },
                    ]}
                    yAxisLabel="Poids (g)"
                    height={350}
                />
            </Card>
        {/if}

        {#if pesees.length > 0}
            <Card title="Historique des pesées">
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-4 py-2 text-left">Date</th>
                                <th class="px-4 py-2 text-left">Poids (g)</th>
                                <th class="px-4 py-2 text-left">Méthode</th>
                                <th class="px-4 py-2 text-left">Notes</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-100">
                            {#each pesees.slice().reverse() as pesee}
                                <tr class="hover:bg-gray-50">
                                    <td class="px-4 py-2"
                                        >{new Date(
                                            pesee.date_pesee,
                                        ).toLocaleDateString("fr-FR")}</td
                                    >
                                    <td class="px-4 py-2 font-medium"
                                        >{pesee.poids} g</td
                                    >
                                    <td class="px-4 py-2"
                                        >{pesee.methode || "-"}</td
                                    >
                                    <td class="px-4 py-2"
                                        >{pesee.notes || "-"}</td
                                    >
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                </div>
            </Card>
        {/if}

        {#if piscicole.notes}
            <Card title="Notes">
                <p class="text-gray-600 whitespace-pre-wrap">
                    {piscicole.notes}
                </p>
            </Card>
        {/if}
    {/if}

    <Modal
        open={showPeseeModal}
        title="Ajouter une pesée"
        on:close={() => (showPeseeModal = false)}
        size="md"
    >
        <PeseeForm
            formData={{
                animal_id: id,
                date_pesee: new Date().toISOString().split("T")[0],
                poids: 0,
            }}
            loading={false}
            on:submit={handleAddPesee}
            on:cancel={() => (showPeseeModal = false)}
        />
    </Modal>
</div>
