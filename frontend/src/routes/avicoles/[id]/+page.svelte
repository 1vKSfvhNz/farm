<!-- src/routes/avicoles/[id]/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { avicolesApi } from "$lib/api/avicoles";
    import { peseeApi } from "$lib/api/pesee";
    import { permissionsStore } from "$lib/stores/permissions";
    import Card from "$lib/components/ui/Card.svelte";
    import Button from "$lib/components/ui/Button.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";
    import LineChart from "$lib/components/charts/LineChart.svelte";
    import Modal from "$lib/components/ui/Modal.svelte";
    import Input from "$lib/components/ui/Input.svelte";
    import PeseeForm from "$components/forms/PeseeForm.svelte";

    let id: number;
    let avicole: any = null;
    let pesees: any[] = [];
    let loading = true;
    let showPeseeModal = false;
    let showOeufsModal = false;
    let eggCount = 0;
    let eggWeight = 0;

    const canEdit = permissionsStore.canEditEspece("avicoles");

    onMount(async () => {
        const path = window.location.pathname;
        const match = path.match(/\/avicoles\/(\d+)/);
        if (match) {
            id = parseInt(match[1]);
            await loadData();
        }
    });

    async function loadData() {
        loading = true;
        try {
            const [avicoleData, peseesData] = await Promise.all([
                avicolesApi.getAvicole(id),
                peseeApi.getPesees({ animal_id: id, limit: 100 }),
            ]);
            avicole = avicoleData;
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

    async function handleAddOeufs() {
        if (eggCount > 0) {
            try {
                await avicolesApi.addEggProduction(id, eggCount, eggWeight);
                showOeufsModal = false;
                await loadData();
                eggCount = 0;
                eggWeight = 0;
            } catch (error) {
                console.error("Failed to add eggs:", error);
            }
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
    {:else if !avicole}
        <div class="text-center py-12">
            <p class="text-gray-500">Avicole non trouvé</p>
            <Button
                on:click={() => (window.location.href = "/avicoles")}
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
                    {avicole.identification}
                </h1>
                <p class="text-sm text-gray-500 mt-1">
                    Race: {avicole.race} | {avicole.sexe === "male"
                        ? "Coq"
                        : "Poule"}
                </p>
            </div>
            <div class="flex gap-2">
                {#if canEdit}
                    <Button
                        on:click={() =>
                            (window.location.href = `/avicoles/${id}/edit`)}
                        variant="outline"
                    >
                        Modifier
                    </Button>
                {/if}
                <Button
                    on:click={() => (window.location.href = "/avicoles")}
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
                            {#if avicole.statut === "vivant"}
                                <span class="text-green-600">Vivant</span>
                            {:else if avicole.statut === "vendu"}
                                <span class="text-blue-600">Vendu</span>
                            {:else}
                                <span class="text-red-600">Décédé</span>
                            {/if}
                        </span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Âge:</span>
                        <span class="font-medium"
                            >{avicole.age_jours || 0} jours</span
                        >
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Date arrivée:</span>
                        <span class="font-medium"
                            >{new Date(avicole.date_arrivee).toLocaleDateString(
                                "fr-FR",
                            )}</span
                        >
                    </div>
                    {#if avicole.prix_achat}
                        <div class="flex justify-between">
                            <span class="text-gray-500">Prix d'achat:</span>
                            <span class="font-medium"
                                >{avicole.prix_achat.toLocaleString("fr-FR")} €</span
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
                            >{avicole.production_viande
                                ? "✅ Oui"
                                : "❌ Non"}</span
                        >
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Production ponte:</span>
                        <span class="font-medium"
                            >{avicole.production_ponte
                                ? "✅ Oui"
                                : "❌ Non"}</span
                        >
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Reproduction:</span>
                        <span class="font-medium"
                            >{avicole.production_reproduction
                                ? "✅ Oui"
                                : "❌ Non"}</span
                        >
                    </div>
                    {#if avicole.production_ponte}
                        <div
                            class="flex justify-between pt-2 border-t border-gray-200"
                        >
                            <span class="text-gray-500">Œufs pondus:</span>
                            <span class="font-medium text-lg"
                                >{avicole.oeufs_pondus_total || 0}</span
                            >
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">Poids total œufs:</span>
                            <span class="font-medium"
                                >{avicole.poids_oeufs_total || 0} g</span
                            >
                        </div>
                    {/if}
                </div>
                {#if canEdit && avicole.production_ponte && avicole.sexe === "femelle"}
                    <div class="mt-4">
                        <Button
                            on:click={() => (showOeufsModal = true)}
                            variant="success"
                            size="sm"
                            fullWidth
                        >
                            + Ajouter des œufs
                        </Button>
                    </div>
                {/if}
            </Card>

            <Card title="Dernière pesée">
                <div class="space-y-2 text-sm">
                    {#if pesees.length > 0}
                        {@const dernierePesee = pesees[pesees.length - 1]}
                        <div class="flex justify-between">
                            <span class="text-gray-500">Poids:</span>
                            <span class="font-medium text-lg"
                                >{dernierePesee.poids} kg</span
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
                            label: "Poids (kg)",
                            data: chartData,
                            color: "#3b82f6",
                        },
                    ]}
                    yAxisLabel="Poids (kg)"
                    height={350}
                />
            </Card>
        {/if}

        {#if avicole.notes}
            <Card title="Notes">
                <p class="text-gray-600 whitespace-pre-wrap">{avicole.notes}</p>
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

    <Modal
        open={showOeufsModal}
        title="Ajouter une production d'œufs"
        on:close={() => (showOeufsModal = false)}
        size="md"
    >
        <div class="space-y-4">
            <p class="text-sm text-gray-600">
                Animal: <span class="font-medium"
                    >{avicole?.identification}</span
                >
            </p>
            <Input
                label="Nombre d'œufs"
                bind:value={eggCount}
                inputType="number"
                required
                placeholder="0"
            />
            <Input
                label="Poids total (g)"
                bind:value={eggWeight}
                inputType="number"
                placeholder="0"
            />
            <div class="flex justify-end gap-3 pt-4">
                <Button
                    on:click={() => (showOeufsModal = false)}
                    variant="outline"
                >
                    Annuler
                </Button>
                <Button on:click={handleAddOeufs} variant="primary">
                    Ajouter
                </Button>
            </div>
        </div>
    </Modal>
</div>
