<!-- src/routes/entomoculture/[id]/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { entomocultureApi } from "$lib/api/entomoculture";
    import { permissionsStore } from "$lib/stores/permissions";
    import Card from "$lib/components/ui/Card.svelte";
    import Button from "$lib/components/ui/Button.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";
    import GaugeChart from "$lib/components/charts/GaugeChart.svelte";

    let id: number;
    let lot: any = null;
    let cycles: any[] = [];
    let loading = true;

    const canEdit = permissionsStore.canEditEspece("entomoculture");

    onMount(async () => {
        const path = window.location.pathname;
        const match = path.match(/\/entomoculture\/(\d+)/);
        if (match) {
            id = parseInt(match[1]);
            await loadData();
        }
    });

    async function loadData() {
        loading = true;
        try {
            const [lotData, cyclesData] = await Promise.all([
                entomocultureApi.getLot(id),
                entomocultureApi.getCycles(id),
            ]);
            lot = lotData;
            cycles = cyclesData;
        } catch (error) {
            console.error("Failed to load data:", error);
        } finally {
            loading = false;
        }
    }

    function getStadeLabel(stade: string): string {
        const labels: Record<string, string> = {
            oeuf: "Œuf",
            larve: "Larve",
            pupe: "Pupe",
            adulte: "Adulte",
        };
        return labels[stade] || stade;
    }

    function getMortaliteColor(mortalite: number): string {
        if (mortalite <= 10) return "text-green-600";
        if (mortalite <= 20) return "text-yellow-600";
        return "text-red-600";
    }
</script>

<div class="space-y-6">
    {#if loading}
        <div class="flex justify-center items-center h-64">
            <Spinner size="lg" />
        </div>
    {:else if !lot}
        <div class="text-center py-12">
            <p class="text-gray-500">Lot non trouvé</p>
            <Button
                on:click={() => (window.location.href = "/entomoculture")}
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
                    {lot.identification}
                </h1>
                <p class="text-sm text-gray-500 mt-1">
                    Espèce: {lot.espece} | Stade: {getStadeLabel(
                        lot.stade_actuel,
                    )}
                </p>
            </div>
            <div class="flex gap-2">
                {#if canEdit}
                    <Button
                        on:click={() =>
                            (window.location.href = `/entomoculture/${id}/edit`)}
                        variant="outline"
                    >
                        Modifier
                    </Button>
                {/if}
                <Button
                    on:click={() => (window.location.href = "/entomoculture")}
                    variant="outline"
                >
                    Retour
                </Button>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card title="Informations générales">
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Type production:</span>
                        <span class="font-medium">{lot.type_production}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Date arrivée:</span>
                        <span class="font-medium"
                            >{new Date(lot.date_arrivee).toLocaleDateString(
                                "fr-FR",
                            )}</span
                        >
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Provenance:</span>
                        <span class="font-medium">{lot.provenance || "-"}</span>
                    </div>
                    {#if lot.prix_achat}
                        <div class="flex justify-between">
                            <span class="text-gray-500">Prix d'achat:</span>
                            <span class="font-medium"
                                >{lot.prix_achat.toLocaleString("fr-FR")} €</span
                            >
                        </div>
                    {/if}
                </div>
            </Card>

            <Card title="Quantité">
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Quantité estimée:</span>
                        <span class="font-medium"
                            >{lot.quantite_estimative?.toLocaleString(
                                "fr-FR",
                            ) || "-"}</span
                        >
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Poids initial:</span>
                        <span class="font-medium"
                            >{lot.poids_initial
                                ? `${lot.poids_initial} g`
                                : "-"}</span
                        >
                    </div>
                </div>
            </Card>

            <Card title="Mortalité">
                <div class="space-y-2 text-sm">
                    {#if lot.taux_mortalite}
                        <div class="flex justify-center">
                            <GaugeChart
                                value={lot.taux_mortalite}
                                title=""
                                unit="%"
                                size="sm"
                            />
                        </div>
                        <div class="flex justify-between mt-2">
                            <span class="text-gray-500">Taux:</span>
                            <span
                                class="font-medium {getMortaliteColor(
                                    lot.taux_mortalite,
                                )}">{lot.taux_mortalite}%</span
                            >
                        </div>
                    {:else}
                        <p class="text-gray-500 text-center">Aucune donnée</p>
                    {/if}
                </div>
            </Card>

            <Card title="Cycles">
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Nombre de cycles:</span>
                        <span class="font-medium">{cycles.length}</span>
                    </div>
                    {#if cycles.length > 0}
                        <div class="flex justify-between">
                            <span class="text-gray-500">Production totale:</span
                            >
                            <span class="font-medium"
                                >{cycles.reduce(
                                    (sum, c) =>
                                        sum + (c.production_grammes || 0),
                                    0,
                                )} g</span
                            >
                        </div>
                    {/if}
                </div>
            </Card>
        </div>

        {#if cycles.length > 0}
            <Card title="Historique des cycles">
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-4 py-2 text-left">Début</th>
                                <th class="px-4 py-2 text-left">Fin</th>
                                <th class="px-4 py-2 text-left">Stade début</th>
                                <th class="px-4 py-2 text-left">Stade fin</th>
                                <th class="px-4 py-2 text-left"
                                    >Production (g)</th
                                >
                                <th class="px-4 py-2 text-left">Mortalité</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-100">
                            {#each cycles.slice().reverse() as cycle}
                                <tr class="hover:bg-gray-50">
                                    <td class="px-4 py-2"
                                        >{new Date(
                                            cycle.date_debut,
                                        ).toLocaleDateString("fr-FR")}</td
                                    >
                                    <td class="px-4 py-2"
                                        >{cycle.date_fin
                                            ? new Date(
                                                  cycle.date_fin,
                                              ).toLocaleDateString("fr-FR")
                                            : "En cours"}</td
                                    >
                                    <td class="px-4 py-2"
                                        >{getStadeLabel(cycle.stade_debut)}</td
                                    >
                                    <td class="px-4 py-2"
                                        >{cycle.stade_fin
                                            ? getStadeLabel(cycle.stade_fin)
                                            : "-"}</td
                                    >
                                    <td class="px-4 py-2"
                                        >{cycle.production_grammes || "-"}</td
                                    >
                                    <td
                                        class="px-4 py-2"
                                        class:text-red-600={cycle.taux_mortalite >
                                            20}
                                    >
                                        {cycle.taux_mortalite
                                            ? `${cycle.taux_mortalite}%`
                                            : "-"}
                                    </td>
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                </div>
            </Card>
        {/if}

        {#if lot.notes}
            <Card title="Notes">
                <p class="text-gray-600 whitespace-pre-wrap">{lot.notes}</p>
            </Card>
        {/if}
    {/if}
</div>
