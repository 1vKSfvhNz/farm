<!-- lib/components/tables/EntomocultureTable.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import DataTable from "./DataTable.svelte";
    import Button from "../ui/Button.svelte";

    export let lots: Array<{
        id: number;
        identification: string;
        espece: string;
        stade_actuel: string;
        date_arrivee: string;
        quantite_estimative?: number;
        poids_initial?: number;
        enclos_name?: string;
        type_production: string;
        taux_mortalite?: number;
    }> = [];
    export let loading: boolean = false;
    export let selectable: boolean = true;

    const dispatch = createEventDispatcher();

    let selectedIds = new Set<number>();
    let sortKey: string | null = "identification";
    let sortDirection: "asc" | "desc" = "asc";

    const columns = [
        { key: "identification", label: "Lot", sortable: true },
        { key: "espece", label: "Espèce", sortable: true },
        { key: "stade_actuel", label: "Stade", sortable: true },
        { key: "date_arrivee", label: "Arrivée", sortable: true },
        { key: "quantite", label: "Quantité", sortable: true },
        { key: "poids", label: "Poids", sortable: true },
        { key: "mortalite", label: "Mortalité", sortable: true },
        { key: "enclos_name", label: "Enclos", sortable: true },
    ];

    const stadeLabels: Record<string, string> = {
        oeuf: '<span class="px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">Œuf</span>',
        larve: '<span class="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">Larve</span>',
        pupe: '<span class="px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">Pupe</span>',
        adulte: '<span class="px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">Adulte</span>',
    };

    const customRenderers = {
        stade_actuel: (value: string) => stadeLabels[value] || value,
        date_arrivee: (value: string) =>
            new Date(value).toLocaleDateString("fr-FR"),
        quantite: (_value: number, row: any) =>
            row.quantite_estimative
                ? row.quantite_estimative.toLocaleString("fr-FR")
                : "-",
        poids: (_value: number, row: any) =>
            row.poids_initial ? `${row.poids_initial} g` : "-",
        mortalite: (_value: number, row: any) => {
            if (row.taux_mortalite) {
                const color =
                    row.taux_mortalite > 20 ? "text-red-600" : "text-green-600";
                return `<span class="${color}">${row.taux_mortalite}%</span>`;
            }
            return "-";
        },
    };

    function handleView(item: any) {
        dispatch("view", item);
    }
    function handleEdit(item: any) {
        dispatch("edit", item);
    }
    function handleDelete(item: any) {
        dispatch("delete", item);
    }
    function handleAddCycle(item: any) {
        dispatch("addCycle", item);
    }
    function handleSelect(selected: number[]) {
        selectedIds = new Set(selected);
        dispatch("select", { selectedIds: Array.from(selectedIds) });
    }
</script>

<DataTable
    {columns}
    data={lots}
    {loading}
    {selectable}
    bind:selectedIds
    bind:sortKey
    bind:sortDirection
    {customRenderers}
    on:select={(e) => handleSelect(e.detail.selectedIds)}
>
    <div slot="actions" class="flex gap-2">
        <Button size="sm" variant="primary" on:click={() => dispatch("add")}>
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
            Nouveau lot
        </Button>
    </div>
    <div slot="actions-row" let:row>
        <div class="flex items-center gap-2 justify-end">
            <button
                on:click={() => handleAddCycle(row)}
                class="p-1 text-green-600 hover:text-green-700 transition-colors"
                title="Ajouter cycle"
            >
                <svg
                    class="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                    />
                </svg>
            </button>
            <button
                on:click={() => handleView(row)}
                class="p-1 text-gray-400 hover:text-primary-600 transition-colors"
                title="Voir"
            >
                <svg
                    class="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M15 12a3 3 0 11-6 0 3 3 0 016 0zM2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                    />
                </svg>
            </button>
            <button
                on:click={() => handleEdit(row)}
                class="p-1 text-gray-400 hover:text-blue-600 transition-colors"
                title="Modifier"
            >
                <svg
                    class="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                    />
                </svg>
            </button>
            <button
                on:click={() => handleDelete(row)}
                class="p-1 text-gray-400 hover:text-red-600 transition-colors"
                title="Supprimer"
            >
                <svg
                    class="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                </svg>
            </button>
        </div>
    </div>
</DataTable>
