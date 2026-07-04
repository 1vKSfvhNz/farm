<!-- lib/components/tables/ApicultureTable.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import DataTable from "./DataTable.svelte";
    import Button from "../ui/Button.svelte";

    export let ruches: Array<{
        id: number;
        identification: string;
        emplacement?: string;
        race?: string;
        statut: string;
        date_installation: string;
        nombre_cadres?: number;
        production_miel_annee?: number;
    }> = [];
    export let loading: boolean = false;
    export let selectable: boolean = true;

    const dispatch = createEventDispatcher();

    let selectedIds = new Set<number>();
    let sortKey: string | null = "identification";
    let sortDirection: "asc" | "desc" = "asc";

    const columns = [
        { key: "identification", label: "Identification", sortable: true },
        { key: "emplacement", label: "Emplacement", sortable: true },
        { key: "race", label: "Race", sortable: true },
        { key: "statut", label: "Statut", sortable: true },
        { key: "date_installation", label: "Installation", sortable: true },
        { key: "nombre_cadres", label: "Cadres", sortable: true },
        { key: "production", label: "Production", sortable: true },
    ];

    const statutLabels: Record<string, string> = {
        active: '<span class="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">Active</span>',
        orpheline:
            '<span class="px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">Orpheline</span>',
        en_essaimage:
            '<span class="px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">En essaimage</span>',
        morte: '<span class="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800">Morte</span>',
    };

    const customRenderers = {
        statut: (value: string) => statutLabels[value] || value,
        date_installation: (value: string) =>
            new Date(value).toLocaleDateString("fr-FR"),
        production: (_value: any, row: any) => {
            if (row.production_miel_annee) {
                return `${row.production_miel_annee} kg/an`;
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
    function handleAddRecolte(item: any) {
        dispatch("addRecolte", item);
    }
    function handleSelect(selected: number[]) {
        selectedIds = new Set(selected);
        dispatch("select", { selectedIds: Array.from(selectedIds) });
    }
</script>

<DataTable
    {columns}
    data={ruches}
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
            Nouvelle ruche
        </Button>
    </div>
    <div slot="actions-row" let:row>
        <div class="flex items-center gap-2 justify-end">
            <button
                on:click={() => handleAddRecolte(row)}
                class="p-1 text-yellow-600 hover:text-yellow-700 transition-colors"
                title="Ajouter récolte"
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
                        d="M12 6v12m6-6H6"
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
