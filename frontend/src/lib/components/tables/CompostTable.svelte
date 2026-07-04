<!-- lib/components/tables/CompostTable.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import DataTable from "./DataTable.svelte";
    import Button from "../ui/Button.svelte";

    export let composts: Array<{
        id: number;
        name: string;
        type: string;
        date_demarrage: string;
        volume_initial: number;
        volume_final?: number;
        date_maturite_estimee?: string;
        date_maturite_reelle?: string;
        statut: string;
    }> = [];
    export let loading: boolean = false;

    const dispatch = createEventDispatcher();

    let sortKey: string | null = "date_demarrage";
    let sortDirection: "asc" | "desc" = "desc";

    const columns = [
        { key: "name", label: "Nom", sortable: true },
        { key: "type", label: "Type", sortable: true },
        { key: "date_demarrage", label: "Démarrage", sortable: true },
        { key: "volume", label: "Volume", sortable: true },
        { key: "statut", label: "Statut", sortable: true },
        { key: "date_maturite", label: "Maturité", sortable: true },
    ];

    const typeLabels: Record<string, string> = {
        "déchets verts": "Déchets verts",
        fumier: "Fumier",
        mixte: "Mixte",
    };

    const customRenderers = {
        type: (value: string) => typeLabels[value] || value,
        volume: (_value: any, row: any) => {
            if (row.volume_final) {
                return `${row.volume_initial} m³ → ${row.volume_final} m³`;
            }
            return `${row.volume_initial} m³`;
        },
        statut: (_value: any, row: any) => {
            if (row.date_maturite_reelle) {
                return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">Mature</span>';
            }
            if (row.date_maturite_estimee) {
                const date = new Date(row.date_maturite_estimee);
                const today = new Date();
                if (date < today) {
                    return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">En retard</span>';
                }
                return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">En cours</span>';
            }
            return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800">Démarré</span>';
        },
        date_maturite: (_value: any, row: any) => {
            if (row.date_maturite_reelle) {
                return new Date(row.date_maturite_reelle).toLocaleDateString(
                    "fr-FR",
                );
            }
            if (row.date_maturite_estimee) {
                return `Estimée: ${new Date(row.date_maturite_estimee).toLocaleDateString("fr-FR")}`;
            }
            return "-";
        },
        date_demarrage: (value: string) =>
            new Date(value).toLocaleDateString("fr-FR"),
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

    function handleMarkMature(item: any) {
        dispatch("mature", item);
    }
</script>

<DataTable
    {columns}
    data={composts}
    {loading}
    selectable={false}
    bind:sortKey
    bind:sortDirection
    {customRenderers}
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
            Nouveau compost
        </Button>
    </div>

    <div slot="actions-row" let:row>
        <div class="flex items-center gap-2 justify-end">
            {#if !row.date_maturite_reelle}
                <button
                    on:click={() => handleMarkMature(row)}
                    class="p-1 text-green-600 hover:text-green-700 transition-colors"
                    title="Marquer comme mature"
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
                            d="M5 13l4 4L19 7"
                        />
                    </svg>
                </button>
            {/if}
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
