<!-- lib/components/tables/VaccinationTable.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import DataTable from "./DataTable.svelte";
    import Button from "../ui/Button.svelte";

    export let vaccinations: Array<{
        id: number;
        animal_identification: string;
        maladie_nom: string;
        date_prevue: string;
        date_realisee?: string;
        est_a_jour: boolean;
        rappel_necessaire: boolean;
        veterinaire_responsable?: string;
    }> = [];
    export let loading: boolean = false;

    const dispatch = createEventDispatcher();

    let sortKey: string | null = "date_prevue";
    let sortDirection: "asc" | "desc" = "asc";

    const columns = [
        { key: "animal_identification", label: "Animal", sortable: true },
        { key: "maladie_nom", label: "Maladie", sortable: true },
        { key: "date_prevue", label: "Date prévue", sortable: true },
        { key: "statut", label: "Statut", sortable: true },
        {
            key: "veterinaire_responsable",
            label: "Vétérinaire",
            sortable: true,
        },
    ];

    const customRenderers = {
        date_prevue: (value: string) => {
            const date = new Date(value);
            const today = new Date();
            const isOverdue = date < today;
            return `<span class="${isOverdue ? "text-red-600" : "text-gray-600"}">${date.toLocaleDateString("fr-FR")}</span>`;
        },
        statut: (_value: any, row: any) => {
            if (row.date_realisee) {
                return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">Réalisé</span>';
            }
            const date = new Date(row.date_prevue);
            const today = new Date();
            if (date < today) {
                return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">En retard</span>';
            }
            return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">À venir</span>';
        },
    };

    function handleEdit(item: any) {
        dispatch("edit", item);
    }

    function handleDelete(item: any) {
        dispatch("delete", item);
    }

    function handleRealize(item: any) {
        dispatch("realize", item);
    }
</script>

<DataTable
    {columns}
    data={vaccinations}
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
            Nouvelle vaccination
        </Button>
    </div>

    <div slot="actions-row" let:row>
        <div class="flex items-center gap-2 justify-end">
            {#if !row.date_realisee}
                <button
                    on:click={() => handleRealize(row)}
                    class="p-1 text-green-600 hover:text-green-700 transition-colors"
                    title="Marquer comme réalisé"
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
                            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                    </svg>
                </button>
            {/if}
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
