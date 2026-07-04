<!-- lib/components/tables/EnclosTable.svelte - Version simplifiée -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import DataTable from "./DataTable.svelte";
    import Button from "../ui/Button.svelte";

    export let enclos: Array<{
        id: number;
        name: string;
        type: string;
        surface: number;
        capacite_maximale: number;
        occupation_actuelle: number;
        taux_occupation: number;
        zone?: string;
    }> = [];
    export let loading: boolean = false;

    const dispatch = createEventDispatcher();

    let sortKey: string | null = "name";
    let sortDirection: "asc" | "desc" = "asc";

    const columns = [
        { key: "name", label: "Nom", sortable: true },
        { key: "type", label: "Type", sortable: true },
        { key: "surface", label: "Surface (m²)", sortable: true },
        { key: "occupation", label: "Occupation", sortable: true },
        { key: "taux_occupation", label: "Taux", sortable: true },
        { key: "zone", label: "Zone", sortable: true },
    ];

    const typeLabels: Record<string, string> = {
        enclos: "Enclos",
        bassin: "Bassin",
        pâturage: "Pâturage",
        cage: "Cage",
        bac: "Bac",
    };

    // Renderers personnalisés
    const customRenderers = {
        type: (value: string) => typeLabels[value] || value,
        occupation: (_value: any, row: any) => {
            const rate = row.taux_occupation;
            const colorClass =
                rate >= 90
                    ? "text-red-600"
                    : rate >= 75
                      ? "text-yellow-600"
                      : "text-green-600";
            return `<span class="${colorClass}">${row.occupation_actuelle} / ${row.capacite_maximale}</span>`;
        },
        surface: (value: number) => `${value} m²`,
        zone: (value: string) => value || "-",
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
</script>

<DataTable
    {columns}
    data={enclos}
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
            Nouvel enclos
        </Button>
    </div>

    <div slot="actions-row" let:row>
        <div class="flex items-center gap-2 justify-end">
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
                        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
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
