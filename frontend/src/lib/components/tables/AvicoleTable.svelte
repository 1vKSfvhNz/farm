<!-- lib/components/tables/AvicoleTable.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import DataTable from "./DataTable.svelte";
    import Button from "../ui/Button.svelte";

    export let avicoles: Array<{
        id: number;
        identification: string;
        race: string;
        sexe: string;
        statut: string;
        age_jours?: number;
        dernier_poids?: number;
        enclos_name?: string;
        production_viande: boolean;
        production_ponte: boolean;
        production_reproduction: boolean;
        oeufs_pondus_total: number;
    }> = [];
    export let loading: boolean = false;
    export let selectable: boolean = true;

    const dispatch = createEventDispatcher();

    let selectedIds = new Set<number>();
    let sortKey: string | null = "identification";
    let sortDirection: "asc" | "desc" = "asc";

    const columns = [
        { key: "identification", label: "Identification", sortable: true },
        { key: "race", label: "Race", sortable: true },
        { key: "statut", label: "Statut", sortable: true },
        { key: "age_jours", label: "Âge", sortable: true },
        { key: "dernier_poids", label: "Poids (kg)", sortable: true },
        { key: "production", label: "Production", sortable: false },
        { key: "oeufs_pondus", label: "Œufs pondus", sortable: true },
        { key: "enclos_name", label: "Enclos", sortable: true },
    ];

    function formatAge(days: number): string {
        if (!days) return "-";
        const weeks = Math.floor(days / 7);
        if (weeks < 8) return `${weeks} sem`;
        const months = Math.floor(days / 30);
        return `${months} mois`;
    }

    const customRenderers = {
        sexe: (value: string) => {
            if (value === "male") {
                return '<span class="inline-flex items-center gap-1"><span class="w-2 h-2 bg-blue-500 rounded-full"></span> Coq</span>';
            }
            return '<span class="inline-flex items-center gap-1"><span class="w-2 h-2 bg-pink-500 rounded-full"></span> Poule</span>';
        },
        statut: (value: string) => {
            const statusMap: Record<string, string> = {
                vivant: '<span class="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">Vivant</span>',
                vendu: '<span class="px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">Vendu</span>',
                decede: '<span class="px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">Décédé</span>',
                transfere:
                    '<span class="px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">Transféré</span>',
            };
            return statusMap[value] || value;
        },
        age_jours: (value: number) => formatAge(value),
        dernier_poids: (value: number) => (value ? `${value} kg` : "-"),
        oeufs_pondus: (value: number) =>
            value ? value.toLocaleString("fr-FR") : "0",
        production: (_value: any, row: any) => {
            const productions = [];
            if (row.production_viande) productions.push("🍗 Viande");
            if (row.production_ponte) productions.push("🥚 Ponte");
            if (row.production_reproduction)
                productions.push("🔄 Reproduction");
            return productions.length > 0 ? productions.join("<br/>") : "-";
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
    function handleAddOeufs(item: any) {
        dispatch("addOeufs", item);
    }
    function handleSelect(selected: number[]) {
        selectedIds = new Set(selected);
        dispatch("select", { selectedIds: Array.from(selectedIds) });
    }
</script>

<DataTable
    {columns}
    data={avicoles}
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
            Nouvel avicole
        </Button>
    </div>
    <div slot="actions-row" let:row>
        <div class="flex items-center gap-2 justify-end">
            {#if row.sexe === "femelle" && row.production_ponte}
                <button
                    on:click={() => handleAddOeufs(row)}
                    class="p-1 text-yellow-600 hover:text-yellow-700 transition-colors"
                    title="Ajouter œufs"
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
