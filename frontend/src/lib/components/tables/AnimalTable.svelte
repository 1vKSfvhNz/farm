<!-- lib/components/tables/AnimalTable.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import DataTable from "./DataTable.svelte";
    import Button from "../ui/Button.svelte";

    export let animals: Array<{
        id: number;
        identification: string;
        race: string;
        sexe: string;
        statut: string;
        age_jours?: number;
        dernier_poids?: number;
        enclos_id: number;
        enclos_name?: string;
        type_espece: string;
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
        { key: "sexe", label: "Sexe", sortable: true },
        { key: "statut", label: "Statut", sortable: true },
        { key: "age_jours", label: "Âge", sortable: true },
        { key: "dernier_poids", label: "Poids (kg)", sortable: true },
        { key: "enclos_name", label: "Enclos", sortable: true },
    ];

    function formatAge(days: number): string {
        if (!days) return "-";
        const years = Math.floor(days / 365);
        const months = Math.floor((days % 365) / 30);
        if (years > 0) return `${years} an${years > 1 ? "s" : ""}`;
        if (months > 0) return `${months} mois`;
        return `${days} jour${days > 1 ? "s" : ""}`;
    }

    // Renderers personnalisés pour chaque colonne
    const customRenderers = {
        sexe: (value: string) => {
            if (value === "male") {
                return '<span class="inline-flex items-center gap-1"><span class="w-2 h-2 bg-blue-500 rounded-full"></span> Mâle</span>';
            } else if (value === "femelle") {
                return '<span class="inline-flex items-center gap-1"><span class="w-2 h-2 bg-pink-500 rounded-full"></span> Femelle</span>';
            } else {
                return `<span class="inline-flex items-center gap-1"><span class="w-2 h-2 bg-purple-500 rounded-full"></span> ${value}</span>`;
            }
        },
        statut: (value: string) => {
            if (value === "vivant") {
                return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">Vivant</span>';
            } else if (value === "vendu") {
                return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">Vendu</span>';
            } else if (value === "decede") {
                return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">Décédé</span>';
            } else {
                return `<span class="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800">${value}</span>`;
            }
        },
        age_jours: (value: number) => formatAge(value),
        dernier_poids: (value: number) => (value ? `${value} kg` : "-"),
    };

    function handleView(animal: any) {
        dispatch("view", animal);
    }

    function handleEdit(animal: any) {
        dispatch("edit", animal);
    }

    function handleDelete(animal: any) {
        dispatch("delete", animal);
    }

    function handleSelect(selected: number[]) {
        selectedIds = new Set(selected);
        dispatch("select", { selectedIds: Array.from(selectedIds) });
    }
</script>

<DataTable
    {columns}
    data={animals}
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
            Nouvel animal
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
