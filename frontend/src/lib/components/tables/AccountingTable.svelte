<!-- lib/components/tables/AccountingTable.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import DataTable from "./DataTable.svelte";
    import Button from "../ui/Button.svelte";

    export let transactions: Array<{
        id: number;
        type: "depense" | "recette";
        categorie: string;
        montant: number;
        date: string;
        description?: string;
        fournisseur?: string;
        client?: string;
    }> = [];
    export let loading: boolean = false;
    export let typeFilter: "all" | "depense" | "recette" = "all";

    const dispatch = createEventDispatcher();

    let sortKey: string | null = "date";
    let sortDirection: "asc" | "desc" = "desc";

    let filteredTransactions = transactions;

    $: {
        if (typeFilter === "all") {
            filteredTransactions = transactions;
        } else {
            filteredTransactions = transactions.filter(
                (t) => t.type === typeFilter,
            );
        }
    }

    const columns = [
        { key: "date", label: "Date", sortable: true, width: "120px" },
        { key: "type", label: "Type", sortable: true, width: "100px" },
        { key: "categorie", label: "Catégorie", sortable: true },
        { key: "description", label: "Description" },
        { key: "montant", label: "Montant", sortable: true, width: "120px" },
    ];

    const categorieLabels: Record<string, string> = {
        achat_animaux: "Achat animaux",
        achat_oeufs: "Achat œufs",
        alimentation: "Alimentation",
        vaccins_soins: "Vaccins & Soins",
        equipement: "Équipement",
        personnel: "Personnel",
        eau_electricite: "Eau & Électricité",
        entretien: "Entretien",
        compostage: "Compostage",
        transport: "Transport",
        frais_divers: "Frais divers",
        vente_animaux_vivants: "Vente animaux vivants",
        vente_viande: "Vente viande",
        vente_lait: "Vente lait",
        vente_laine: "Vente laine",
        vente_oeufs: "Vente œufs",
        vente_larves: "Vente larves",
        vente_compost: "Vente compost",
        vente_fumier: "Vente fumier",
        subventions: "Subventions",
        autres: "Autres",
    };

    // Renderers personnalisés
    const customRenderers = {
        type: (value: string) => {
            if (value === "depense") {
                return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">Dépense</span>';
            }
            return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">Recette</span>';
        },
        categorie: (value: string) => categorieLabels[value] || value,
        montant: (value: number, row: any) => {
            const sign = row.type === "depense" ? "-" : "+";
            const colorClass =
                row.type === "depense"
                    ? "text-red-600"
                    : "text-green-600 font-medium";
            return `<span class="${colorClass}">${sign}${value.toLocaleString("fr-FR")} €</span>`;
        },
        date: (value: string) => new Date(value).toLocaleDateString("fr-FR"),
    };

    function handleEdit(transaction: any) {
        dispatch("edit", transaction);
    }

    function handleDelete(transaction: any) {
        dispatch("delete", transaction);
    }
</script>

<div class="space-y-4">
    <div class="flex justify-between items-center">
        <div class="flex gap-2">
            <button
                on:click={() => (typeFilter = "all")}
                class={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                    typeFilter === "all"
                        ? "bg-primary-600 text-white"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
            >
                Tous
            </button>

            <button
                on:click={() => (typeFilter = "depense")}
                class={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                    typeFilter === "depense"
                        ? "bg-red-600 text-white"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
            >
                Dépenses
            </button>

            <button
                on:click={() => (typeFilter = "recette")}
                class={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                    typeFilter === "recette"
                        ? "bg-green-600 text-white"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
            >
                Recettes
            </button>
        </div>

        <div class="flex gap-2">
            <Button
                size="sm"
                variant="outline"
                on:click={() => dispatch("export")}
            >
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
                        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                    />
                </svg>
                Exporter
            </Button>
            <Button
                size="sm"
                variant="primary"
                on:click={() => dispatch("add")}
            >
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
                Nouvelle transaction
            </Button>
        </div>
    </div>

    <DataTable
        {columns}
        data={filteredTransactions}
        {loading}
        selectable={false}
        bind:sortKey
        bind:sortDirection
        {customRenderers}
    >
        <div slot="actions-row" let:row>
            <div class="flex items-center gap-2 justify-end">
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
</div>
