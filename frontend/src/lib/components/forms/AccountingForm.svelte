<!-- lib/components/forms/AccountingForm.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Input from "../ui/Input.svelte";
    import Select from "../ui/Select.svelte";
    import DatePicker from "../ui/DatePicker.svelte";
    import Button from "../ui/Button.svelte";

    export let type: "depense" | "recette" = "depense";
    export let formData: {
        categorie: string;
        montant: number;
        date: string;
        description?: string;
        fournisseur?: string;
        client?: string;
        quantite?: number;
        prix_unitaire?: number;
        piece_jointe_url?: string;
    };
    export let loading: boolean = false;

    const dispatch = createEventDispatcher();

    let errors: Record<string, string> = {};

    // Variables locales pour le binding
    let localCategorie = formData.categorie;
    let localMontant = formData.montant?.toString() || "";
    let localDate = formData.date;
    let localDescription = formData.description || "";
    let localFournisseur = formData.fournisseur || "";
    let localClient = formData.client || "";
    let localQuantite = formData.quantite?.toString() || "";
    let localPrixUnitaire = formData.prix_unitaire?.toString() || "";
    let localPieceJointe = formData.piece_jointe_url || "";

    const depenseCategories = [
        { value: "achat_animaux", label: "Achat animaux" },
        { value: "achat_oeufs", label: "Achat œufs" },
        { value: "alimentation", label: "Alimentation" },
        { value: "vaccins_soins", label: "Vaccins & Soins" },
        { value: "equipement", label: "Équipement" },
        { value: "personnel", label: "Personnel" },
        { value: "eau_electricite", label: "Eau & Électricité" },
        { value: "entretien", label: "Entretien" },
        { value: "compostage", label: "Compostage" },
        { value: "transport", label: "Transport" },
        { value: "frais_divers", label: "Frais divers" },
    ];

    const recetteCategories = [
        { value: "vente_animaux_vivants", label: "Vente animaux vivants" },
        { value: "vente_viande", label: "Vente viande" },
        { value: "vente_lait", label: "Vente lait" },
        { value: "vente_laine", label: "Vente laine" },
        { value: "vente_oeufs", label: "Vente œufs" },
        { value: "vente_larves", label: "Vente larves" },
        { value: "vente_compost", label: "Vente compost" },
        { value: "vente_fumier", label: "Vente fumier" },
        { value: "subventions", label: "Subventions" },
        { value: "autres", label: "Autres" },
    ];

    function validate(): boolean {
        errors = {};
        if (!localCategorie) errors.categorie = "La catégorie est requise";
        if (!localMontant || parseFloat(localMontant) <= 0)
            errors.montant = "Le montant doit être supérieur à 0";
        if (!localDate) errors.date = "La date est requise";
        return Object.keys(errors).length === 0;
    }

    function handleSubmit() {
        if (validate()) {
            dispatch("submit", {
                type,
                categorie: localCategorie,
                montant: parseFloat(localMontant),
                date: localDate,
                description: localDescription || undefined,
                fournisseur: localFournisseur || undefined,
                client: localClient || undefined,
                quantite: localQuantite ? parseFloat(localQuantite) : undefined,
                prix_unitaire: localPrixUnitaire
                    ? parseFloat(localPrixUnitaire)
                    : undefined,
                piece_jointe_url: localPieceJointe || undefined,
            });
        }
    }

    function handleCancel() {
        dispatch("cancel");
    }

    $: categories = type === "depense" ? depenseCategories : recetteCategories;
</script>

<div class="space-y-5">
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Select
            label="Catégorie"
            bind:value={localCategorie}
            options={categories}
            required
            error={errors.categorie}
        />

        <Input
            label="Montant (€)"
            bind:value={localMontant}
            inputType="number"
            required
            error={errors.montant}
            placeholder="0.00"
        />

        <DatePicker
            label="Date"
            bind:value={localDate}
            required
            error={errors.date}
        />

        {#if type === "depense"}
            <Input
                label="Fournisseur"
                bind:value={localFournisseur}
                placeholder="Nom du fournisseur"
            />
        {:else}
            <Input
                label="Client"
                bind:value={localClient}
                placeholder="Nom du client"
            />
        {/if}

        <Input
            label="Quantité"
            bind:value={localQuantite}
            inputType="number"
            placeholder="1"
        />

        <Input
            label="Prix unitaire (€)"
            bind:value={localPrixUnitaire}
            inputType="number"
            placeholder="0.00"
        />
    </div>

    <div>
        <!-- svelte-ignore a11y-label-has-associated-control -->
        <label class="block text-sm font-medium text-gray-700 mb-1"
            >Description</label
        >
        <textarea
            bind:value={localDescription}
            rows={3}
            class="w-full rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-2 focus:ring-primary-100 px-4 py-2 transition-all"
            placeholder="Description détaillée..."
        />
    </div>

    {#if type === "depense"}
        <Input
            label="Pièce jointe (URL)"
            bind:value={localPieceJointe}
            placeholder="https://..."
        />
    {/if}

    <div class="flex justify-end gap-3 pt-4 border-t border-gray-200">
        <Button on:click={handleCancel} variant="outline">Annuler</Button>
        <Button on:click={handleSubmit} {loading} variant="primary">
            Ajouter
        </Button>
    </div>
</div>
