<!-- lib/components/forms/EntomocultureForm.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Input from "../ui/Input.svelte";
    import Select from "../ui/Select.svelte";
    import DatePicker from "../ui/DatePicker.svelte";
    import Button from "../ui/Button.svelte";

    export let formData: {
        identification: string;
        espece: string;
        stade_actuel: string;
        date_arrivee: string;
        provenance?: string;
        prix_achat?: number;
        poids_initial?: number;
        quantite_estimative?: number;
        enclos_id?: number;
        type_production: string;
        notes?: string;
    };
    export let enclosOptions: Array<{ value: number; label: string }> = [];
    export let loading: boolean = false;
    export let isEdit: boolean = false;

    const dispatch = createEventDispatcher();

    let errors: Record<string, string> = {};

    let localIdentification = formData.identification;
    let localEspece = formData.espece;
    let localStadeActuel = formData.stade_actuel;
    let localDateArrivee = formData.date_arrivee;
    let localProvenance = formData.provenance || "";
    let localPrixAchat = formData.prix_achat?.toString() || "";
    let localPoidsInitial = formData.poids_initial?.toString() || "";
    let localQuantiteEstimative =
        formData.quantite_estimative?.toString() || "";
    let localEnclosId = formData.enclos_id || "";
    let localTypeProduction = formData.type_production;
    let localNotes = formData.notes || "";

    const stadeOptions = [
        { value: "oeuf", label: "Œuf" },
        { value: "larve", label: "Larve" },
        { value: "pupe", label: "Pupe" },
        { value: "adulte", label: "Adulte" },
    ];

    const typeProductionOptions = [
        { value: "larves", label: "Production de larves" },
        { value: "reproduction", label: "Reproduction" },
        { value: "oeufs", label: "Production d'œufs" },
    ];

    function validate(): boolean {
        errors = {};
        if (!localIdentification)
            errors.identification = "L'identification est requise";
        if (!localEspece) errors.espece = "L'espèce est requise";
        if (!localStadeActuel) errors.stade_actuel = "Le stade est requis";
        if (!localDateArrivee)
            errors.date_arrivee = "La date d'arrivée est requise";
        if (!localTypeProduction)
            errors.type_production = "Le type de production est requis";
        return Object.keys(errors).length === 0;
    }

    function handleSubmit() {
        if (validate()) {
            dispatch("submit", {
                identification: localIdentification,
                espece: localEspece,
                stade_actuel: localStadeActuel,
                date_arrivee: localDateArrivee,
                provenance: localProvenance || undefined,
                prix_achat: localPrixAchat
                    ? parseFloat(localPrixAchat)
                    : undefined,
                poids_initial: localPoidsInitial
                    ? parseFloat(localPoidsInitial)
                    : undefined,
                quantite_estimative: localQuantiteEstimative
                    ? parseInt(localQuantiteEstimative)
                    : undefined,
                enclos_id: localEnclosId
                    ? typeof localEnclosId === "string"
                        ? parseInt(localEnclosId, 10)
                        : localEnclosId
                    : undefined,
                type_production: localTypeProduction,
                notes: localNotes || undefined,
            });
        }
    }

    function handleCancel() {
        dispatch("cancel");
    }
</script>

<div class="space-y-6">
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Input
            label="Identification du lot"
            bind:value={localIdentification}
            required
            error={errors.identification}
            placeholder="Ex: LOT-LARVE-001"
        />
        <Input
            label="Espèce"
            bind:value={localEspece}
            required
            error={errors.espece}
            placeholder="Ex: Tenebrio molitor, Hermetia illucens..."
        />
        <Select
            label="Stade actuel"
            bind:value={localStadeActuel}
            options={stadeOptions}
            required
            error={errors.stade_actuel}
        />
        <Select
            label="Type de production"
            bind:value={localTypeProduction}
            options={typeProductionOptions}
            required
            error={errors.type_production}
        />
        <DatePicker
            label="Date d'arrivée"
            bind:value={localDateArrivee}
            required
            error={errors.date_arrivee}
        />
        <Input
            label="Provenance"
            bind:value={localProvenance}
            placeholder="Ex: Élevage..."
        />
        <Input
            label="Prix d'achat (€)"
            bind:value={localPrixAchat}
            inputType="number"
            placeholder="0"
        />
        <Input
            label="Poids initial (g)"
            bind:value={localPoidsInitial}
            inputType="number"
            placeholder="0"
        />
        <Input
            label="Quantité estimative"
            bind:value={localQuantiteEstimative}
            inputType="number"
            placeholder="Nombre d'individus"
        />
        <Select
            label="Enclos"
            bind:value={localEnclosId}
            options={[{ value: "", label: "Aucun" }, ...enclosOptions]}
        />
    </div>

    <div>
        <!-- svelte-ignore a11y-label-has-associated-control -->
        <label class="block text-sm font-medium text-gray-700 mb-1">Notes</label
        >
        <textarea
            bind:value={localNotes}
            rows={3}
            class="w-full rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-2 focus:ring-primary-100 px-4 py-2 transition-all"
            placeholder="Informations supplémentaires..."
        />
    </div>

    <div class="flex justify-end gap-3 pt-4 border-t border-gray-200">
        <Button on:click={handleCancel} variant="outline">Annuler</Button>
        <Button on:click={handleSubmit} {loading} variant="primary"
            >{isEdit ? "Mettre à jour" : "Créer le lot"}</Button
        >
    </div>
</div>
