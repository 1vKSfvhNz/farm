<!-- lib/components/forms/CompostForm.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Input from "../ui/Input.svelte";
    import Select from "../ui/Select.svelte";
    import DatePicker from "../ui/DatePicker.svelte";
    import Button from "../ui/Button.svelte";

    export let formData: {
        name: string;
        type: string;
        date_demarrage: string;
        volume_initial: number;
        volume_final?: number;
        date_maturite_estimee?: string;
        date_maturite_reelle?: string;
        utilisation_finale?: string;
        notes?: string;
    };
    export let loading: boolean = false;
    export let isEdit: boolean = false;

    const dispatch = createEventDispatcher();

    let errors: Record<string, string> = {};

    let localName = formData.name;
    let localType = formData.type;
    let localDateDemarrage = formData.date_demarrage;
    let localVolumeInitial = formData.volume_initial?.toString() || "";
    let localVolumeFinal = formData.volume_final?.toString() || "";
    let localDateMaturiteEstimee = formData.date_maturite_estimee || "";
    let localDateMaturiteReelle = formData.date_maturite_reelle || "";
    let localUtilisationFinale = formData.utilisation_finale || "";
    let localNotes = formData.notes || "";

    const typeOptions = [
        { value: "déchets verts", label: "Déchets verts" },
        { value: "fumier", label: "Fumier" },
        { value: "mixte", label: "Mixte" },
    ];

    const utilisationOptions = [
        { value: "amendement_sol", label: "Amendement du sol" },
        { value: "paillage", label: "Paillage" },
        { value: "vente", label: "Vente" },
        { value: "autres", label: "Autres" },
    ];

    function validate(): boolean {
        errors = {};
        if (!localName) errors.name = "Le nom est requis";
        if (!localType) errors.type = "Le type est requis";
        if (!localDateDemarrage)
            errors.date_demarrage = "La date de démarrage est requise";
        if (!localVolumeInitial || parseFloat(localVolumeInitial) <= 0)
            errors.volume_initial = "Le volume initial doit être supérieur à 0";
        return Object.keys(errors).length === 0;
    }

    function handleSubmit() {
        if (validate()) {
            dispatch("submit", {
                name: localName,
                type: localType,
                date_demarrage: localDateDemarrage,
                volume_initial: parseFloat(localVolumeInitial),
                volume_final: localVolumeFinal
                    ? parseFloat(localVolumeFinal)
                    : undefined,
                date_maturite_estimee: localDateMaturiteEstimee || undefined,
                date_maturite_reelle: localDateMaturiteReelle || undefined,
                utilisation_finale: localUtilisationFinale || undefined,
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
            label="Nom du compost"
            bind:value={localName}
            required
            error={errors.name}
            placeholder="Ex: Compost Fumier 2024"
        />
        <Select
            label="Type"
            bind:value={localType}
            options={typeOptions}
            required
            error={errors.type}
        />
        <DatePicker
            label="Date de démarrage"
            bind:value={localDateDemarrage}
            required
            error={errors.date_demarrage}
        />
        <DatePicker
            label="Date maturité estimée"
            bind:value={localDateMaturiteEstimee}
        />
        <Input
            label="Volume initial (m³)"
            bind:value={localVolumeInitial}
            inputType="number"
            required
            error={errors.volume_initial}
            placeholder="0"
        />
        <Input
            label="Volume final (m³)"
            bind:value={localVolumeFinal}
            inputType="number"
            placeholder="0"
        />
    </div>

    {#if localDateMaturiteReelle}
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <DatePicker
                label="Date maturité réelle"
                bind:value={localDateMaturiteReelle}
            />
            <Select
                label="Utilisation finale"
                bind:value={localUtilisationFinale}
                options={utilisationOptions}
                placeholder="Sélectionner..."
            />
        </div>
    {/if}

    <div>
        <!-- svelte-ignore a11y-label-has-associated-control -->
        <label class="block text-sm font-medium text-gray-700 mb-1">Notes</label
        >
        <textarea
            bind:value={localNotes}
            rows={3}
            class="w-full rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-2 focus:ring-primary-100 px-4 py-2 transition-all"
            placeholder="Observations sur le processus..."
        />
    </div>

    <div class="flex justify-end gap-3 pt-4 border-t border-gray-200">
        <Button on:click={handleCancel} variant="outline">Annuler</Button>
        <Button on:click={handleSubmit} {loading} variant="primary"
            >{isEdit ? "Mettre à jour" : "Créer"}</Button
        >
    </div>
</div>
