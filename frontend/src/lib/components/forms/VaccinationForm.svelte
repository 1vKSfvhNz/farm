<!-- lib/components/forms/VaccinationForm.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Input from "../ui/Input.svelte";
    import Select from "../ui/Select.svelte";
    import DatePicker from "../ui/DatePicker.svelte";
    import Button from "../ui/Button.svelte";

    export let formData: {
        animal_id: number;
        maladie_id: number;
        vaccin_id?: number;
        date_prevue: string;
        date_realisee?: string;
        dose?: string;
        rappel_necessaire: boolean;
        date_prochain_rappel?: string;
        veterinaire_responsable?: string;
        cout?: number;
        notes?: string;
    };
    export let maladieOptions: Array<{ value: number; label: string }> = [];
    export let vaccinOptions: Array<{ value: number; label: string }> = [];
    export let loading: boolean = false;
    export let isEdit: boolean = false;

    const dispatch = createEventDispatcher();

    let errors: Record<string, string> = {};

    // Variables locales pour le binding
    let localMaladieId = formData.maladie_id;
    let localVaccinId = formData.vaccin_id || "";
    let localDatePrevue = formData.date_prevue;
    let localDateRealisee = formData.date_realisee || "";
    let localDose = formData.dose || "";
    let localRappelNecessaire = formData.rappel_necessaire;
    let localDateProchainRappel = formData.date_prochain_rappel || "";
    let localVeterinaireResponsable = formData.veterinaire_responsable || "";
    let localCout = formData.cout?.toString() || "";
    let localNotes = formData.notes || "";

    function validate(): boolean {
        errors = {};
        if (!localDatePrevue) errors.date_prevue = "La date prévue est requise";
        if (!localMaladieId) errors.maladie_id = "La maladie est requise";
        return Object.keys(errors).length === 0;
    }

    function handleSubmit() {
        if (validate()) {
            dispatch("submit", {
                animal_id: formData.animal_id,
                maladie_id: localMaladieId,
                vaccin_id: localVaccinId ? Number(localVaccinId) : undefined,
                date_prevue: localDatePrevue,
                date_realisee: localDateRealisee || undefined,
                dose: localDose || undefined,
                rappel_necessaire: localRappelNecessaire,
                date_prochain_rappel: localDateProchainRappel || undefined,
                veterinaire_responsable:
                    localVeterinaireResponsable || undefined,
                cout: localCout ? parseFloat(localCout) : undefined,
                notes: localNotes || undefined,
            });
        }
    }

    function handleCancel() {
        dispatch("cancel");
    }

    const vaccinOptionsStr = [
        { value: "", label: "Sélectionner un vaccin..." },
        ...vaccinOptions.map((opt) => ({
            value: String(opt.value),
            label: opt.label,
        })),
    ];
</script>

<div class="space-y-5">
    <Select
        label="Maladie"
        bind:value={localMaladieId}
        options={maladieOptions}
        required
        error={errors.maladie_id}
    />

    <Select
        label="Vaccin"
        bind:value={localVaccinId}
        options={vaccinOptionsStr}
        placeholder="Sélectionner un vaccin..."
    />

    <DatePicker
        label="Date prévue"
        bind:value={localDatePrevue}
        required
        error={errors.date_prevue}
    />

    <DatePicker label="Date réalisée" bind:value={localDateRealisee} />

    <Input label="Dose" bind:value={localDose} placeholder="Ex: 2ml, 5ml..." />

    <Input
        label="Vétérinaire responsable"
        bind:value={localVeterinaireResponsable}
        placeholder="Nom du vétérinaire"
    />

    <Input
        label="Coût (€)"
        bind:value={localCout}
        inputType="number"
        placeholder="0.00"
    />

    <div class="flex items-center gap-2">
        <input
            type="checkbox"
            bind:checked={localRappelNecessaire}
            class="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
        />
        <!-- svelte-ignore a11y-label-has-associated-control -->
        <label class="text-sm text-gray-700">Rappel nécessaire</label>
    </div>

    {#if localRappelNecessaire}
        <DatePicker
            label="Date du prochain rappel"
            bind:value={localDateProchainRappel}
        />
    {/if}

    <div>
        <!-- svelte-ignore a11y-label-has-associated-control -->
        <label class="block text-sm font-medium text-gray-700 mb-1">Notes</label
        >
        <textarea
            bind:value={localNotes}
            rows={2}
            class="w-full rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-2 focus:ring-primary-100 px-4 py-2 transition-all"
            placeholder="Observations..."
        />
    </div>

    <div class="flex justify-end gap-3 pt-4 border-t border-gray-200">
        <Button on:click={handleCancel} variant="outline">Annuler</Button>
        <Button on:click={handleSubmit} {loading} variant="primary">
            {isEdit ? "Mettre à jour" : "Ajouter la vaccination"}
        </Button>
    </div>
</div>
