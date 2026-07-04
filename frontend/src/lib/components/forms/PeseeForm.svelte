<!-- lib/components/forms/PeseeForm.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Input from "../ui/Input.svelte";
    import DatePicker from "../ui/DatePicker.svelte";
    import Button from "../ui/Button.svelte";

    export let formData: {
        animal_id: number;
        date_pesee: string;
        poids?: number;
        methode?: string;
        notes?: string;
    };
    export let loading: boolean = false;
    export let isEdit: boolean = false;

    const dispatch = createEventDispatcher();

    let errors: Record<string, string> = {};

    // Variables locales pour le binding
    let localDatePesee = formData.date_pesee;
    let localPoids = formData.poids?.toString() || "";
    let localMethode = formData.methode || "";
    let localNotes = formData.notes || "";

    function validate(): boolean {
        errors = {};
        if (!localDatePesee) errors.date_pesee = "La date est requise";
        if (!localPoids || parseFloat(localPoids) <= 0)
            errors.poids = "Le poids doit être supérieur à 0";
        return Object.keys(errors).length === 0;
    }

    function handleSubmit() {
        if (validate()) {
            dispatch("submit", {
                animal_id: formData.animal_id,
                date_pesee: localDatePesee,
                poids: parseFloat(localPoids),
                methode: localMethode || undefined,
                notes: localNotes || undefined,
            });
        }
    }

    function handleCancel() {
        dispatch("cancel");
    }
</script>

<div class="space-y-5">
    <DatePicker
        label="Date de pesée"
        bind:value={localDatePesee}
        required
        error={errors.date_pesee}
    />

    <Input
        label="Poids (kg)"
        bind:value={localPoids}
        inputType="number"
        required
        error={errors.poids}
        placeholder="0.00"
    />

    <Input
        label="Méthode"
        bind:value={localMethode}
        placeholder="Ex: Balance électronique, Pesée manuelle..."
    />

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
            {isEdit ? "Mettre à jour" : "Ajouter la pesée"}
        </Button>
    </div>
</div>
