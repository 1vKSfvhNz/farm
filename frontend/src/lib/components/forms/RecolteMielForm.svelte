<!-- lib/components/forms/RecolteMielForm.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Input from "../ui/Input.svelte";
    import Select from "../ui/Select.svelte";
    import DatePicker from "../ui/DatePicker.svelte";
    import Button from "../ui/Button.svelte";

    export let formData: {
        ruche_id: number;
        date_recolte: string;
        poids_kg: number;
        qualite?: string;
        taux_eau?: number;
        notes?: string;
    };
    export let rucheOptions: Array<{ value: number; label: string }> = [];
    export let loading: boolean = false;
    export let isEdit: boolean = false;

    const dispatch = createEventDispatcher();

    let errors: Record<string, string> = {};

    let localRucheId = formData.ruche_id;
    let localDateRecolte = formData.date_recolte;
    let localPoidsKg = formData.poids_kg?.toString() || "";
    let localQualite = formData.qualite || "";
    let localTauxEau = formData.taux_eau?.toString() || "";
    let localNotes = formData.notes || "";

    const qualiteOptions = [
        { value: "toutes_fleurs", label: "Toutes fleurs" },
        { value: "miellat", label: "Miellat" },
        { value: "acacia", label: "Acacia" },
        { value: "chataignier", label: "Châtaignier" },
        { value: "lavande", label: "Lavande" },
        { value: "tournesol", label: "Tournesol" },
        { value: "foret", label: "Forêt" },
    ];

    function validate(): boolean {
        errors = {};
        if (!localRucheId) errors.ruche_id = "La ruche est requise";
        if (!localDateRecolte) errors.date_recolte = "La date est requise";
        if (!localPoidsKg || parseFloat(localPoidsKg) <= 0)
            errors.poids_kg = "Le poids doit être supérieur à 0";
        if (
            localTauxEau &&
            (parseFloat(localTauxEau) < 0 || parseFloat(localTauxEau) > 30)
        ) {
            errors.taux_eau = "Le taux d'eau doit être entre 0 et 30%";
        }
        return Object.keys(errors).length === 0;
    }

    function handleSubmit() {
        if (validate()) {
            dispatch("submit", {
                ruche_id: localRucheId,
                date_recolte: localDateRecolte,
                poids_kg: parseFloat(localPoidsKg),
                qualite: localQualite || undefined,
                taux_eau: localTauxEau ? parseFloat(localTauxEau) : undefined,
                notes: localNotes || undefined,
            });
        }
    }

    function handleCancel() {
        dispatch("cancel");
    }
</script>

<div class="space-y-6">
    <Select
        label="Ruche"
        bind:value={localRucheId}
        options={rucheOptions}
        required
        error={errors.ruche_id}
    />

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <DatePicker
            label="Date de récolte"
            bind:value={localDateRecolte}
            required
            error={errors.date_recolte}
        />

        <Input
            label="Poids (kg)"
            bind:value={localPoidsKg}
            inputType="number"
            required
            error={errors.poids_kg}
            placeholder="0.00"
        />

        <Select
            label="Qualité"
            bind:value={localQualite}
            options={qualiteOptions}
            placeholder="Sélectionner..."
        />

        <Input
            label="Taux d'eau (%)"
            bind:value={localTauxEau}
            inputType="number"
            placeholder="15 - 20"
            hint="Taux normal: 15-20%"
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
            placeholder="Observations sur la récolte..."
        />
    </div>

    <div class="flex justify-end gap-3 pt-4 border-t border-gray-200">
        <Button on:click={handleCancel} variant="outline">Annuler</Button>
        <Button on:click={handleSubmit} {loading} variant="primary">
            {isEdit ? "Mettre à jour" : "Ajouter la récolte"}
        </Button>
    </div>
</div>
