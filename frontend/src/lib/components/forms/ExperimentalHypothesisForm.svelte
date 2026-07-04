<!-- lib/components/forms/ExperimentalHypothesisForm.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Input from "../ui/Input.svelte";
    import Select from "../ui/Select.svelte";
    import Button from "../ui/Button.svelte";

    export let formData: {
        espece: string;
        race?: string;
        parametre: string;
        valeur_estimee: number;
        unite?: string;
    };
    export let loading: boolean = false;
    export let isEdit: boolean = false;

    const dispatch = createEventDispatcher();

    let errors: Record<string, string> = {};

    let localEspece = formData.espece;
    let localRace = formData.race || "";
    let localParametre = formData.parametre;
    let localValeur = formData.valeur_estimee?.toString() || "";
    let localUnite = formData.unite || "";

    const especeOptions = [
        { value: "bovin", label: "Bovin" },
        { value: "ovin", label: "Ovin" },
        { value: "caprin", label: "Caprin" },
        { value: "avicole", label: "Avicole" },
        { value: "piscicole", label: "Piscicole" },
        { value: "apiculture", label: "Apiculture" },
        { value: "entomoculture", label: "Entomoculture" },
    ];

    const parametreOptions = [
        { value: "poids_moyen", label: "Poids moyen (kg)" },
        { value: "production_quotidienne", label: "Production quotidienne" },
        { value: "taux_croissance", label: "Taux de croissance" },
        { value: "mortalite", label: "Taux de mortalité (%)" },
        {
            value: "conversion_alimentaire",
            label: "Indice de conversion alimentaire",
        },
        { value: "age_maturite", label: "Âge de maturité (jours)" },
        { value: "production_lait", label: "Production laitière (L/jour)" },
        { value: "production_oeufs", label: "Production d'œufs (par jour)" },
        { value: "production_miel", label: "Production de miel (kg/an)" },
    ];

    function validate(): boolean {
        errors = {};
        if (!localEspece) errors.espece = "L'espèce est requise";
        if (!localParametre) errors.parametre = "Le paramètre est requis";
        if (!localValeur || parseFloat(localValeur) <= 0)
            errors.valeur = "La valeur doit être supérieure à 0";
        return Object.keys(errors).length === 0;
    }

    function handleSubmit() {
        if (validate()) {
            dispatch("submit", {
                espece: localEspece,
                race: localRace || undefined,
                parametre: localParametre,
                valeur_estimee: parseFloat(localValeur),
                unite: localUnite || undefined,
            });
        }
    }

    function handleCancel() {
        dispatch("cancel");
    }
</script>

<div class="space-y-5">
    <Select
        label="Espèce"
        bind:value={localEspece}
        options={especeOptions}
        required
        error={errors.espece}
    />

    <Input label="Race" bind:value={localRace} placeholder="Optionnel" />

    <Select
        label="Paramètre"
        bind:value={localParametre}
        options={parametreOptions}
        required
        error={errors.parametre}
    />

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Input
            label="Valeur estimée"
            bind:value={localValeur}
            inputType="number"
            required
            error={errors.valeur}
            placeholder="0.00"
        />

        <Input
            label="Unité"
            bind:value={localUnite}
            placeholder="Ex: kg, L/jour, %..."
        />
    </div>

    <div class="bg-blue-50 rounded-lg p-4 border border-blue-200">
        <div class="flex items-start gap-3">
            <svg
                class="w-5 h-5 text-blue-600 mt-0.5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
            >
                <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
            </svg>
            <div class="text-sm text-blue-800">
                <p class="font-medium">Mode expérimental</p>
                <p class="mt-1">
                    Cette hypothèse sera utilisée pour les prédictions en mode
                    expérimental. Elle devra être validée par un administrateur.
                </p>
            </div>
        </div>
    </div>

    <div class="flex justify-end gap-3 pt-4 border-t border-gray-200">
        <Button on:click={handleCancel} variant="outline">Annuler</Button>
        <Button on:click={handleSubmit} {loading} variant="primary">
            {isEdit ? "Mettre à jour" : "Créer l'hypothèse"}
        </Button>
    </div>
</div>
