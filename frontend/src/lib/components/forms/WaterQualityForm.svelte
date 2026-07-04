<!-- lib/components/forms/WaterQualityForm.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Input from "../ui/Input.svelte";
    import Select from "../ui/Select.svelte";
    import DateTimePicker from "../ui/DatePicker.svelte";
    import Button from "../ui/Button.svelte";

    export let formData: {
        enclos_id: number;
        timestamp: string;
        ph?: number;
        temperature?: number;
        oxygene_dissous?: number;
        ammoniac?: number;
        nitrites?: number;
        nitrates?: number;
        conductivite?: number;
        turbidite?: number;
        source?: string;
    };
    export let enclosOptions: Array<{ value: number; label: string }> = [];
    export let loading: boolean = false;
    export let isEdit: boolean = false;

    const dispatch = createEventDispatcher();

    let errors: Record<string, string> = {};

    let localEnclosId = formData.enclos_id;
    let localTimestamp = formData.timestamp;
    let localPh = formData.ph?.toString() || "";
    let localTemperature = formData.temperature?.toString() || "";
    let localOxygene = formData.oxygene_dissous?.toString() || "";
    let localAmmoniac = formData.ammoniac?.toString() || "";
    let localNitrites = formData.nitrites?.toString() || "";
    let localNitrates = formData.nitrates?.toString() || "";
    let localConductivite = formData.conductivite?.toString() || "";
    let localTurbidite = formData.turbidite?.toString() || "";
    let localSource = formData.source || "";

    const sourceOptions = [
        { value: "manuel", label: "Mesure manuelle" },
        { value: "sonde", label: "Sonde automatique" },
        { value: "laboratoire", label: "Analyse laboratoire" },
        { value: "autre", label: "Autre" },
    ];

    function validate(): boolean {
        errors = {};
        if (!localEnclosId) errors.enclos_id = "L'enclos est requis";
        if (!localTimestamp)
            errors.timestamp = "La date et l'heure sont requises";
        return Object.keys(errors).length === 0;
    }

    function handleSubmit() {
        if (validate()) {
            dispatch("submit", {
                enclos_id: localEnclosId,
                timestamp: localTimestamp,
                ph: localPh ? parseFloat(localPh) : undefined,
                temperature: localTemperature
                    ? parseFloat(localTemperature)
                    : undefined,
                oxygene_dissous: localOxygene
                    ? parseFloat(localOxygene)
                    : undefined,
                ammoniac: localAmmoniac ? parseFloat(localAmmoniac) : undefined,
                nitrites: localNitrites ? parseFloat(localNitrites) : undefined,
                nitrates: localNitrates ? parseFloat(localNitrates) : undefined,
                conductivite: localConductivite
                    ? parseFloat(localConductivite)
                    : undefined,
                turbidite: localTurbidite
                    ? parseFloat(localTurbidite)
                    : undefined,
                source: localSource || undefined,
            });
        }
    }

    function handleCancel() {
        dispatch("cancel");
    }

    // Vérifications des seuils pour affichage d'alertes
    $: phWarning =
        localPh && (parseFloat(localPh) < 6.5 || parseFloat(localPh) > 8.5);
    $: temperatureWarning =
        localTemperature &&
        (parseFloat(localTemperature) < 10 ||
            parseFloat(localTemperature) > 25);
    $: oxygeneWarning = localOxygene && parseFloat(localOxygene) < 5;
    $: ammoniacWarning = localAmmoniac && parseFloat(localAmmoniac) > 0.5;
    $: nitritesWarning = localNitrites && parseFloat(localNitrites) > 0.5;
    $: nitratesWarning = localNitrates && parseFloat(localNitrates) > 50;
</script>

<div class="space-y-6">
    <!-- En-tête avec rappel des valeurs normales -->
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
                <p class="font-medium">Valeurs de référence</p>

                <p class="mt-1">
                    pH: 6.5-8.5 | Température: 10-25°C | Oxygène: &gt;5 mg/L |
                    Ammoniac: &lt;0.5 mg/L | Nitrites: &lt;0.5 mg/L | Nitrates:
                    &lt;50 mg/L
                </p>
            </div>
        </div>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Select
            label="Enclos / Bassin"
            bind:value={localEnclosId}
            options={enclosOptions}
            required
            error={errors.enclos_id}
        />

        <DateTimePicker
            label="Date et heure"
            bind:value={localTimestamp}
            required
            error={errors.timestamp}
        />

        <Select
            label="Source de la mesure"
            bind:value={localSource}
            options={sourceOptions}
            placeholder="Sélectionner..."
        />
    </div>

    <div class="border-t border-gray-200 pt-4">
        <h4 class="text-md font-medium text-gray-900 mb-3">
            Paramètres physico-chimiques
        </h4>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <!-- pH -->
            <div class="relative">
                <Input
                    label="pH"
                    bind:value={localPh}
                    inputType="number"
                    step="0.1"
                    placeholder="6.5 - 8.5"
                    error={phWarning ? "Valeur hors norme (6.5-8.5)" : ""}
                />
                {#if phWarning}
                    <div class="absolute right-2 top-8">
                        <span
                            class="w-2 h-2 bg-red-500 rounded-full inline-block"
                        ></span>
                    </div>
                {/if}
            </div>

            <!-- Température -->
            <div class="relative">
                <Input
                    label="Température (°C)"
                    bind:value={localTemperature}
                    inputType="number"
                    step="0.1"
                    placeholder="10 - 25"
                    error={temperatureWarning
                        ? "Valeur hors norme (10-25°C)"
                        : ""}
                />
                {#if temperatureWarning}
                    <div class="absolute right-2 top-8">
                        <span
                            class="w-2 h-2 bg-red-500 rounded-full inline-block"
                        ></span>
                    </div>
                {/if}
            </div>

            <!-- Oxygène dissous -->
            <div class="relative">
                <Input
                    label="Oxygène dissous (mg/L)"
                    bind:value={localOxygene}
                    inputType="number"
                    step="0.1"
                    placeholder="> 5"
                    error={oxygeneWarning ? "Valeur basse (< 5 mg/L)" : ""}
                />
                {#if oxygeneWarning}
                    <div class="absolute right-2 top-8">
                        <span
                            class="w-2 h-2 bg-red-500 rounded-full inline-block"
                        ></span>
                    </div>
                {/if}
            </div>

            <!-- Ammoniac -->
            <div class="relative">
                <Input
                    label="Ammoniac (mg/L)"
                    bind:value={localAmmoniac}
                    inputType="number"
                    step="0.01"
                    placeholder="< 0.5"
                    error={ammoniacWarning ? "Valeur élevée (> 0.5 mg/L)" : ""}
                />
                {#if ammoniacWarning}
                    <div class="absolute right-2 top-8">
                        <span
                            class="w-2 h-2 bg-red-500 rounded-full inline-block"
                        ></span>
                    </div>
                {/if}
            </div>

            <!-- Nitrites -->
            <div class="relative">
                <Input
                    label="Nitrites (mg/L)"
                    bind:value={localNitrites}
                    inputType="number"
                    step="0.01"
                    placeholder="< 0.5"
                    error={nitritesWarning ? "Valeur élevée (> 0.5 mg/L)" : ""}
                />
                {#if nitritesWarning}
                    <div class="absolute right-2 top-8">
                        <span
                            class="w-2 h-2 bg-red-500 rounded-full inline-block"
                        ></span>
                    </div>
                {/if}
            </div>

            <!-- Nitrates -->
            <div class="relative">
                <Input
                    label="Nitrates (mg/L)"
                    bind:value={localNitrates}
                    inputType="number"
                    step="1"
                    placeholder="< 50"
                    error={nitratesWarning ? "Valeur élevée (> 50 mg/L)" : ""}
                />
                {#if nitratesWarning}
                    <div class="absolute right-2 top-8">
                        <span
                            class="w-2 h-2 bg-red-500 rounded-full inline-block"
                        ></span>
                    </div>
                {/if}
            </div>

            <!-- Conductivité -->
            <Input
                label="Conductivité (µS/cm)"
                bind:value={localConductivite}
                inputType="number"
                placeholder="0 - 2000"
            />

            <!-- Turbidité -->
            <Input
                label="Turbidité (NTU)"
                bind:value={localTurbidite}
                inputType="number"
                placeholder="0 - 50"
            />
        </div>
    </div>

    <!-- Résumé des alertes -->
    {#if phWarning || temperatureWarning || oxygeneWarning || ammoniacWarning || nitritesWarning || nitratesWarning}
        <div class="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
            <div class="flex items-start gap-3">
                <svg
                    class="w-5 h-5 text-yellow-600 mt-0.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                    />
                </svg>
                <div class="text-sm text-yellow-800">
                    <p class="font-medium">
                        Attention : paramètres hors normes détectés
                    </p>
                    <ul class="mt-1 list-disc list-inside">
                        {#if phWarning}<li>pH hors norme (6.5-8.5)</li>{/if}
                        {#if temperatureWarning}<li>
                                Température hors norme (10-25°C)
                            </li>{/if}
                        {#if oxygeneWarning}<li>
                                Oxygène dissous bas (&lt;5 mg/L)
                            </li>{/if}
                        {#if ammoniacWarning}<li>
                                Ammoniac élevé (&gt;0.5 mg/L)
                            </li>{/if}
                        {#if nitritesWarning}<li>
                                Nitrites élevés (&gt;0.5 mg/L)
                            </li>{/if}
                        {#if nitratesWarning}<li>
                                Nitrates élevés (&gt;50 mg/L)
                            </li>{/if}
                    </ul>
                </div>
            </div>
        </div>
    {/if}

    <div class="flex justify-end gap-3 pt-4 border-t border-gray-200">
        <Button on:click={handleCancel} variant="outline">Annuler</Button>
        <Button on:click={handleSubmit} {loading} variant="primary">
            {isEdit ? "Mettre à jour" : "Ajouter la mesure"}
        </Button>
    </div>
</div>
