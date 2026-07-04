<!-- lib/components/forms/AvicoleForm.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Input from "../ui/Input.svelte";
    import Select from "../ui/Select.svelte";
    import DatePicker from "../ui/DatePicker.svelte";
    import Button from "../ui/Button.svelte";

    export let formData: {
        identification: string;
        race: string;
        sexe: string;
        date_naissance?: string;
        date_arrivee: string;
        provenance?: string;
        prix_achat?: number;
        enclos_id: number;
        statut: string;
        production_viande: boolean;
        production_ponte: boolean;
        production_reproduction: boolean;
        notes?: string;
    };
    export let enclosOptions: Array<{ value: number; label: string }> = [];
    export let loading: boolean = false;
    export let isEdit: boolean = false;

    const dispatch = createEventDispatcher();

    let errors: Record<string, string> = {};

    let localIdentification = formData.identification;
    let localRace = formData.race;
    let localSexe = formData.sexe;
    let localDateNaissance = formData.date_naissance || "";
    let localDateArrivee = formData.date_arrivee;
    let localProvenance = formData.provenance || "";
    let localPrixAchat = formData.prix_achat?.toString() || "";
    let localEnclosId = formData.enclos_id;
    let localStatut = formData.statut;
    let localProductionViande = formData.production_viande;
    let localProductionPonte = formData.production_ponte;
    let localProductionReproduction = formData.production_reproduction;
    let localNotes = formData.notes || "";

    const sexeOptions = [
        { value: "male", label: "Mâle (Coq)" },
        { value: "femelle", label: "Femelle (Poule)" },
    ];

    const statutOptions = [
        { value: "vivant", label: "Vivant" },
        { value: "vendu", label: "Vendu" },
        { value: "decede", label: "Décédé" },
        { value: "transfere", label: "Transféré" },
    ];

    function validate(): boolean {
        errors = {};
        if (!localIdentification)
            errors.identification = "L'identification est requise";
        if (!localRace) errors.race = "La race est requise";
        if (!localSexe) errors.sexe = "Le sexe est requis";
        if (!localDateArrivee)
            errors.date_arrivee = "La date d'arrivée est requise";
        if (!localEnclosId) errors.enclos_id = "L'enclos est requis";
        return Object.keys(errors).length === 0;
    }

    function handleSubmit() {
        if (validate()) {
            dispatch("submit", {
                identification: localIdentification,
                race: localRace,
                sexe: localSexe,
                date_naissance: localDateNaissance || undefined,
                date_arrivee: localDateArrivee,
                provenance: localProvenance || undefined,
                prix_achat: localPrixAchat
                    ? parseFloat(localPrixAchat)
                    : undefined,
                enclos_id: localEnclosId,
                statut: localStatut,
                production_viande: localProductionViande,
                production_ponte: localProductionPonte,
                production_reproduction: localProductionReproduction,
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
            label="Identification"
            bind:value={localIdentification}
            required
            error={errors.identification}
            placeholder="Ex: POU-001"
        />
        <Input
            label="Race"
            bind:value={localRace}
            required
            error={errors.race}
            placeholder="Ex: Leghorn, Sussex..."
        />
        <Select
            label="Sexe"
            bind:value={localSexe}
            options={sexeOptions}
            required
            error={errors.sexe}
        />
        <Select
            label="Statut"
            bind:value={localStatut}
            options={statutOptions}
            required
        />
        <DatePicker label="Date de naissance" bind:value={localDateNaissance} />
        <DatePicker
            label="Date d'arrivée"
            bind:value={localDateArrivee}
            required
            error={errors.date_arrivee}
        />
        <Input
            label="Provenance"
            bind:value={localProvenance}
            placeholder="Ex: Couvoir Dupont..."
        />
        <Input
            label="Prix d'achat (€)"
            bind:value={localPrixAchat}
            inputType="number"
            placeholder="0"
        />
        <Select
            label="Enclos"
            bind:value={localEnclosId}
            options={enclosOptions}
            required
            error={errors.enclos_id}
        />
    </div>

    <div class="border-t border-gray-200 pt-4">
        <h4 class="text-md font-medium text-gray-900 mb-3">
            Types de production
        </h4>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <label class="flex items-center gap-2">
                <input
                    type="checkbox"
                    bind:checked={localProductionViande}
                    class="rounded border-gray-300 text-primary-600"
                />
                <span class="text-sm text-gray-700">Production viande</span>
            </label>
            <label class="flex items-center gap-2">
                <input
                    type="checkbox"
                    bind:checked={localProductionPonte}
                    class="rounded border-gray-300 text-primary-600"
                />
                <span class="text-sm text-gray-700">Production œufs</span>
            </label>
            <label class="flex items-center gap-2">
                <input
                    type="checkbox"
                    bind:checked={localProductionReproduction}
                    class="rounded border-gray-300 text-primary-600"
                />
                <span class="text-sm text-gray-700">Reproduction</span>
            </label>
        </div>
    </div>

    {#if localProductionPonte}
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
                    <p class="font-medium">Production d'œufs</p>
                    <p class="mt-1">
                        Vous pourrez enregistrer la production d'œufs
                        quotidienne depuis la fiche de l'animal.
                    </p>
                </div>
            </div>
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
            placeholder="Informations supplémentaires..."
        />
    </div>

    <div class="flex justify-end gap-3 pt-4 border-t border-gray-200">
        <Button on:click={handleCancel} variant="outline">Annuler</Button>
        <Button on:click={handleSubmit} {loading} variant="primary"
            >{isEdit ? "Mettre à jour" : "Créer"}</Button
        >
    </div>
</div>
