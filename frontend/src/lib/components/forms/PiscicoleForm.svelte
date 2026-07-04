<!-- lib/components/forms/PiscicoleForm.svelte -->
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
        production_reproduction: boolean;
        taille_moyenne?: number;
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
    let localProductionReproduction = formData.production_reproduction;
    let localTailleMoyenne = formData.taille_moyenne?.toString() || "";
    let localNotes = formData.notes || "";

    const sexeOptions = [
        { value: "male", label: "Mâle" },
        { value: "femelle", label: "Femelle" },
        { value: "hermaphrodite", label: "Hermaphrodite" },
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
        if (!localRace) errors.race = "L'espèce est requise";
        if (!localSexe) errors.sexe = "Le sexe est requis";
        if (!localDateArrivee)
            errors.date_arrivee = "La date d'arrivée est requise";
        if (!localEnclosId) errors.enclos_id = "Le bassin est requis";
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
                production_reproduction: localProductionReproduction,
                taille_moyenne: localTailleMoyenne
                    ? parseFloat(localTailleMoyenne)
                    : undefined,
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
            placeholder="Ex: POIS-001"
        />
        <Input
            label="Espèce"
            bind:value={localRace}
            required
            error={errors.race}
            placeholder="Ex: Truite, Carpe, Tilapia..."
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
            placeholder="Ex: Écloserie..."
        />
        <Input
            label="Prix d'achat (€)"
            bind:value={localPrixAchat}
            inputType="number"
            placeholder="0"
        />
        <Select
            label="Bassin"
            bind:value={localEnclosId}
            options={enclosOptions}
            required
            error={errors.enclos_id}
        />
        <Input
            label="Taille moyenne (cm)"
            bind:value={localTailleMoyenne}
            inputType="number"
            placeholder="0"
        />
    </div>

    <div class="border-t border-gray-200 pt-4">
        <h4 class="text-md font-medium text-gray-900 mb-3">
            Types de production
        </h4>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                    bind:checked={localProductionReproduction}
                    class="rounded border-gray-300 text-primary-600"
                />
                <span class="text-sm text-gray-700">Reproduction</span>
            </label>
        </div>
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
            >{isEdit ? "Mettre à jour" : "Créer"}</Button
        >
    </div>
</div>
