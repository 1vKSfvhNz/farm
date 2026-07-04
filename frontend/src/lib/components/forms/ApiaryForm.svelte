<!-- lib/components/forms/ApicultureForm.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Input from "../ui/Input.svelte";
    import Select from "../ui/Select.svelte";
    import DatePicker from "../ui/DatePicker.svelte";
    import Button from "../ui/Button.svelte";

    export let formData: {
        identification: string;
        emplacement?: string;
        date_installation: string;
        race?: string;
        statut: string;
        nombre_cadres?: number;
        notes?: string;
    };
    export let loading: boolean = false;
    export let isEdit: boolean = false;

    const dispatch = createEventDispatcher();

    let errors: Record<string, string> = {};

    let localIdentification = formData.identification;
    let localEmplacement = formData.emplacement || "";
    let localDateInstallation = formData.date_installation;
    let localRace = formData.race || "";
    let localStatut = formData.statut;
    let localNombreCadres = formData.nombre_cadres?.toString() || "";
    let localNotes = formData.notes || "";

    const statutOptions = [
        { value: "active", label: "Active" },
        { value: "orpheline", label: "Orpheline" },
        { value: "en_essaimage", label: "En essaimage" },
        { value: "morte", label: "Morte" },
    ];

    function validate(): boolean {
        errors = {};
        if (!localIdentification)
            errors.identification = "L'identification est requise";
        if (!localDateInstallation)
            errors.date_installation = "La date d'installation est requise";
        if (!localStatut) errors.statut = "Le statut est requis";
        return Object.keys(errors).length === 0;
    }

    function handleSubmit() {
        if (validate()) {
            dispatch("submit", {
                identification: localIdentification,
                emplacement: localEmplacement || undefined,
                date_installation: localDateInstallation,
                race: localRace || undefined,
                statut: localStatut,
                nombre_cadres: localNombreCadres
                    ? parseInt(localNombreCadres)
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
            placeholder="Ex: Ruche-01, R-001..."
        />
        <Input
            label="Emplacement"
            bind:value={localEmplacement}
            placeholder="Ex: Jardin nord, Rucher principal..."
        />
        <DatePicker
            label="Date d'installation"
            bind:value={localDateInstallation}
            required
            error={errors.date_installation}
        />
        <Input
            label="Race"
            bind:value={localRace}
            placeholder="Ex: Noire, Buckfast, Italienne..."
        />
        <Select
            label="Statut"
            bind:value={localStatut}
            options={statutOptions}
            required
            error={errors.statut}
        />
        <Input
            label="Nombre de cadres"
            bind:value={localNombreCadres}
            inputType="number"
            placeholder="0"
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
            placeholder="Observations sur la ruche..."
        />
    </div>

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
                <p class="font-medium">Information</p>
                <p class="mt-1">
                    Une ruche orpheline nécessite une intervention rapide pour
                    introduire une nouvelle reine.
                </p>
            </div>
        </div>
    </div>

    <div class="flex justify-end gap-3 pt-4 border-t border-gray-200">
        <Button on:click={handleCancel} variant="outline">Annuler</Button>
        <Button on:click={handleSubmit} {loading} variant="primary">
            {isEdit ? "Mettre à jour" : "Créer la ruche"}
        </Button>
    </div>
</div>
