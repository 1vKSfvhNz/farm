<!-- lib/components/forms/CaprinForm.svelte -->
<!-- svelte-ignore a11y-label-has-associated-control -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import { onMount } from "svelte";
    import Input from "../ui/Input.svelte";
    import Select from "../ui/Select.svelte";
    import DatePicker from "../ui/DatePicker.svelte";
    import Button from "../ui/Button.svelte";
    import type { EnclosResponse, PaginatedResponse } from "$lib/types";
    import { enclosApi } from "$lib/api";
    import { ENCLOS_TYPES } from "$lib/utils/constants";
    import { fileToBase64, getFullPhotoUrl } from "$lib/utils/media";
    
    export let formData: {
        type_espece: string;
        race: string;
        sexe: string;
        date_naissance: string | null;
        date_arrivee: string;
        provenance: string;
        prix_achat?: number;
        enclos_id: number;
        statut: string;
        production_viande: boolean;
        production_reproduction: boolean;
        notes?: string;
        photo_url?: string | null;
    };
    export let loading: boolean = false;
    export let isEdit: boolean = false;

    const dispatch = createEventDispatcher();

    let errors: Record<string, string> = {};
    let enclosOptions: Array<{ value: number; label: string }> = [];
    let enclosLoading = false;
    let enclosLoaded = false;
    
    // Gestion de la photo
    let photoPreview: string | null = null;
    let photoFile: File | null = null;
    let photoBase64: string | null = null;
    let poidsInitial: string = "";

    let localRace = formData.race;
    let localSexe = formData.sexe;
    let localDateNaissance = formData.date_naissance || "";
    let localDateArrivee = formData.date_arrivee;
    let localProvenance = formData.provenance || "";
    let localPrixAchat = formData.prix_achat?.toString() || "";
    let localEnclosId = formData.enclos_id || "";
    let localStatut = formData.statut;
    let localProductionViande = formData.production_viande;
    let localProductionReproduction = formData.production_reproduction;
    let localNotes = formData.notes || "";

    const sexeOptions = [
        { value: "male", label: "Mâle (Bouc)" },
        { value: "femelle", label: "Femelle (Chèvre)" },
        { value: "hermaphrodite", label: "Hermaphrodite" },
    ];

    const statutOptions = [
        { value: "vivant", label: "Vivant" },
        { value: "decede", label: "Décédé" },
        { value: "transfere", label: "Transféré" },
    ];

    // Initialiser la prévisualisation si une photo existe déjà
    onMount(() => {
        if (formData.photo_url) {
            photoPreview = getFullPhotoUrl(formData.photo_url);
        }
        loadEnclos();
    });

    async function loadEnclos() {
        if (enclosLoaded) return;
        enclosLoading = true;
        try {
            const response: PaginatedResponse<EnclosResponse> = await enclosApi.getEnclos({ 
                limit: 200, 
                enclos_type: [ENCLOS_TYPES.ENCLOS, ENCLOS_TYPES.PATURAGE] 
            });
            enclosOptions = response.items.map(enclos => ({
                value: enclos.id,
                label: `${enclos.name} (${enclos.type} - ${enclos.surface}m²)`
            }));
            enclosLoaded = true;
        } catch (error) {
            console.error("Erreur lors du chargement des enclos:", error);
            errors.enclos_id = "Impossible de charger la liste des enclos";
        } finally {
            enclosLoading = false;
        }
    }

    async function handlePhotoChange(event: Event) {
        const input = event.target as HTMLInputElement;
        if (input.files && input.files[0]) {
            photoFile = input.files[0];
            
            // Prévisualisation
            photoPreview = URL.createObjectURL(photoFile);
            
            // Convertir en base64 pour l'envoi
            photoBase64 = await fileToBase64(photoFile);
        }
    }

    function removePhoto() {
        if (photoPreview && !formData.photo_url) {
            URL.revokeObjectURL(photoPreview);
        }
        photoPreview = null;
        photoFile = null;
        photoBase64 = null;
    }

    function validate(): boolean {
        errors = {};
        if (!localRace) errors.race = "La race est requise";
        if (!localSexe) errors.sexe = "Le sexe est requis";
        if (!localDateArrivee) errors.date_arrivee = "La date d'arrivée est requise";
        if (!localEnclosId) errors.enclos_id = "L'enclos est requis";
        return Object.keys(errors).length === 0;
    }

    function handleSubmit(event: Event) {
        event.preventDefault();
        event.stopPropagation();
        
        if (validate()) {
            // Vérification : si le statut est "décédé", afficher un avertissement
            if (localStatut === "decede" && isEdit) {
                const confirmSubmit = confirm(
                    "⚠️ ATTENTION : Ce caprin sera marqué comme DÉCÉDÉ.\n\n" +
                    "Une fois ce statut enregistré, AUCUNE modification ne sera plus possible sur cet animal.\n\n" +
                    "Voulez-vous vraiment continuer ?"
                );
                if (!confirmSubmit) {
                    return;
                }
            }

            // Construction des données pour le serveur
            const submitData: Record<string, any> = {
                type_espece: "caprin",
                race: localRace,
                sexe: localSexe,
                date_naissance: localDateNaissance || null,
                date_arrivee: localDateArrivee,
                provenance: localProvenance || null,
                prix_achat: localPrixAchat ? parseFloat(localPrixAchat) : null,
                enclos_id: localEnclosId,
                statut: localStatut,
                production_viande: localProductionViande,
                production_reproduction: localProductionReproduction,
                notes: localNotes || null,
                // Poids initial uniquement pour la création
                ...(poidsInitial && !isEdit ? { poids_initial: parseFloat(poidsInitial) } : {}),
                // Photo en base64 (si une nouvelle photo a été sélectionnée)
                ...(photoBase64 ? { photo_base64: photoBase64 } : {})
            };

            // Nettoyer les champs avec des valeurs null/undefined pour éviter les erreurs
            Object.keys(submitData).forEach(key => {
                if (submitData[key] === null || submitData[key] === undefined) {
                    delete submitData[key];
                }
            });
            
            console.log("Envoi des données au serveur:", submitData);
            dispatch("submit", submitData);
        }
    }

    function handleCancel() {
        dispatch("cancel");
    }
</script>

<form on:submit={handleSubmit} class="space-y-6">
    <!-- Section Photo -->
    <div class="border rounded-lg p-4 bg-gray-50">
        <h4 class="text-md font-medium text-gray-900 mb-3">📷 Photo du caprin</h4>
        <div class="flex items-center gap-4">
            {#if photoPreview}
                <div class="relative">
                    <img 
                        src={photoPreview} 
                        alt="Preview" 
                        class="w-32 h-32 object-cover rounded-lg border-2 border-gray-300" 
                    />
                    <button
                        type="button"
                        on:click={removePhoto}
                        class="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs hover:bg-red-600"
                    >
                        ✕
                    </button>
                </div>
            {:else}
                <div class="w-32 h-32 bg-gray-200 rounded-lg border-2 border-dashed border-gray-400 flex items-center justify-center text-gray-500">
                    <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                </div>
            {/if}
            <div>
                <input
                    type="file"
                    accept="image/*"
                    on:change={handlePhotoChange}
                    class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100"
                />
                <p class="text-xs text-gray-500 mt-1">Formats acceptés: JPG, PNG, GIF. Poids max: 5MB</p>
                {#if formData.photo_url && photoPreview}
                    <p class="text-xs text-green-600 mt-1">✅ Photo existante conservée</p>
                {/if}
            </div>
        </div>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Input
            label="Race"
            bind:value={localRace}
            required
            error={errors.race}
            placeholder="Ex: Alpine, Saanen, Boer..."
        />
        <Select
            label="Sexe"
            bind:value={localSexe}
            options={sexeOptions}
            required
            error={errors.sexe}
        />
        <!-- Sélecteur Statut avec bandeau d'avertissement -->
        <div class="md:col-span-2">
            <Select
                label="Statut"
                bind:value={localStatut}
                options={statutOptions}
                required
            />
            
            <!-- Bandeau d'avertissement pour le statut "Décédé" -->
            {#if localStatut === "decede" && isEdit}
                <div class="mt-3 p-4 bg-red-50 border-l-4 border-red-500 rounded-r-lg">
                    <div class="flex items-start gap-3">
                        <div class="flex-shrink-0">
                            <svg class="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                        </div>
                        <div>
                            <h5 class="text-sm font-semibold text-red-800">
                                ⚠️ Action irréversible
                            </h5>
                            <div class="mt-1 text-sm text-red-700">
                                <p class="font-medium">
                                    Si cet animal est marqué comme <span class="font-bold">DÉCÉDÉ</span> :
                                </p>
                                <ul class="mt-2 list-disc list-inside space-y-1">
                                    <li>Aucune modification ne sera plus possible après validation</li>
                                    <li>Le statut sera définitivement verrouillé</li>
                                    <li>Vous ne pourrez plus mettre à jour cet animal</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            {/if}            
        </div>
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
            placeholder="Ex: Élevage Dupont..."
        />
        <Input
            label="Prix d'achat (FCFA)"
            bind:value={localPrixAchat}
            inputType="number"
            placeholder="0"
        />
        
        <!-- Poids initial (seulement à la création) -->
        {#if !isEdit}
            <Input
                label="Poids initial (kg)"
                bind:value={poidsInitial}
                inputType="number"
                placeholder="0"
                hint="Poids du caprin à l'arrivée"
            />
        {/if}
        
        <div class="md:col-span-2">
            {#if enclosLoading}
                <div class="bg-gray-50 rounded-lg p-4">
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        Enclos
                    </label>
                    <div class="text-sm text-gray-500 flex items-center gap-2">
                        <span class="inline-block w-4 h-4 border-2 border-gray-300 border-t-green-600 rounded-full animate-spin"></span>
                        Chargement des enclos...
                    </div>
                </div>
            {:else}
                <Select
                    label="Enclos"
                    bind:value={localEnclosId}
                    options={enclosOptions}
                    required
                    error={errors.enclos_id}
                    placeholder="Sélectionner un enclos"
                />
            {/if}
        </div>
    </div>

    <!-- Types de production -->
    <div class="border-t border-gray-200 pt-4">
        <h4 class="text-md font-medium text-gray-900 mb-3">Types de production</h4>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <label class="flex items-center gap-2">
                <input type="checkbox" bind:checked={localProductionViande} class="rounded" />
                <span class="text-sm text-gray-700">🥩 Production viande</span>
            </label>
            <label class="flex items-center gap-2">
                <input type="checkbox" bind:checked={localProductionReproduction} class="rounded" />
                <span class="text-sm text-gray-700">🐣 Reproduction</span>
            </label>
        </div>
    </div>

    <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">Notes</label>
        <textarea
            bind:value={localNotes}
            rows={3}
            class="w-full rounded-lg border border-gray-300 focus:border-green-500 focus:ring-2 focus:ring-green-100 px-4 py-2"
            placeholder="Informations supplémentaires..."
        />
    </div>

    <div class="flex justify-end gap-3 pt-4 border-t border-gray-200">
        <Button on:click={handleCancel} variant="outline">Annuler</Button>
        <Button type="submit" on:click={handleSubmit} {loading} variant="primary">
            {isEdit ? "Mettre à jour" : "Créer"}
        </Button>
    </div>
</form>