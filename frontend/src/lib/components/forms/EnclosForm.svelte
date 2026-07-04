<!-- src/lib/components/forms/EnclosForm.svelte -->
<!-- svelte-ignore a11y-label-has-associated-control -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Input from "../ui/Input.svelte";
    import Select from "../ui/Select.svelte";
    import Button from "../ui/Button.svelte";

    export let formData: {
        name: string;
        type: string;
        longueur: number;
        largeur: number;
        hauteur?: number;
        zone?: string;
        localisation_gps?: string;
        description?: string;
    };
    export let loading: boolean = false;
    export let isEdit: boolean = false;

    const dispatch = createEventDispatcher();

    let errors: Record<string, string> = {};

    let localName = formData.name;
    let localType = formData.type;
    let localLongueur = formData.longueur?.toString() || "";
    let localLargeur = formData.largeur?.toString() || "";
    let localHauteur = formData.hauteur?.toString() || "";
    let localZone = formData.zone || "";
    let localLocalisationGps = formData.localisation_gps || "";
    let localDescription = formData.description || "";

    const typeOptions = [
        { value: "enclos", label: "Enclos" },
        { value: "bassin", label: "Bassin" },
        { value: "pâturage", label: "Pâturage" },
        { value: "cage", label: "Cage" },
        { value: "bac", label: "Bac" },
    ];

    // Convertit une chaîne en nombre flottant (accepte virgules et points)
    function parseFloatFromInput(value: string): number {
        if (!value || value.trim() === "") return NaN;
        // Remplacer la virgule par un point
        const normalized = value.replace(',', '.');
        return parseFloat(normalized);
    }

    // Valide et nettoie l'input d'un nombre flottant
    function handleFloatInput(event: Event, setter: (value: string) => void): void {
        const input = event.target as HTMLInputElement;
        let value = input.value;
        
        // Permet seulement les chiffres, un point, une virgule, et le signe moins
        value = value.replace(/[^\d,\-.]/g, '');
        
        // Évite plusieurs points ou virgules
        const dotCount = (value.match(/\./g) || []).length;
        const commaCount = (value.match(/,/g) || []).length;
        if (dotCount > 1 || commaCount > 1) return;
        
        // Si c'est une virgule, on peut la garder pour l'affichage
        setter(value);
    }

    // Validation du format GPS (latitude, longitude)
    function validateGps(gps: string): boolean {
        if (!gps || gps.trim() === "") return true;
        
        const pattern = /^-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+$/;
        if (!pattern.test(gps.trim())) {
            return false;
        }
        
        const [lat, lon] = gps.split(',').map(coord => parseFloat(coord.trim()));
        if (isNaN(lat) || isNaN(lon)) return false;
        if (lat < -90 || lat > 90) return false;
        if (lon < -180 || lon > 180) return false;
        
        return true;
    }

    function validate(): boolean {
        errors = {};
        
        if (!localName || localName.trim() === "") 
            errors.name = "Le nom est requis";
        
        if (!localType) 
            errors.type = "Le type est requis";
        
        // Validation des nombres flottants
        const longueurNum = parseFloatFromInput(localLongueur);
        if (!localLongueur || isNaN(longueurNum) || longueurNum <= 0)
            errors.longueur = "La longueur doit être un nombre supérieur à 0 (ex: 10.5 ou 10,5)";
        
        const largeurNum = parseFloatFromInput(localLargeur);
        if (!localLargeur || isNaN(largeurNum) || largeurNum <= 0)
            errors.largeur = "La largeur doit être un nombre supérieur à 0 (ex: 5.25 ou 5,25)";
        
        if (localHauteur && localHauteur.trim() !== "") {
            const hauteurNum = parseFloatFromInput(localHauteur);
            if (isNaN(hauteurNum) || hauteurNum <= 0) {
                errors.hauteur = "La hauteur doit être un nombre positif (ex: 1.50 ou 1,50)";
            }
        }
        
        if (localLocalisationGps && localLocalisationGps.trim() !== "") {
            if (!validateGps(localLocalisationGps)) {
                errors.localisation_gps = "Format GPS invalide. Utilisez le format: latitude, longitude (ex: 48.8566, 2.3522)";
            }
        }
        
        return Object.keys(errors).length === 0;
    }

    function handleSubmit() {
        if (validate()) {
            let hauteurValue = undefined;
            if (localHauteur && localHauteur.trim() !== "") {
                hauteurValue = parseFloatFromInput(localHauteur);
            }
            
            dispatch("submit", {
                name: localName.trim(),
                type: localType,
                longueur: parseFloatFromInput(localLongueur),
                largeur: parseFloatFromInput(localLargeur),
                hauteur: hauteurValue,
                zone: localZone?.trim() || undefined,
                localisation_gps: localLocalisationGps?.trim() || undefined,
                description: localDescription?.trim() || undefined,
            });
        }
    }

    function handleCancel() {
        dispatch("cancel");
    }

    async function getCurrentLocation() {
        if (!navigator.geolocation) {
            errors.localisation_gps = "La géolocalisation n'est pas supportée par votre navigateur";
            return;
        }

        try {
            const position = await new Promise<GeolocationPosition>((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject);
            });
            
            const { latitude, longitude } = position.coords;
            localLocalisationGps = `${latitude.toFixed(6)}, ${longitude.toFixed(6)}`;
            
            if (errors.localisation_gps) {
                delete errors.localisation_gps;
            }
        } catch (error) {
            console.error("Erreur de géolocalisation:", error);
            errors.localisation_gps = "Impossible d'obtenir votre position. Vérifiez les permissions.";
        }
    }

    // Calculs automatiques
    $: {
        const longueurNum = parseFloatFromInput(localLongueur);
        const largeurNum = parseFloatFromInput(localLargeur);
        surface = !isNaN(longueurNum) && !isNaN(largeurNum) && longueurNum > 0 && largeurNum > 0
            ? (longueurNum * largeurNum).toFixed(2)
            : "0";
    }
    
    $: {
        if (localHauteur && localHauteur.trim() !== "" && localLongueur && localLargeur) {
            const hauteurNum = parseFloatFromInput(localHauteur);
            const longueurNum = parseFloatFromInput(localLongueur);
            const largeurNum = parseFloatFromInput(localLargeur);
            if (!isNaN(hauteurNum) && hauteurNum > 0 && !isNaN(longueurNum) && !isNaN(largeurNum)) {
                volume = (longueurNum * largeurNum * hauteurNum).toFixed(2);
            } else {
                volume = null;
            }
        } else {
            volume = null;
        }
    }
    
    let surface = "0";
    let volume: string | null = null;
</script>

<div class="space-y-5">
    <!-- Message d'information -->
    <div class="bg-blue-50 rounded-lg p-3 text-sm text-blue-800 border border-blue-200">
        <div class="flex items-start gap-2">
            <span class="text-lg">ℹ️</span>
            <div>
                <p class="font-medium">Dimensions et calculs automatiques</p>
                <p class="text-xs mt-1">La surface et le volume sont calculés automatiquement. Vous pouvez utiliser le point (.) ou la virgule (,) pour les décimaux.</p>
            </div>
        </div>
    </div>

    <!-- Formulaire principal -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Input
            label="Nom"
            bind:value={localName}
            required
            error={errors.name}
            placeholder="Ex: Enclos Nord, Bassin Principal..."
        />

        <Select
            label="Type"
            bind:value={localType}
            options={typeOptions}
            required
            error={errors.type}
        />

        <!-- Champ Longueur en texte pour gérer les virgules -->
        <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">
                Longueur (m) <span class="text-red-500">*</span>
            </label>
            <input
                type="text"
                bind:value={localLongueur}
                on:input={(e) => handleFloatInput(e, (val) => localLongueur = val)}
                class="w-full rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-2 focus:ring-primary-100 px-4 py-2 transition-all"
                placeholder="Ex: 10.5 ou 10,5"
            />
            {#if errors.longueur}
                <p class="text-red-500 text-xs mt-1">{errors.longueur}</p>
            {/if}
        </div>

        <!-- Champ Largeur en texte pour gérer les virgules -->
        <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">
                Largeur (m) <span class="text-red-500">*</span>
            </label>
            <input
                type="text"
                bind:value={localLargeur}
                on:input={(e) => handleFloatInput(e, (val) => localLargeur = val)}
                class="w-full rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-2 focus:ring-primary-100 px-4 py-2 transition-all"
                placeholder="Ex: 5.25 ou 5,25"
            />
            {#if errors.largeur}
                <p class="text-red-500 text-xs mt-1">{errors.largeur}</p>
            {/if}
        </div>

        <!-- Champ Hauteur en texte pour gérer les virgules -->
        <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">
                Hauteur (m) <span class="text-gray-400 text-xs">(optionnel)</span>
            </label>
            <input
                type="text"
                bind:value={localHauteur}
                on:input={(e) => handleFloatInput(e, (val) => localHauteur = val)}
                class="w-full rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-2 focus:ring-primary-100 px-4 py-2 transition-all"
                placeholder="Ex: 2.5 ou 2,5"
            />
            {#if errors.hauteur}
                <p class="text-red-500 text-xs mt-1">{errors.hauteur}</p>
            {/if}
        </div>

        <Input
            label="Zone"
            bind:value={localZone}
            placeholder="Ex: Zone A, Parc Nord..."
        />

        <div class="md:col-span-2">
            <div class="space-y-2">
                <div class="flex justify-between items-center">
                    <label class="block text-sm font-medium text-gray-700">
                        Localisation GPS
                    </label>
                    <button
                        type="button"
                        on:click={getCurrentLocation}
                        class="text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 px-2 py-1 rounded transition-colors flex items-center gap-1"
                    >
                        📍 Utiliser ma position
                    </button>
                </div>
                <Input
                    bind:value={localLocalisationGps}
                    error={errors.localisation_gps}
                    placeholder="Ex: 48.8566, 2.3522"
                />
                <p class="text-xs text-gray-500">
                    Format: latitude, longitude (ex: 48.8566, 2.3522)
                </p>
            </div>
        </div>
    </div>

    <!-- Calculs automatiques -->
    <div class="bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg p-4 border border-gray-200">
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div class="flex justify-between items-center">
                <div class="flex items-center gap-2">
                    <span class="text-lg">📐</span>
                    <span class="text-sm text-gray-600">Surface:</span>
                </div>
                <span class="text-lg font-semibold text-gray-800">{surface} m²</span>
            </div>
            
            {#if volume}
                <div class="flex justify-between items-center">
                    <div class="flex items-center gap-2">
                        <span class="text-lg">📦</span>
                        <span class="text-sm text-gray-600">Volume:</span>
                    </div>
                    <span class="text-lg font-semibold text-gray-800">{volume} m³</span>
                </div>
            {/if}
        </div>
    </div>

    <!-- Description -->
    <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">
            Description
        </label>
        <textarea
            bind:value={localDescription}
            rows={3}
            class="w-full rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-2 focus:ring-primary-100 px-4 py-2 transition-all resize-none"
            placeholder="Informations supplémentaires sur l'enclos (type de sol, équipements, remarques...)"
        />
        <div class="flex justify-between items-center mt-1">
            <p class="text-xs text-gray-400">
                {localDescription.length} caractères
            </p>
            {#if localDescription.length > 200}
                <p class="text-xs text-orange-500">⚠️ Description longue (max recommandé: 200 caractères pour l'affichage)</p>
            {/if}
        </div>
    </div>

    <!-- Actions -->
    <div class="flex justify-end gap-3 pt-4 border-t border-gray-200">
        <Button on:click={handleCancel} variant="outline" disabled={loading}>
            Annuler
        </Button>
        <Button on:click={handleSubmit} {loading} variant="primary">
            {#if loading}
                <span class="flex items-center gap-2">
                    <svg class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    {isEdit ? "Mise à jour..." : "Création..."}
                </span>
            {:else}
                <span class="flex items-center gap-2">
                    {isEdit ? "✓" : "+"}
                    {isEdit ? "Mettre à jour" : "Créer"}
                </span>
            {/if}
        </Button>
    </div>
</div>