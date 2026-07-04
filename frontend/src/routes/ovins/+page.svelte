<!-- src/routes/ovins/+page.svelte -->
<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { goto } from "$app/navigation";
    import { ovinsApi } from "$lib/api/ovins";
    import { enclosApi } from "$lib/api/enclos";
    import Button from "$lib/components/ui/Button.svelte";
    import Modal from "$lib/components/ui/Modal.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";
    import Input from "$lib/components/ui/Input.svelte";
    import Select from "$lib/components/ui/Select.svelte";
    import AlertBadge from "$lib/components/ui/AlertBadge.svelte";
    import OvinForm from "$lib/components/forms/OvinForm.svelte";
    import type { OvinResponse } from "$lib/types/ovin";
    import type { StatutAnimal, Sexe, AnimauxFilters } from "$lib/types/animal";
    import type { EnclosResponse } from "$lib/types/enclos";
    import { API_URL } from "$lib/api/client";
    import { getStatutBadge } from "$lib/stores/animal";
    import { ovinsStore } from "$lib/stores/ovins";
    import { getFullPhotoUrl, handleImageError } from "$lib/utils/media";

    // ✅ Restaurer les filtres depuis sessionStorage s'ils existent
    function getStoredFilters(): AnimauxFilters | null {
        const stored = ovinsStore.filters.get();
        if (stored) {
            return stored;
        }
        return null;
    }

    // ✅ Initialiser avec les filtres stockés ou les valeurs par défaut
    const storedFilters = getStoredFilters();
    
    // Si des filtres sont stockés, on les utilise, sinon on utilise les valeurs par défaut
    let searchQuery = storedFilters?.searchQuery ?? "";
    let selectedRace = storedFilters?.selectedRace ?? "";
    let selectedEnclos: string | number = storedFilters?.selectedEnclos ?? "";
    let selectedStatuts: StatutAnimal[] = storedFilters?.selectedStatuts ?? ["vivant"];
    let selectedSexes: Sexe[] = storedFilters?.selectedSexes ?? [];

    // Données
    let ovins: OvinResponse[] = [];
    let filteredOvins: OvinResponse[] = [];
    let enclosList: EnclosResponse[] = [];
    let loading = true;
    let showModal = false;
    let selectedOvin: OvinResponse | null = null;
    let isEdit = false;

    // Statistiques
    let stats = {
        total: 0,
        males: 0,
        femelles: 0,
        hermaphrodites: 0,
        productionViande: 0,
        productionLaine: 0,
        productionReproduction: 0,
    };

    // Options
    const statutOptions: { value: StatutAnimal; label: string; color: string }[] = [
        { value: "vivant", label: "Vivant", color: "green" },
        { value: "vendu", label: "Vendu", color: "orange" },
        { value: "decede", label: "Décédé", color: "red" },
        { value: "transfere", label: "Transféré", color: "blue" },
    ];

    const sexeOptions: { value: Sexe; label: string; emoji: string }[] = [
        { value: "male", label: "Bélier", emoji: "♂" },
        { value: "femelle", label: "Brebis", emoji: "♀" },
        { value: "hermaphrodite", label: "Hermaphrodite", emoji: "⚥" },
    ];

    let raceOptions: { value: string; label: string }[] = [
        { value: "", label: "Toutes les races" },
        { value: "Merino", label: "Merino" },
        { value: "Suffolk", label: "Suffolk" },
        { value: "Dorset", label: "Dorset" },
        { value: "Rambouillet", label: "Rambouillet" },
        { value: "Texel", label: "Texel" },
    ];

    let enclosOptions: { value: number; label: string }[] = [
        { value: 0, label: "Tous les enclos" }
    ];

    // Mise à jour des filtres dans sessionStorage et l'URL
    function updateFilters() {
        // ✅ Sauvegarder TOUS les filtres dans sessionStorage via le store
        const filtersToStore: AnimauxFilters = {
            searchQuery,
            selectedRace,
            selectedEnclos: String(selectedEnclos),
            selectedStatuts,
            selectedSexes
        };
        ovinsStore.filters.set(filtersToStore);
        
        // Mettre à jour l'URL
        const params = new URLSearchParams();
        if (searchQuery) params.set('search', searchQuery);
        if (selectedRace) params.set('race', selectedRace);
        if (selectedEnclos && selectedEnclos !== 0 && selectedEnclos !== "") {
            params.set('enclos', String(selectedEnclos));
        }
        if (selectedStatuts.length > 0 && !(selectedStatuts.length === 1 && selectedStatuts[0] === 'vivant')) {
            params.set('statuts', selectedStatuts.join(','));
        }
        if (selectedSexes.length > 0) {
            params.set('sexes', selectedSexes.join(','));
        }
        
        const url = `/ovins${params.toString() ? '?' + params.toString() : ''}`;
        history.replaceState({}, '', url);
    }

    onMount(async () => {
        // ✅ Charger les enclos puis les données
        await loadEnclos();
        await loadData();
    });

    onDestroy(() => {
        // Ne pas effacer les filtres, ils restent en sessionStorage
    });

    async function loadEnclos() {
        try {
            const response = await enclosApi.getEnclos({ limit: 100 });
            enclosList = response.items || [];
            enclosOptions = [
                { value: 0, label: "Tous les enclos" },
                ...enclosList.map(e => ({ 
                    value: e.id,
                    label: e.name
                }))
            ];
            
            // ✅ Vérifier si l'enclos sélectionné existe toujours
            if (selectedEnclos && !enclosOptions.some(opt => opt.value === Number(selectedEnclos))) {
                selectedEnclos = "";
            }
        } catch (error) {
            console.error("Failed to load enclos:", error);
        }
    }

    async function loadData() {
        loading = true;
        try {
            // ✅ Convertir enclos_id en nombre si sélectionné
            const enclosId = selectedEnclos ? Number(selectedEnclos) : undefined;
            
            const response = await ovinsApi.getOvins({ 
                limit: 200, 
                statut: selectedStatuts,
                race: selectedRace,
                sexe: selectedSexes,
                enclos_id: enclosId,
            });
            ovins = response.items || response || [];
            stats.total = response.total || ovins.length;
            applyFilters();
            calculateStats();
            updateFilters(); // ✅ Sauvegarder les filtres après chargement
        } catch (error) {
            console.error("Failed to load ovins:", error);
        } finally {
            loading = false;
        }
    }

    function toggleStatut(value: StatutAnimal) {
        const index = selectedStatuts.indexOf(value);
        if (index === -1) {
            selectedStatuts = [...selectedStatuts, value];
        } else {
            selectedStatuts = selectedStatuts.filter(s => s !== value);
        }
        if (selectedStatuts.length === 0) {
            selectedStatuts = statutOptions.map(s => s.value);
        }
        updateFilters();
        loadData();
    }

    function toggleAllStatuts() {
        if (selectedStatuts.length === statutOptions.length) {
            selectedStatuts = [];
        } else {
            selectedStatuts = statutOptions.map(s => s.value);
        }
        updateFilters();
        loadData();
    }

    function toggleSexe(value: Sexe) {
        const index = selectedSexes.indexOf(value);
        if (index === -1) {
            selectedSexes = [...selectedSexes, value];
        } else {
            selectedSexes = selectedSexes.filter(s => s !== value);
        }
        updateFilters();
        loadData();
    }

    function toggleAllSexes() {
        if (selectedSexes.length === sexeOptions.length) {
            selectedSexes = [];
        } else {
            selectedSexes = sexeOptions.map(s => s.value);
        }
        updateFilters();
        loadData();
    }

    function applyFilters() {
        filteredOvins = ovins.filter((o) => {
            const matchesSearch =
                !searchQuery ||
                o.identification?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                o.race?.toLowerCase().includes(searchQuery.toLowerCase());

            const matchesRace = !selectedRace || o.race === selectedRace;
            const matchesEnclos = !selectedEnclos || o.enclos_id === Number(selectedEnclos);
            const matchesStatut = selectedStatuts.length === 0 || selectedStatuts.includes(o.statut as StatutAnimal);
            const matchesSexe = selectedSexes.length === 0 || selectedSexes.includes(o.sexe as Sexe);

            return matchesSearch && matchesRace && matchesEnclos && matchesStatut && matchesSexe;
        });
    }

    function calculateStats() {
        stats.males = ovins.filter((o) => o.sexe === "male").length;
        stats.femelles = ovins.filter((o) => o.sexe === "femelle").length;
        stats.hermaphrodites = ovins.filter((o) => o.sexe === "hermaphrodite").length;
        stats.productionViande = ovins.filter((o) => o.production_viande).length;
        stats.productionLaine = ovins.filter((o) => o.production_laine).length;
        stats.productionReproduction = ovins.filter((o) => o.production_reproduction).length;
    }

    function resetFilters() {
        searchQuery = "";
        selectedRace = "";
        selectedEnclos = "";
        selectedStatuts = ["vivant"];
        selectedSexes = [];
        updateFilters();
        loadData();
        // ✅ Réinitialiser les filtres stockés
        ovinsStore.filters.reset();
    }

    function goBack() {
        ovinsStore.filters.reset();
        ovinsStore.clear();
        goto("/");
    }

    function handleAdd() {
        selectedOvin = null;
        isEdit = false;
        showModal = true;
    }

    function handleEdit(ovin: OvinResponse) {
        selectedOvin = ovin;
        isEdit = true;
        showModal = true;
    }

    function handleView(ovin: OvinResponse) {
        // ✅ Sauvegarder TOUS les filtres avant de naviguer
        const filters: AnimauxFilters = {
            searchQuery,
            selectedRace,
            selectedEnclos: String(selectedEnclos),
            selectedStatuts,
            selectedSexes
        };

        ovinsStore.filters.set(filters);        
        ovinsStore.setOvin(ovin);
        goto(`/ovins/${ovin.id}`);
    }

    async function handleSubmit(event: CustomEvent) {
        const formData = event.detail;
        console.log("Données reçues:", formData);
        
        try {
            if (isEdit && selectedOvin) {
                await ovinsApi.updateOvin(selectedOvin.id, formData);
            } else {
                await ovinsApi.createOvin(formData);
            }
            showModal = false;
            await loadData();
        } catch (error) {
            console.error("Failed to save ovin:", error);
        }
    }

    function areAllStatutsSelected() {
        return selectedStatuts.length === statutOptions.length;
    }

    function areAllSexesSelected() {
        return selectedSexes.length === sexeOptions.length;
    }
</script>

<div class="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Bouton retour -->
        <div class="mb-6">
            <button
                on:click={goBack}
                class="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 hover:border-gray-400 transition-all duration-200 shadow-sm"
            >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
                <span>Retour</span>
            </button>
        </div>

        <!-- En-tête -->
        <div class="mb-8">
            <div class="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
                <div>
                    <div class="flex items-center gap-3 mb-2">
                        <div class="text-5xl">🐑</div>
                        <div>
                            <h1 class="text-3xl font-bold text-gray-900">
                                Gestion des ovins
                            </h1>
                            <p class="text-sm text-gray-600 mt-1">
                                Gérez votre troupeau ovin
                            </p>
                        </div>
                    </div>
                </div>
                <Button on:click={handleAdd} variant="primary" size="md" className="shadow-lg">
                    <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
                    </svg>
                    Nouvel ovin
                </Button>
            </div>
        </div>

        <!-- Cartes statistiques -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4 mb-8">
            <div class="bg-white rounded-xl shadow-md p-5 hover:shadow-lg transition-shadow duration-300 border-l-4 border-blue-500">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm text-gray-500">Total ovins</p>
                        <p class="text-2xl font-bold text-gray-900">{stats.total}</p>
                    </div>
                    <div class="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center text-2xl">🐑</div>
                </div>
            </div>

            <div class="bg-white rounded-xl shadow-md p-5 hover:shadow-lg transition-shadow duration-300 border-l-4 border-purple-500">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm text-gray-500">Béliers / Brebis</p>
                        <p class="text-2xl font-bold text-gray-900">{stats.males} / {stats.femelles}</p>
                    </div>
                    <div class="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center text-2xl">⚥</div>
                </div>
            </div>

            <div class="bg-white rounded-xl shadow-md p-5 hover:shadow-lg transition-shadow duration-300 border-l-4 border-red-500">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm text-gray-500">Production viande</p>
                        <p class="text-2xl font-bold text-gray-900">{stats.productionViande}</p>
                    </div>
                    <div class="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center text-2xl">🥩</div>
                </div>
            </div>

            <div class="bg-white rounded-xl shadow-md p-5 hover:shadow-lg transition-shadow duration-300 border-l-4 border-yellow-500">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm text-gray-500">Production laine</p>
                        <p class="text-2xl font-bold text-gray-900">{stats.productionLaine}</p>
                    </div>
                    <div class="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center text-2xl">🧶</div>
                </div>
            </div>

            <div class="bg-white rounded-xl shadow-md p-5 hover:shadow-lg transition-shadow duration-300 border-l-4 border-green-500">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-sm text-gray-500">Reproduction</p>
                        <p class="text-2xl font-bold text-gray-900">{stats.productionReproduction}</p>
                    </div>
                    <div class="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center text-2xl">🐣</div>
                </div>
            </div>

            {#if stats.hermaphrodites > 0}
                <div class="bg-white rounded-xl shadow-md p-5 hover:shadow-lg transition-shadow duration-300 border-l-4 border-orange-500">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-sm text-gray-500">Hermaphrodites</p>
                            <p class="text-2xl font-bold text-gray-900">{stats.hermaphrodites}</p>
                        </div>
                        <div class="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center text-2xl">⚥</div>
                    </div>
                </div>
            {/if}
        </div>

        <!-- Barre de recherche et filtres -->
        <div class="bg-white rounded-xl shadow-md mb-6 p-5">
            <div class="flex flex-col gap-4">
                <!-- Recherche -->
                <div class="flex-1">
                    <Input
                        placeholder="🔍 Rechercher par identification ou race..."
                        bind:value={searchQuery}
                        on:input={() => { applyFilters(); loadData(); }}
                    />
                </div>

                <!-- Filtres en ligne -->
                <div class="flex flex-wrap items-center gap-4">
                    <!-- Race -->
                    <div class="min-w-[150px]">
                        <Select
                            bind:value={selectedRace}
                            options={raceOptions}
                            on:change={() => { applyFilters(); loadData(); }}
                            className="w-full"
                        />
                    </div>

                    <!-- Enclos -->
                    <div class="min-w-[180px]">
                        <Select
                            bind:value={selectedEnclos}
                            options={enclosOptions.map(opt => ({
                                ...opt,
                                value: String(opt.value)
                            }))}
                            on:change={() => { applyFilters(); loadData(); }}
                            className="w-full"
                        />
                    </div>

                    <!-- Checkboxes Statuts -->
                    <div class="flex flex-wrap items-center gap-2">
                        <span class="text-sm font-medium text-gray-600 mr-1">Statut:</span>
                        <button
                            on:click={toggleAllStatuts}
                            class="px-2 py-1 text-xs font-medium rounded transition-colors {areAllStatutsSelected() ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-600 hover:bg-gray-300'}"
                        >
                            Tous
                        </button>
                        {#each statutOptions as option}
                            <label class="inline-flex items-center gap-1 px-2 py-1 rounded-full text-sm cursor-pointer transition-colors hover:bg-gray-100 {selectedStatuts.includes(option.value) ? 'bg-blue-50 border border-blue-300' : 'bg-gray-50 border border-gray-200'}">
                                <input
                                    type="checkbox"
                                    checked={selectedStatuts.includes(option.value)}
                                    on:change={() => toggleStatut(option.value)}
                                    class="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                                />
                                <span class="text-xs font-medium">
                                    {option.label}
                                </span>
                            </label>
                        {/each}
                    </div>

                    <!-- Checkboxes Sexes -->
                    <div class="flex flex-wrap items-center gap-2">
                        <span class="text-sm font-medium text-gray-600 mr-1">Sexe:</span>
                        <button
                            on:click={toggleAllSexes}
                            class="px-2 py-1 text-xs font-medium rounded transition-colors {areAllSexesSelected() ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-600 hover:bg-gray-300'}"
                        >
                            Tous
                        </button>
                        {#each sexeOptions as option}
                            <label class="inline-flex items-center gap-1 px-2 py-1 rounded-full text-sm cursor-pointer transition-colors hover:bg-gray-100 {selectedSexes.includes(option.value) ? 'bg-blue-50 border border-blue-300' : 'bg-gray-50 border border-gray-200'}">
                                <input
                                    type="checkbox"
                                    checked={selectedSexes.includes(option.value)}
                                    on:change={() => toggleSexe(option.value)}
                                    class="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                                />
                                <span class="text-xs font-medium">{option.emoji} {option.label}</span>
                            </label>
                        {/each}
                    </div>

                    <!-- Réinitialiser -->
                    {#if searchQuery || selectedRace || selectedEnclos || selectedStatuts.length !== 1 || selectedStatuts[0] !== 'vivant' || selectedSexes.length > 0}
                        <Button variant="outline" size="sm" on:click={resetFilters}>
                            🔄 Réinitialiser
                        </Button>
                    {/if}
                </div>
            </div>
        </div>

        <!-- Tableau des ovins -->
        {#if loading}
            <div class="flex justify-center items-center h-64">
                <Spinner size="lg" />
            </div>
        {:else if filteredOvins.length === 0}
            <div class="bg-white rounded-xl shadow-md p-12 text-center">
                <div class="text-7xl mb-4">🐑</div>
                <h3 class="text-xl font-medium text-gray-900 mb-2">Aucun ovin trouvé</h3>
                <p class="text-gray-500 mb-6">Aucun ovin ne correspond à vos critères de recherche.</p>
                <Button on:click={handleAdd} variant="primary">➕ Ajouter un ovin</Button>
            </div>
        {:else}
            <div class="bg-white rounded-xl shadow-md overflow-hidden">
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gradient-to-r from-gray-50 to-gray-100">
                            <tr>
                                <th class="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Photo</th>
                                <th class="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Identification</th>
                                <th class="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Race</th>
                                <th class="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Sexe</th>
                                <th class="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Âge</th>
                                <th class="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Production</th>
                                <th class="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Statut</th>
                                <th class="px-6 py-4 text-right text-xs font-semibold text-gray-600 uppercase tracking-wider">Actions</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-100">
                            {#each filteredOvins as ovin}
                                <tr class="hover:bg-gray-50 transition-colors duration-150">
                                    <td class="px-6 py-4 whitespace-nowrap">
                                        {#if ovin.photo_url}
                                            <img
                                                src={getFullPhotoUrl(ovin.photo_url)}
                                                alt={ovin.identification}
                                                class="w-12 h-12 rounded-full object-cover border-2 border-gray-200"
                                                loading="lazy"
                                                on:error={handleImageError}
                                            />
                                        {:else}
                                            <div class="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center text-2xl">🐑</div>
                                        {/if}
                                    </td>
                                    <td class="px-6 py-4 whitespace-nowrap">
                                        <span class="font-semibold text-gray-900">{ovin.identification}</span>
                                    </td>
                                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-600">{ovin.race || "-"}</td>
                                    <td class="px-6 py-4 whitespace-nowrap">
                                        {#if ovin.sexe === "male"}
                                            <span class="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">♂ Bélier</span>
                                        {:else if ovin.sexe === "femelle"}
                                            <span class="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-pink-100 text-pink-800">♀ Brebis</span>
                                        {:else}
                                            <span class="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">⚥ Hermaphrodite</span>
                                        {/if}
                                    </td>
                                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-600">{ovin.age_mois || "-"} mois</td>
                                    <td class="px-6 py-4 whitespace-nowrap">
                                        <div class="flex gap-1 flex-wrap">
                                            {#if ovin.production_viande}
                                                <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800">🥩 Viande</span>
                                            {/if}
                                            {#if ovin.production_laine}
                                                <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800">🧶 Laine</span>
                                            {/if}
                                            {#if ovin.production_reproduction}
                                                <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">🐣 Repro</span>
                                            {/if}
                                        </div>
                                    </td>
                                    <td class="px-6 py-4 whitespace-nowrap">
                                        {#if ovin}
                                            {@const badge = getStatutBadge(ovin.statut)}
                                            <AlertBadge niveau={badge.niveau} label={badge.label} size="sm" />
                                        {/if}
                                    </td>
                                    <td class="px-6 py-4 whitespace-nowrap text-right">
                                        <div class="flex justify-end gap-2">
                                            {#if ovin.statut === 'vivant' || ovin.statut === 'transfere'}
                                                <button
                                                    on:click={() => handleEdit(ovin)}
                                                    class="p-1 text-gray-400 hover:text-green-600 transition-colors"
                                                    title="Modifier"
                                                >
                                                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                                                    </svg>
                                                </button>
                                            {/if}
                                            <button
                                                on:click={() => handleView(ovin)}
                                                class="p-1 text-gray-400 hover:text-blue-600 transition-colors"
                                                title="Voir"
                                            >
                                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                                </svg>
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                </div>
            </div>
        {/if}
    </div>

    <!-- Modal formulaire -->
    <Modal open={showModal} title={isEdit ? "✏️ Modifier l'ovin" : "➕ Ajouter un ovin"} on:close={() => (showModal = false)} size="lg">
        <OvinForm
            formData={selectedOvin ? {
                type_espece: 'ovin',
                race: selectedOvin.race,
                sexe: selectedOvin.sexe,
                date_naissance: selectedOvin.date_naissance || null,
                date_arrivee: selectedOvin.date_arrivee,
                provenance: selectedOvin.provenance,
                prix_achat: selectedOvin.prix_achat || undefined,
                enclos_id: selectedOvin.enclos_id || 0,
                statut: selectedOvin.statut,
                production_viande: selectedOvin.production_viande,
                production_reproduction: selectedOvin.production_reproduction,
                production_laine: selectedOvin.production_laine,
                qualite_laine: selectedOvin.qualite_laine || "",
                notes: selectedOvin.notes || undefined,
                photo_url: selectedOvin.photo_url || null
            } : {
                type_espece: "ovin",
                race: "",
                sexe: "male",
                date_naissance: null,
                date_arrivee: new Date().toISOString().split("T")[0],
                provenance: "",
                prix_achat: undefined,
                enclos_id: 0,
                statut: "vivant",
                production_viande: false,
                production_reproduction: false,
                production_laine: false,
                qualite_laine: "",
                notes: "",
                photo_url: null
            }}
            loading={false}
            {isEdit}
            on:submit={handleSubmit}
            on:cancel={() => (showModal = false)}
        />
    </Modal>
</div>