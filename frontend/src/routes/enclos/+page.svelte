<!-- src/routes/enclos/+page.svelte -->
<script lang="ts">
  import { onMount } from "svelte";
  import { enclosApi } from "$lib/api/enclos";
  import Button from "$lib/components/ui/Button.svelte";
  import Card from "$lib/components/ui/Card.svelte";
  import Modal from "$lib/components/ui/Modal.svelte";
  import EnclosForm from "$lib/components/forms/EnclosForm.svelte";
  import Input from "$lib/components/ui/Input.svelte";
  import Select from "$lib/components/ui/Select.svelte";
  import Spinner from "$lib/components/ui/Spinner.svelte";
  import type { EnclosResponse, EnclosType } from "$lib/types/enclos";

  let enclos: EnclosResponse[] = [];
  let filteredEnclos: EnclosResponse[] = [];
  let loading = true;
  let showModal = false;
  let selectedEnclos: EnclosResponse | null = null;
  let isEdit = false;
  
  // Filtres
  let searchQuery = "";
  let selectedType: string = "";
  let selectedZone: string = "";
  
  // Gestion du debounce
  let searchTimeout: NodeJS.Timeout;
  let filterTimeout: NodeJS.Timeout;
  let isFiltering = false;
  
  // Statistiques
  let stats = {
    total: 0,
  };

  const typeOptions = [
    { value: "", label: "Tous les types" },
    { value: "enclos", label: "Enclos" },
    { value: "bassin", label: "Bassin" },
    { value: "pâturage", label: "Pâturage" },
    { value: "cage", label: "Cage" },
    { value: "bac", label: "Bac" }
  ];

  // Stocker toutes les zones disponibles
  let allZones: string[] = [];
  let zoneOptions: { value: string; label: string }[] = [];

  onMount(async () => {
    await loadAllZones();
    await loadEnclos();
  });

  // Fonction pour charger les enclos avec les filtres actuels
  async function loadEnclos() {
    // Éviter les appels multiples simultanés
    if (isFiltering) return;
    
    isFiltering = true;
    loading = true;
    
    try {
      const params: {
        skip?: number;
        limit?: number;
        page?: number;
        enclos_type?: string | string[];
        zone?: string;
      } = { limit: 100 };
      
      // Ne passer le paramètre que si la valeur n'est pas vide
      if (selectedType && selectedType !== "") {
        params.enclos_type = selectedType;
      }
      
      if (selectedZone && selectedZone !== "") {
        params.zone = selectedZone;
      }
      
      console.log("Loading enclos with params:", params);
      
      const response = await enclosApi.getEnclos(params);
      enclos = response.items || response || [];
      
      calculateStats();
      applySearchFilter(); // Appliquer le filtre de recherche après chargement
    } catch (error) {
      console.error("Failed to load enclos:", error);
    } finally {
      loading = false;
      isFiltering = false;
    }
  }

  // Charger toutes les zones disponibles
  async function loadAllZones() {
    try {
      const response = await enclosApi.getEnclos({ limit: 1000 });
      const allEnclos = response.items || response || [];
      const zones = new Set(allEnclos.map(e => e.zone).filter(Boolean));
      allZones = Array.from(zones) as string[];
      updateZoneOptions();
    } catch (error) {
      console.error("Failed to load zones:", error);
    }
  }

  function updateZoneOptions() {
    zoneOptions = [
      { value: "", label: "Toutes les zones" },
      ...allZones.map(zone => ({ value: zone, label: zone }))
    ];
  }

  // Appliquer le filtre de recherche (côté client)
  function applySearchFilter() {
    if (!searchQuery) {
      filteredEnclos = enclos;
    } else {
      filteredEnclos = enclos.filter(e => 
        e.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (e.description && e.description.toLowerCase().includes(searchQuery.toLowerCase()))
      );
    }
  }

  // Gestionnaires d'événements avec debounce
  function handleTypeChange() {
    clearTimeout(filterTimeout);
    filterTimeout = setTimeout(() => {
      loadEnclos();
    }, 300);
  }

  function handleZoneChange() {
    clearTimeout(filterTimeout);
    filterTimeout = setTimeout(() => {
      loadEnclos();
    }, 300);
  }

  function handleSearchInput() {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      applySearchFilter();
    }, 300);
  }

  function calculateStats() {
    stats.total = enclos.length;
  }

  async function resetFilters() {
    // Annuler les timeouts en cours
    clearTimeout(filterTimeout);
    clearTimeout(searchTimeout);
    
    searchQuery = "";
    selectedType = "";
    selectedZone = "";
    
    // Recharger immédiatement
    await loadEnclos();
  }

  function handleAdd() {
    selectedEnclos = null;
    isEdit = false;
    showModal = true;
  }

  function handleEdit(enclo: EnclosResponse) {
    selectedEnclos = enclo;
    isEdit = true;
    showModal = true;
  }

  function handleView(enclo: EnclosResponse) {
    window.location.href = `/enclos/${enclo.id}`;
  }

  async function handleSubmit(event: CustomEvent) {
    const formData = event.detail;
        
    if (!formData) {
        alert("Pas de données reçus");
        return;
    }
    
    try {
        if (isEdit && selectedEnclos) {
            await enclosApi.updateEnclos(selectedEnclos.id, formData);
        } else {
            await enclosApi.createEnclos(formData);
        }
        showModal = false;
        await new Promise(resolve => setTimeout(resolve, 100));
        await loadAllZones();
        await loadEnclos();
    } catch (error) {
        console.error("Failed to save enclos:", error);
    }
  }

  function getTypeIcon(type: EnclosType): string {
    const icons: Record<EnclosType, string> = {
      enclos: "🏡",
      bassin: "💧",
      pâturage: "🌿",
      cage: "🪶",
      bac: "📦"
    };
    return icons[type] || "📍";
  }

  function formatDate(dateString: string): string {
    if (!dateString) return "";
    const date = new Date(dateString);
    return date.toLocaleDateString('fr-FR', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric'
    });
  }

  function goBack() {
    window.location.href = "/";
  }
</script>

<head>
  <style>
    .line-clamp-2 {
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
  </style>
</head>

<div class="min-h-screen bg-gray-50">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <!-- Bouton retour -->
    <div class="mb-4">
      <Button
        on:click={goBack}
        variant="ghost"
        size="sm"
        className="gap-2 border border-gray-300 text-gray-700 hover:bg-gray-100 hover:border-gray-400"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
        </svg>
        <span>Retour</span>
      </Button>
    </div>

    <!-- En-tête -->
    <div class="mb-8">
      <div class="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
        <div>
          <h1 class="text-3xl font-bold text-gray-900">🏡 Gestion des enclos</h1>
          <p class="text-sm text-gray-500 mt-1">Gérez vos espaces d'élevage</p>
        </div>
        <Button on:click={handleAdd} variant="primary" size="md">
          <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
          Nouvel enclos
        </Button>
      </div>
    </div>

    <!-- Cartes statistiques -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <Card padding="md" bordered={true}>
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm text-gray-500">Total enclos</p>
            <p class="text-2xl font-bold text-gray-900">{stats.total}</p>
          </div>
          <div class="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center text-xl">🏡</div>
        </div>
      </Card>
    </div>

    <!-- Filtres -->
    <Card padding="md" bordered={true} className="mb-6">
      <div class="flex flex-col lg:flex-row gap-4">
        <div class="flex-1">
          <Input
            placeholder="Rechercher par nom, description..."
            bind:value={searchQuery}
            on:input={handleSearchInput}
            icon="search"
            iconPosition="left"
          />
        </div>
        <div class="flex flex-wrap gap-2">
          <Select 
            bind:value={selectedType} 
            options={typeOptions} 
            on:change={handleTypeChange}
            className="w-36" 
          />
          <Select 
            bind:value={selectedZone} 
            options={zoneOptions} 
            on:change={handleZoneChange}
            className="w-40" 
          />
          {#if searchQuery || (selectedType && selectedType !== "") || (selectedZone && selectedZone !== "")}
            <Button variant="outline" size="sm" on:click={resetFilters}>
              Réinitialiser
            </Button>
          {/if}
        </div>
      </div>
    </Card>

    <!-- Liste des enclos -->
    {#if loading}
      <div class="flex justify-center items-center h-64"><Spinner size="lg" /></div>
    {:else if filteredEnclos.length === 0}
      <Card padding="lg" bordered={true}>
        <div class="text-center py-12">
          <div class="text-6xl mb-4">🏡</div>
          <h3 class="text-lg font-medium text-gray-900 mb-2">Aucun enclos trouvé</h3>
          <p class="text-gray-500 mb-4">Aucun enclos ne correspond à vos critères de recherche.</p>
          <Button on:click={handleAdd} variant="primary">Ajouter un enclos</Button>
        </div>
      </Card>
    {:else}
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {#each filteredEnclos as enclo}
          <Card padding="none" bordered={true} hoverable={true}>
            <div class="p-5">
              <!-- En-tête -->
              <div class="flex items-start justify-between mb-4">
                <div class="flex items-center gap-3">
                  <div class="w-12 h-12 bg-gray-100 rounded-xl flex items-center justify-center text-2xl">
                    {getTypeIcon(enclo.type)}
                  </div>
                  <div>
                    <h3 class="text-lg font-semibold text-gray-900">{enclo.name}</h3>
                    <p class="text-sm text-gray-500 capitalize">{enclo.type}</p>
                  </div>
                </div>
                <div class="flex gap-1">
                  <button on:click={() => handleEdit(enclo)} class="p-1.5 text-gray-400 hover:text-green-600 transition-colors rounded-lg hover:bg-gray-100" title="Modifier">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                  </button>
                  <button on:click={() => handleView(enclo)} class="p-1.5 text-gray-400 hover:text-blue-600 transition-colors rounded-lg hover:bg-gray-100" title="Voir">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  </button>
                </div>
              </div>

              <!-- Dimensions -->
              <div class="grid grid-cols-2 gap-3 mb-3 text-sm">
                <div>
                  <span class="text-gray-500">Surface:</span>
                  <span class="font-medium ml-2">{enclo.surface} m²</span>
                </div>
                {#if enclo.volume}
                  <div>
                    <span class="text-gray-500">Volume:</span>
                    <span class="font-medium ml-2">{enclo.volume} m³</span>
                  </div>
                {/if}
              </div>

              <!-- Dimensions détaillées -->
              <div class="grid grid-cols-3 gap-2 mb-3 text-xs text-gray-500 bg-gray-50 rounded-lg p-2">
                <div>Long.: {enclo.longueur}m</div>
                <div>Larg.: {enclo.largeur}m</div>
                {#if enclo.hauteur}
                  <div>Haut.: {enclo.hauteur}m</div>
                {/if}
              </div>

              <!-- Zone -->
              {#if enclo.zone}
                <div class="flex items-center gap-2 text-sm text-gray-500 mb-3">
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <span>Zone: {enclo.zone}</span>
                </div>
              {/if}

              <!-- Description (si présente) -->
              {#if enclo.description}
                <div class="mt-3 pt-3 border-t border-gray-100">
                  <p class="text-sm text-gray-600 line-clamp-2">
                    <span class="text-gray-400 text-xs">📝</span> {enclo.description}
                  </p>
                </div>
              {/if}

              <!-- Date de création -->
              <div class="mt-3 text-xs text-gray-400">
                Créé le {formatDate(enclo.created_at)}
              </div>
            </div>
          </Card>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Modals -->
  <Modal open={showModal} title={isEdit ? "Modifier l'enclos" : "Ajouter un enclos"} on:close={() => (showModal = false)} size="lg">
    <EnclosForm 
      formData={selectedEnclos || { 
        name: "", 
        type: "enclos", 
        longueur: 0, 
        largeur: 0,
        hauteur: undefined,
        zone: "",
        description: ""
      }} 
      loading={false} 
      {isEdit} 
      on:submit={handleSubmit} 
      on:cancel={() => (showModal = false)} 
    />
  </Modal>
</div>