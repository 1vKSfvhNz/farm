<!-- src/routes/bea/+page.svelte -->
<!-- svelte-ignore a11y-label-has-associated-control -->
<script lang="ts">
  import { onMount } from "svelte";
  import { beaApi } from "$lib/api/bea";
  import { enclosApi } from "$lib/api/enclos";
  import GaugeChart from "$lib/components/charts/GaugeChart.svelte";
  import LineChart from "$lib/components/charts/LineChart.svelte";
  import Button from "$lib/components/ui/Button.svelte";
  import Modal from "$lib/components/ui/Modal.svelte";
  import Select from "$lib/components/ui/Select.svelte";
  import DatePicker from "$lib/components/ui/DatePicker.svelte";
  import Input from "$components/ui/Input.svelte";

  let indices: any[] = [];
  let enclosList: any[] = [];
  let selectedEnclosId: number = 0;
  let selectedEnclos: any = null;
  let loading = true;
  let showModal = false;
  let criteres: any[] = [];

  let formData = {
    enclos_id: 0,
    date: new Date().toISOString().split("T")[0],
    indice_global: 0,
    indice_proprete: undefined,
    indice_acces_eau: undefined,
    indice_densite: undefined,
    indice_comportement: undefined,
    notes: "",
  };

  onMount(async () => {
    await loadEnclos();
    await loadCriteres();
  });

  async function loadEnclos() {
    try {
      const response = await enclosApi.getEnclos({ limit: 100 });
      enclosList = response.items;
      if (enclosList.length > 0) {
        selectedEnclosId = enclosList[0].id;
        selectedEnclos = enclosList[0];
        await loadIndices();
      }
    } catch (error) {
      console.error("Failed to load enclos:", error);
    } finally {
      loading = false;
    }
  }

  async function loadCriteres() {
    try {
      criteres = await beaApi.getCriteres();
    } catch (error) {
      console.error("Failed to load criteres:", error);
    }
  }

  async function loadIndices() {
    if (!selectedEnclosId) return;
    loading = true;
    try {
      const response = await beaApi.getIndices(selectedEnclosId, { limit: 12 });
      indices = response.items;

      // Récupérer le dernier indice pour l'enclos sélectionné
      const latest = await beaApi.getLatestIndex(selectedEnclosId);
      if (latest) {
        selectedEnclos = { ...selectedEnclos, bea: latest };
      }
    } catch (error) {
      console.error("Failed to load indices:", error);
    } finally {
      loading = false;
    }
  }

  async function handleEnclosChange() {
    selectedEnclos = enclosList.find((e: any) => e.id === selectedEnclosId);
    await loadIndices();
  }

  function handleAdd() {
    formData = {
      enclos_id: selectedEnclosId,
      date: new Date().toISOString().split("T")[0],
      indice_global: 0,
      indice_proprete: undefined,
      indice_acces_eau: undefined,
      indice_densite: undefined,
      indice_comportement: undefined,
      notes: "",
    };
    showModal = true;
  }

  async function handleSubmit() {
    try {
      await beaApi.createIndex(formData);
      showModal = false;
      await loadIndices();
    } catch (error) {
      console.error("Failed to add index:", error);
    }
  }

  function getIndiceColor(indice: number): string {
    if (indice >= 80) return "text-green-600";
    if (indice >= 50) return "text-yellow-600";
    return "text-red-600";
  }

  function getIndiceLabel(indice: number): string {
    if (indice >= 80) return "Excellent";
    if (indice >= 60) return "Bon";
    if (indice >= 40) return "Moyen";
    if (indice >= 20) return "Médiocre";
    return "Critique";
  }

  // Préparer les données du graphique
  let chartLabels: string[] = [];
  let chartValues: number[] = [];

  $: {
    chartLabels = indices.map((i) =>
      new Date(i.date).toLocaleDateString("fr-FR"),
    );
    chartValues = indices.map((i) => i.indice_global);
  }
</script>

<div class="space-y-6">
  <div class="flex justify-between items-center">
    <div>
      <h1 class="text-2xl font-bold text-gray-900">Bien-être animal</h1>
      <p class="text-sm text-gray-500 mt-1">
        Évaluation du bien-être dans les enclos
      </p>
    </div>
    <Button on:click={handleAdd} variant="primary">
      <svg
        class="w-4 h-4 mr-1"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M12 4v16m8-8H4"
        />
      </svg>
      Nouvelle évaluation
    </Button>
  </div>

  <div class="bg-white rounded-xl border border-gray-200 p-4">
    <Select
      label="Enclos"
      bind:value={selectedEnclosId}
      options={enclosList.map((e) => ({ value: e.id, label: e.name }))}
      on:change={handleEnclosChange}
      className="w-64"
    />
  </div>

  {#if loading}
    <div class="flex justify-center items-center h-64">
      <div
        class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"
      ></div>
    </div>
  {:else}
    <!-- Indice actuel -->
    {#if selectedEnclos?.bea}
      <div
        class="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl shadow-lg p-6"
      >
        <div class="flex flex-col md:flex-row items-center gap-8">
          <div class="flex-1">
            <p class="text-gray-400 text-sm">Indice de bien-être actuel</p>
            <p class="text-5xl font-bold text-white mt-2">
              {selectedEnclos.bea.indice_global}%
            </p>
            <p
              class="text-lg font-medium {getIndiceColor(
                selectedEnclos.bea.indice_global,
              )} mt-2"
            >
              {getIndiceLabel(selectedEnclos.bea.indice_global)}
            </p>
            <p class="text-gray-400 text-sm mt-4">
              Dernière évaluation: {new Date(
                selectedEnclos.bea.date,
              ).toLocaleDateString("fr-FR")}
            </p>
          </div>
          <div class="flex-1">
            <GaugeChart
              value={selectedEnclos.bea.indice_global}
              title=""
              unit="%"
              size="lg"
            />
          </div>
        </div>
      </div>
    {/if}

    <!-- Critères détaillés -->
    {#if selectedEnclos?.bea && criteres.length > 0}
      <div class="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">
          Détail des critères
        </h3>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {#each criteres as critere}
            {@const value =
              selectedEnclos.bea[`indice_${critere.nom.toLowerCase()}`]}
            {#if value !== undefined}
              <div class="p-4 bg-gray-50 rounded-lg">
                <p class="text-sm font-medium text-gray-700">{critere.nom}</p>
                <p class="text-2xl font-bold {getIndiceColor(value)} mt-1">
                  {value}%
                </p>
                <p class="text-xs text-gray-500 mt-1">
                  Pondération: {critere.ponderation}
                </p>
              </div>
            {/if}
          {/each}
        </div>
      </div>
    {/if}

    <!-- Évolution -->
    {#if indices.length > 0}
      <LineChart
        title="Évolution de l'indice de bien-être"
        labels={chartLabels}
        datasets={[
          { label: "Indice BEA", data: chartValues, color: "#3b82f6" },
        ]}
        yAxisLabel="Indice (%)"
        height={350}
      />
    {/if}

    <!-- Historique -->
    <div class="bg-white rounded-xl border border-gray-200 shadow-sm">
      <div class="p-4 border-b border-gray-200">
        <h3 class="font-semibold text-gray-900">Historique des évaluations</h3>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="bg-gray-50">
            <tr>
              <th class="px-4 py-3 text-left">Date</th>
              <th class="px-4 py-3 text-left">Indice global</th>
              <th class="px-4 py-3 text-left">Propreté</th>
              <th class="px-4 py-3 text-left">Accès eau</th>
              <th class="px-4 py-3 text-left">Densité</th>
              <th class="px-4 py-3 text-left">Comportement</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-100">
            {#each indices as idx}
              <tr class="hover:bg-gray-50">
                <td class="px-4 py-3"
                  >{new Date(idx.date).toLocaleDateString("fr-FR")}</td
                >
                <td class="px-4 py-3">
                  <span class="font-medium {getIndiceColor(idx.indice_global)}">
                    {idx.indice_global}%
                  </span>
                </td>
                <td class="px-4 py-3"
                  >{idx.indice_proprete ? `${idx.indice_proprete}%` : "-"}</td
                >
                <td class="px-4 py-3"
                  >{idx.indice_acces_eau ? `${idx.indice_acces_eau}%` : "-"}</td
                >
                <td class="px-4 py-3"
                  >{idx.indice_densite ? `${idx.indice_densite}%` : "-"}</td
                >
                <td class="px-4 py-3"
                  >{idx.indice_comportement
                    ? `${idx.indice_comportement}%`
                    : "-"}</td
                >
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </div>
  {/if}

  <Modal
    open={showModal}
    title="Nouvelle évaluation BEA"
    on:close={() => (showModal = false)}
    size="lg"
  >
    <div class="space-y-4">
      <DatePicker label="Date" bind:value={formData.date} required />

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Input
          label="Indice global (%)"
          bind:value={formData.indice_global}
          inputType="number"
          min="0"
          max="100"
          required
        />
        <Input
          label="Indice propreté (%)"
          bind:value={formData.indice_proprete}
          inputType="number"
          min="0"
          max="100"
        />
        <Input
          label="Indice accès eau (%)"
          bind:value={formData.indice_acces_eau}
          inputType="number"
          min="0"
          max="100"
        />
        <Input
          label="Indice densité (%)"
          bind:value={formData.indice_densite}
          inputType="number"
          min="0"
          max="100"
        />
        <Input
          label="Indice comportement (%)"
          bind:value={formData.indice_comportement}
          inputType="number"
          min="0"
          max="100"
        />
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700 mb-1">Notes</label
        >
        <textarea
          bind:value={formData.notes}
          rows={3}
          class="w-full rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-2 focus:ring-primary-100 px-4 py-2"
          placeholder="Observations..."
        />
      </div>

      <div class="flex justify-end gap-3 pt-4">
        <Button on:click={() => (showModal = false)} variant="outline">
          Annuler
        </Button>
        <Button on:click={handleSubmit} variant="primary">Enregistrer</Button>
      </div>
    </div>
  </Modal>
</div>
