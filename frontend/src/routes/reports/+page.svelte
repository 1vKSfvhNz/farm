<!-- src/routes/reports/+page.svelte - Version corrigée -->
<script lang="ts">
  import { permissionsStore } from "$lib/stores/permissions";
  import Card from "$lib/components/ui/Card.svelte";
  import Button from "$lib/components/ui/Button.svelte";
  import DatePicker from "$lib/components/ui/DatePicker.svelte";
  import Select from "$lib/components/ui/Select.svelte";
  import { exportsApi } from "$lib/api/exports";

  let startDate = "";
  let endDate = "";
  let reportType = "animaux";
  let format = "csv";
  let loading = false;
  let isAdmin = false;

  const reportTypes = [
    { value: "animaux", label: "Liste des animaux" },
    { value: "financier", label: "Rapport financier" },
    { value: "vaccinations", label: "Calendrier vaccinal" },
    { value: "mortalite", label: "Taux de mortalité" },
    { value: "pesees", label: "Historique des pesées" },
  ];

  const formatOptions = [
    { value: "csv", label: "CSV" },
    { value: "pdf", label: "PDF" },
  ];

  // S'abonner aux changements de permissions
  permissionsStore.subscribe((state) => {
    isAdmin = state.role === "admin";
  });

  async function handleExport() {
    loading = true;
    try {
      let blob: Blob;
      const filename = `${reportType}_${new Date().toISOString().split("T")[0]}.${format}`;

      switch (reportType) {
        case "animaux":
          blob = await exportsApi.exportAnimalsCSV();
          break;
        case "financier":
          if (startDate && endDate) {
            blob = await exportsApi.exportFinancialCSV(startDate, endDate);
          } else {
            alert("Dates requises pour le rapport financier");
            return;
          }
          break;
        case "vaccinations":
          blob = await exportsApi.exportVaccinationsCSV(
            undefined,
            startDate || undefined,
            endDate || undefined,
          );
          break;
        case "mortalite":
          blob = await exportsApi.exportMortalityCSV(
            undefined,
            startDate || undefined,
            endDate || undefined,
          );
          break;
        case "pesees":
          blob = await exportsApi.exportWeighingsCSV(
            undefined,
            startDate || undefined,
            endDate || undefined,
          );
          break;
        default:
          throw new Error("Type de rapport invalide");
      }

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Export failed:", error);
      alert("Erreur lors de l'export");
    } finally {
      loading = false;
    }
  }

  const canExport = permissionsStore.hasPermission("can_export_data");
</script>

<div class="space-y-6">
  <div class="flex justify-between items-center">
    <div>
      <h1 class="text-2xl font-bold text-gray-900">Rapports</h1>
      <p class="text-sm text-gray-500 mt-1">Exportez vos données</p>
    </div>
  </div>

  <Card title="Générer un rapport">
    <div class="space-y-5">
      <Select
        label="Type de rapport"
        bind:value={reportType}
        options={reportTypes}
      />

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <DatePicker label="Date de début" bind:value={startDate} />
        <DatePicker label="Date de fin" bind:value={endDate} />
      </div>

      <Select label="Format" bind:value={format} options={formatOptions} />

      <div class="flex justify-end">
        <Button
          on:click={handleExport}
          {loading}
          variant="primary"
          disabled={!canExport}
        >
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
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
            />
          </svg>
          Exporter
        </Button>
      </div>
    </div>
  </Card>

  {#if isAdmin}
    <Card title="Rapports avancés">
      <div class="space-y-3">
        <button
          on:click={() => (window.location.href = "/exports/animaux")}
          class="w-full text-left px-4 py-2 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors"
        >
          📊 Rapport de performance par espèce
        </button>
        <button
          on:click={() => (window.location.href = "/exports/financier")}
          class="w-full text-left px-4 py-2 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors"
        >
          💰 Analyse financière détaillée
        </button>
        <button
          on:click={() => (window.location.href = "/exports/sante")}
          class="w-full text-left px-4 py-2 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors"
        >
          🏥 Rapport sanitaire
        </button>
      </div>
    </Card>
  {/if}
</div>
