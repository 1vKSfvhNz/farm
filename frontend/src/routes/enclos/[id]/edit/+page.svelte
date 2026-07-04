<!-- src/routes/enclos/[id]/edit/+page.svelte -->
<script lang="ts">
  import { onMount } from "svelte";
  import { enclosApi } from "$lib/api/enclos";
  import { permissionsStore } from "$lib/stores/permissions";
  import EnclosForm from "$lib/components/forms/EnclosForm.svelte";
  import Spinner from "$lib/components/ui/Spinner.svelte";
  import Card from "$lib/components/ui/Card.svelte";
  import type { EnclosResponse } from "$lib/types/enclos";

  let id: number;
  let enclos: EnclosResponse | null = null;
  let loading = true;
  let saving = false;

  const canEdit = permissionsStore.canEditEnclos;

  onMount(async () => {
    const path = window.location.pathname;
    const match = path.match(/\/enclos\/(\d+)\/edit/);
    if (match) {
      id = parseInt(match[1]);
      await loadData();
    }
  });

  async function loadData() {
    loading = true;
    try {
      enclos = await enclosApi.getEnclosById(id);
    } catch (error) {
      console.error("Failed to load enclos:", error);
    } finally {
      loading = false;
    }
  }

  async function handleSubmit(formData: any) {
    saving = true;
    try {
      await enclosApi.updateEnclos(id, formData);
      window.location.href = `/enclos/${id}`;
    } catch (error) {
      console.error("Failed to update enclos:", error);
    } finally {
      saving = false;
    }
  }

  // Redirection si pas de permission
  if (!canEdit()) {
    window.location.href = "/enclos";
  }
</script>

<div class="min-h-screen bg-gray-50">
  <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    {#if loading}
      <div class="flex justify-center items-center h-64">
        <Spinner size="lg" />
      </div>
    {:else if enclos}
      <Card title="Modifier l'enclos" padding="none" bordered={true}>
        <div class="p-6">
          <EnclosForm
            formData={enclos}
            loading={saving}
            isEdit={true}
            on:submit={handleSubmit}
            on:cancel={() => (window.location.href = `/enclos/${id}`)}
          />
        </div>
      </Card>
    {/if}
  </div>
</div>