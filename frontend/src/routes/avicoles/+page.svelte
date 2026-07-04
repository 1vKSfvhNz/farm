<!-- src/routes/avicoles/+page.svelte -->
<script lang="ts">
  import { onMount } from "svelte";
  import { avicolesApi } from "$lib/api/avicoles";
  import AvicoleTable from "$lib/components/tables/AvicoleTable.svelte";
  import Button from "$lib/components/ui/Button.svelte";
  import Modal from "$lib/components/ui/Modal.svelte";
  import AvicoleForm from "$lib/components/forms/AvicoleForm.svelte";
  import Input from "$lib/components/ui/Input.svelte";
    import { enclosApi } from "$lib";

  let avicoles: any[] = [];
  let loading = true;
  let showModal = false;
  let showEggModal = false;
  let selectedAvicole: any = null;
  let eggCount = 0;
  let eggWeight = 0;
  let isEdit = false;
  let enclosOptions: any[] = [];

  onMount(async () => {
    await loadData();
    await loadEnclos();
  });

  async function loadData() {
    loading = true;
    try {
      const response = await avicolesApi.getAvicoles({ limit: 100 });
      avicoles = response.items;
    } catch (error) {
      console.error("Failed to load avicoles:", error);
    } finally {
      loading = false;
    }
  }

  async function loadEnclos() {
    try {
      const response = await enclosApi.getEnclos({ limit: 100 });
      enclosOptions = response.items.map((e: any) => ({
        value: e.id,
        label: e.name,
      }));
    } catch (error) {
      console.error("Failed to load enclos:", error);
    }
  }

  function handleAdd() {
    selectedAvicole = null;
    isEdit = false;
    showModal = true;
  }

  function handleEdit(avicole: any) {
    selectedAvicole = avicole;
    isEdit = true;
    showModal = true;
  }

  function handleView(avicole: any) {
    window.location.href = `/avicoles/${avicole.id}`;
  }

  function handleAddOeufs(avicole: any) {
    selectedAvicole = avicole;
    eggCount = 0;
    eggWeight = 0;
    showEggModal = true;
  }

  async function handleSubmit(formData: any) {
    try {
      if (isEdit && selectedAvicole) {
        await avicolesApi.updateAvicole(selectedAvicole.id, formData);
      } else {
        await avicolesApi.createAvicole(formData);
      }
      showModal = false;
      await loadData();
    } catch (error) {
      console.error("Failed to save avicole:", error);
    }
  }

  async function handleSubmitEggs() {
    if (selectedAvicole && eggCount > 0) {
      try {
        await avicolesApi.addEggProduction(
          selectedAvicole.id,
          eggCount,
          eggWeight,
        );
        showEggModal = false;
        await loadData();
      } catch (error) {
        console.error("Failed to add eggs:", error);
      }
    }
  }

</script>

<div class="space-y-6">
  <div class="flex justify-between items-center">
    <div>
      <h1 class="text-2xl font-bold text-gray-900">Gestion des avicoles</h1>
      <p class="text-sm text-gray-500 mt-1">Gérez votre élevage de volailles</p>
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
      Nouvel avicole
    </Button>
  </div>

  <AvicoleTable
    {avicoles}
    {loading}
    on:view={handleView}
    on:edit={handleEdit}
    on:addOeufs={handleAddOeufs}
    on:add={handleAdd}
  />

  <Modal
    open={showModal}
    title={isEdit ? "Modifier l'avicole" : "Ajouter un avicole"}
    on:close={() => (showModal = false)}
    size="lg"
  >
    <AvicoleForm
      formData={selectedAvicole || {
        identification: "",
        race: "",
        sexe: "male",
        date_arrivee: new Date().toISOString().split("T")[0],
        enclos_id: 0,
        statut: "vivant",
        production_viande: false,
        production_ponte: false,
        production_reproduction: false,
      }}
      {enclosOptions}
      loading={false}
      {isEdit}
      on:submit={handleSubmit}
      on:cancel={() => (showModal = false)}
    />
  </Modal>

  <Modal
    open={showEggModal}
    title="Ajouter une production d'œufs"
    on:close={() => (showEggModal = false)}
    size="md"
  >
    <div class="space-y-4">
      <p class="text-sm text-gray-600">
        Animal: <span class="font-medium"
          >{selectedAvicole?.identification}</span
        >
      </p>
      <Input
        label="Nombre d'œufs"
        bind:value={eggCount}
        inputType="number"
        required
        placeholder="0"
      />
      <Input
        label="Poids total (g)"
        bind:value={eggWeight}
        inputType="number"
        placeholder="0"
      />
      <div class="flex justify-end gap-3 pt-4">
        <Button on:click={() => (showEggModal = false)} variant="outline">
          Annuler
        </Button>
        <Button on:click={handleSubmitEggs} variant="primary">Ajouter</Button>
      </div>
    </div>
  </Modal>
</div>
