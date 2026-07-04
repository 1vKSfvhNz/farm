<!-- src/routes/piscicoles/+page.svelte -->
<script lang="ts">
  import { onMount } from "svelte";
  import { piscicolesApi } from "$lib/api/piscicoles";
  import PiscicoleTable from "$lib/components/tables/PiscicoleTable.svelte";
  import Button from "$lib/components/ui/Button.svelte";
  import Modal from "$lib/components/ui/Modal.svelte";
  import PiscicoleForm from "$lib/components/forms/PiscicoleForm.svelte";
  import ConfirmDialog from "$lib/components/ui/ConfirmDialog.svelte";

  let piscicoles: any[] = [];
  let loading = true;
  let showModal = false;
  let showDeleteConfirm = false;
  let selectedPiscicole: any = null;
  let isEdit = false;
  let enclosOptions: any[] = [];

  onMount(async () => {
    await loadData();
    await loadEnclos();
  });

  async function loadData() {
    loading = true;
    try {
      const response = await piscicolesApi.getPiscicoles({ limit: 100 });
      piscicoles = response.items;
    } catch (error) {
      console.error("Failed to load piscicoles:", error);
    } finally {
      loading = false;
    }
  }

  async function loadEnclos() {
    try {
      const { enclosApi } = await import("$lib/api/enclos");
      const response = await enclosApi.getEnclos({ limit: 100 });
      enclosOptions = response.items
        .filter((e: any) => e.type === "bassin")
        .map((e: any) => ({ value: e.id, label: e.name }));
    } catch (error) {
      console.error("Failed to load enclos:", error);
    }
  }

  function handleAdd() {
    selectedPiscicole = null;
    isEdit = false;
    showModal = true;
  }

  function handleEdit(piscicole: any) {
    selectedPiscicole = piscicole;
    isEdit = true;
    showModal = true;
  }

  function handleDelete(piscicole: any) {
    selectedPiscicole = piscicole;
    showDeleteConfirm = true;
  }

  function handleView(piscicole: any) {
    window.location.href = `/piscicoles/${piscicole.id}`;
  }

  async function handleSubmit(formData: any) {
    try {
      if (isEdit && selectedPiscicole) {
        await piscicolesApi.updatePiscicole(selectedPiscicole.id, formData);
      } else {
        await piscicolesApi.createPiscicole(formData);
      }
      showModal = false;
      await loadData();
    } catch (error) {
      console.error("Failed to save piscicole:", error);
    }
  }

  async function handleConfirmDelete() {
    if (selectedPiscicole) {
      try {
        await piscicolesApi.deletePiscicole(selectedPiscicole.id);
        showDeleteConfirm = false;
        await loadData();
      } catch (error) {
        console.error("Failed to delete piscicole:", error);
      }
    }
  }
</script>

<div class="space-y-6">
  <div class="flex justify-between items-center">
    <div>
      <h1 class="text-2xl font-bold text-gray-900">Gestion des piscicoles</h1>
      <p class="text-sm text-gray-500 mt-1">Gérez votre élevage piscicole</p>
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
      Nouveau piscicole
    </Button>
  </div>

  <PiscicoleTable
    {piscicoles}
    {loading}
    on:view={handleView}
    on:edit={handleEdit}
    on:delete={handleDelete}
    on:add={handleAdd}
  />

  <Modal
    open={showModal}
    title={isEdit ? "Modifier le piscicole" : "Ajouter un piscicole"}
    on:close={() => (showModal = false)}
    size="lg"
  >
    <PiscicoleForm
      formData={selectedPiscicole || {
        identification: "",
        race: "",
        sexe: "male",
        date_arrivee: new Date().toISOString().split("T")[0],
        enclos_id: 0,
        statut: "vivant",
        production_viande: false,
        production_reproduction: false,
      }}
      {enclosOptions}
      loading={false}
      {isEdit}
      on:submit={handleSubmit}
      on:cancel={() => (showModal = false)}
    />
  </Modal>

  <ConfirmDialog
    open={showDeleteConfirm}
    title="Supprimer le piscicole"
    message={`Êtes-vous sûr de vouloir supprimer ${selectedPiscicole?.identification} ? Cette action est irréversible.`}
    confirmVariant="danger"
    on:confirm={handleConfirmDelete}
    on:cancel={() => (showDeleteConfirm = false)}
  />
</div>
