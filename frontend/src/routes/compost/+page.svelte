<!-- src/routes/compost/+page.svelte -->
<script lang="ts">
  import { onMount } from "svelte";
  import { compostApi } from "$lib/api/compost";
  import CompostTable from "$lib/components/tables/CompostTable.svelte";
  import Button from "$lib/components/ui/Button.svelte";
  import Modal from "$lib/components/ui/Modal.svelte";
  import CompostForm from "$lib/components/forms/CompostForm.svelte";
  import ConfirmDialog from "$lib/components/ui/ConfirmDialog.svelte";
  import Input from "$lib/components/ui/Input.svelte";

  let composts: any[] = [];
  let loading = true;
  let showModal = false;
  let showMatureModal = false;
  let showDeleteConfirm = false;
  let selectedCompost: any = null;
  let volumeFinal = 0;
  let isEdit = false;

  onMount(async () => {
    await loadData();
  });

  async function loadData() {
    loading = true;
    try {
      const response = await compostApi.getComposts({ limit: 100 });
      composts = response.items;
    } catch (error) {
      console.error("Failed to load composts:", error);
    } finally {
      loading = false;
    }
  }

  function handleAdd() {
    selectedCompost = null;
    isEdit = false;
    showModal = true;
  }

  function handleEdit(compost: any) {
    selectedCompost = compost;
    isEdit = true;
    showModal = true;
  }

  function handleDelete(compost: any) {
    selectedCompost = compost;
    showDeleteConfirm = true;
  }

  function handleView(compost: any) {
    window.location.href = `/compost/${compost.id}`;
  }

  function handleMarkMature(compost: any) {
    selectedCompost = compost;
    volumeFinal = compost.volume_initial || 0;
    showMatureModal = true;
  }

  async function handleSubmit(formData: any) {
    try {
      if (isEdit && selectedCompost) {
        await compostApi.updateCompost(selectedCompost.id, formData);
      } else {
        await compostApi.createCompost(formData);
      }
      showModal = false;
      await loadData();
    } catch (error) {
      console.error("Failed to save compost:", error);
    }
  }

  async function handleMarkAsMature() {
    if (selectedCompost) {
      try {
        await compostApi.markAsMature(selectedCompost.id, volumeFinal);
        showMatureModal = false;
        await loadData();
      } catch (error) {
        console.error("Failed to mark compost as mature:", error);
      }
    }
  }

  async function handleConfirmDelete() {
    if (selectedCompost) {
      try {
        await compostApi.deleteCompost(selectedCompost.id);
        showDeleteConfirm = false;
        await loadData();
      } catch (error) {
        console.error("Failed to delete compost:", error);
      }
    }
  }
</script>

<div class="space-y-6">
  <div class="flex justify-between items-center">
    <div>
      <h1 class="text-2xl font-bold text-gray-900">Gestion du compost</h1>
      <p class="text-sm text-gray-500 mt-1">
        Suivez vos tas de compost et leur maturation
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
      Nouveau compost
    </Button>
  </div>

  <CompostTable
    {composts}
    {loading}
    on:view={handleView}
    on:edit={handleEdit}
    on:delete={handleDelete}
    on:mature={handleMarkMature}
    on:add={handleAdd}
  />

  <Modal
    open={showModal}
    title={isEdit ? "Modifier le compost" : "Ajouter un compost"}
    on:close={() => (showModal = false)}
    size="lg"
  >
    <CompostForm
      formData={selectedCompost || {
        name: "",
        type: "mixte",
        date_demarrage: new Date().toISOString().split("T")[0],
        volume_initial: 0,
      }}
      loading={false}
      {isEdit}
      on:submit={handleSubmit}
      on:cancel={() => (showModal = false)}
    />
  </Modal>

  <Modal
    open={showMatureModal}
    title="Marquer comme mature"
    on:close={() => (showMatureModal = false)}
    size="md"
  >
    <div class="space-y-4">
      <p class="text-sm text-gray-600">
        Compost: <span class="font-medium">{selectedCompost?.name}</span>
      </p>
      <Input
        label="Volume final (m³)"
        bind:value={volumeFinal}
        inputType="number"
        required
        placeholder="0"
      />
      <div class="flex justify-end gap-3 pt-4">
        <Button on:click={() => (showMatureModal = false)} variant="outline">
          Annuler
        </Button>
        <Button on:click={handleMarkAsMature} variant="primary">Valider</Button>
      </div>
    </div>
  </Modal>

  <ConfirmDialog
    open={showDeleteConfirm}
    title="Supprimer le compost"
    message={`Êtes-vous sûr de vouloir supprimer ${selectedCompost?.name} ? Cette action est irréversible.`}
    confirmVariant="danger"
    on:confirm={handleConfirmDelete}
    on:cancel={() => (showDeleteConfirm = false)}
  />
</div>
