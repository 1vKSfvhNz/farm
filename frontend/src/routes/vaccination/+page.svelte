<!-- src/routes/vaccinations/+page.svelte -->
<script lang="ts">
  import { onMount } from "svelte";
  import { vaccinationApi } from "$lib/api/vaccination";
  import VaccinationTable from "$lib/components/tables/VaccinationTable.svelte";
  import Button from "$lib/components/ui/Button.svelte";
  import Modal from "$lib/components/ui/Modal.svelte";
  import VaccinationForm from "$lib/components/forms/VaccinationForm.svelte";
  import ConfirmDialog from "$lib/components/ui/ConfirmDialog.svelte";
  import { bovinsApi } from "$lib/api/bovins";

  let vaccinations: any[] = [];
  let loading = true;
  let showModal = false;
  let showDeleteConfirm = false;
  let selectedVaccination: any = null;
  let isEdit = false;
  let animalOptions: any[] = [];
  let maladieOptions: any[] = [];
  let vaccinOptions: any[] = [];

  onMount(async () => {
    await loadData();
    await loadOptions();
  });

  async function loadData() {
    loading = true;
    try {
      const response = await vaccinationApi.getVaccinations({ limit: 100 });
      vaccinations = response.items;
    } catch (error) {
      console.error("Failed to load vaccinations:", error);
    } finally {
      loading = false;
    }
  }

  async function loadOptions() {
    try {
      const [maladies, vaccins, animaux] = await Promise.all([
        vaccinationApi.getMaladies(),
        vaccinationApi.getVaccins(),
        bovinsApi.getBovins({ limit: 1000 }),
      ]);
      maladieOptions = maladies.map((m: any) => ({
        value: m.id,
        label: m.nom,
      }));
      vaccinOptions = vaccins.map((v: any) => ({ value: v.id, label: v.nom }));
      animalOptions = animaux.items.map((a: any) => ({
        value: a.id,
        label: a.identification,
      }));
    } catch (error) {
      console.error("Failed to load options:", error);
    }
  }

  function handleAdd() {
    selectedVaccination = null;
    isEdit = false;
    showModal = true;
  }

  function handleEdit(vaccination: any) {
    selectedVaccination = vaccination;
    isEdit = true;
    showModal = true;
  }

  function handleDelete(vaccination: any) {
    selectedVaccination = vaccination;
    showDeleteConfirm = true;
  }

  function handleView(vaccination: any) {
    window.location.href = `/vaccinations/${vaccination.id}`;
  }

  async function handleRealize(vaccination: any) {
    try {
      await vaccinationApi.realizeVaccination(vaccination.id);
      await loadData();
    } catch (error) {
      console.error("Failed to realize vaccination:", error);
    }
  }

  async function handleSubmit(formData: any) {
    try {
      if (isEdit && selectedVaccination) {
        await vaccinationApi.updateVaccination(
          selectedVaccination.id,
          formData,
        );
      } else {
        await vaccinationApi.createVaccination(formData);
      }
      showModal = false;
      await loadData();
    } catch (error) {
      console.error("Failed to save vaccination:", error);
    }
  }

  async function handleConfirmDelete() {
    if (selectedVaccination) {
      try {
        await vaccinationApi.deleteVaccination(selectedVaccination.id);
        showDeleteConfirm = false;
        await loadData();
      } catch (error) {
        console.error("Failed to delete vaccination:", error);
      }
    }
  }
</script>

<div class="space-y-6">
  <div class="flex justify-between items-center">
    <div>
      <h1 class="text-2xl font-bold text-gray-900">Vaccinations</h1>
      <p class="text-sm text-gray-500 mt-1">
        Gérez le calendrier vaccinal de vos animaux
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
      Nouvelle vaccination
    </Button>
  </div>

  <VaccinationTable
    {vaccinations}
    {loading}
    on:view={handleView}
    on:edit={handleEdit}
    on:delete={handleDelete}
    on:realize={handleRealize}
    on:add={handleAdd}
  />

  <Modal
    open={showModal}
    title={isEdit ? "Modifier la vaccination" : "Ajouter une vaccination"}
    on:close={() => (showModal = false)}
    size="lg"
  >
    <VaccinationForm
      formData={selectedVaccination || {
        animal_id: 0,
        maladie_id: 0,
        date_prevue: new Date().toISOString().split("T")[0],
        rappel_necessaire: false,
      }}
      {maladieOptions}
      {vaccinOptions}
      loading={false}
      {isEdit}
      on:submit={handleSubmit}
      on:cancel={() => (showModal = false)}
    />
  </Modal>

  <ConfirmDialog
    open={showDeleteConfirm}
    title="Supprimer la vaccination"
    message="Êtes-vous sûr de vouloir supprimer cette vaccination ? Cette action est irréversible."
    confirmVariant="danger"
    on:confirm={handleConfirmDelete}
    on:cancel={() => (showDeleteConfirm = false)}
  />
</div>
