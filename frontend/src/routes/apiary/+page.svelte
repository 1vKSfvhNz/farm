<!-- src/routes/apiary/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { apiaryApi } from "$lib/api/apiary";
    import ApicultureTable from "$components/tables/ApicultureTable.svelte";
    import Button from "$components/ui/Button.svelte";
    import Modal from "$components/ui/Modal.svelte";
    import ApiaryForm from "$components/forms/ApiaryForm.svelte";
    import RecolteMielForm from "$components/forms/RecolteMielForm.svelte";
    import ConfirmDialog from "$components/ui/ConfirmDialog.svelte";

    let ruches: any[] = [];
    let loading = true;
    let showModal = false;
    let showRecolteModal = false;
    let showDeleteConfirm = false;
    let selectedRuche: any = null;
    let isEdit = false;

    onMount(async () => {
        await loadData();
    });

    async function loadData() {
        loading = true;
        try {
            const response = await apiaryApi.getRuches({ limit: 100 });
            ruches = response.items;
        } catch (error) {
            console.error("Failed to load ruches:", error);
        } finally {
            loading = false;
        }
    }

    function handleAdd() {
        selectedRuche = null;
        isEdit = false;
        showModal = true;
    }

    function handleEdit(ruche: any) {
        selectedRuche = ruche;
        isEdit = true;
        showModal = true;
    }

    function handleDelete(ruche: any) {
        selectedRuche = ruche;
        showDeleteConfirm = true;
    }

    function handleView(ruche: any) {
        window.location.href = `/apiculture/${ruche.id}`;
    }

    function handleAddRecolte(ruche: any) {
        selectedRuche = ruche;
        showRecolteModal = true;
    }

    async function handleSubmit(formData: any) {
        try {
            if (isEdit && selectedRuche) {
                await apiaryApi.updateRuche(selectedRuche.id, formData);
            } else {
                await apiaryApi.createRuche(formData);
            }
            showModal = false;
            await loadData();
        } catch (error) {
            console.error("Failed to save ruche:", error);
        }
    }

    async function handleSubmitRecolte(formData: any) {
        if (selectedRuche) {
            try {
                await apiaryApi.addRecolte(selectedRuche.id, formData);
                showRecolteModal = false;
                await loadData();
            } catch (error) {
                console.error("Failed to add recolte:", error);
            }
        }
    }

    async function handleConfirmDelete() {
        if (selectedRuche) {
            try {
                await apiaryApi.deleteRuche(selectedRuche.id);
                showDeleteConfirm = false;
                await loadData();
            } catch (error) {
                console.error("Failed to delete ruche:", error);
            }
        }
    }
</script>

<div class="space-y-6">
    <div class="flex justify-between items-center">
        <div>
            <h1 class="text-2xl font-bold text-gray-900">Gestion apicole</h1>
            <p class="text-sm text-gray-500 mt-1">
                Gérez vos ruches et votre production de miel
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
            Nouvelle ruche
        </Button>
    </div>

    <ApicultureTable
        {ruches}
        {loading}
        on:view={handleView}
        on:edit={handleEdit}
        on:delete={handleDelete}
        on:addRecolte={handleAddRecolte}
        on:add={handleAdd}
    />

    <Modal
        open={showModal}
        title={isEdit ? "Modifier la ruche" : "Ajouter une ruche"}
        on:close={() => (showModal = false)}
        size="lg"
    >
        <ApiaryForm
            formData={selectedRuche || {
                identification: "",
                date_installation: new Date().toISOString().split("T")[0],
                statut: "active",
            }}
            loading={false}
            {isEdit}
            on:submit={handleSubmit}
            on:cancel={() => (showModal = false)}
        />
    </Modal>

    <Modal
        open={showRecolteModal}
        title="Ajouter une récolte de miel"
        on:close={() => (showRecolteModal = false)}
        size="md"
    >
        <RecolteMielForm
            formData={{
                ruche_id: selectedRuche?.id || 0,
                date_recolte: new Date().toISOString().split("T")[0],
                poids_kg: 0,
            }}
            rucheOptions={ruches.map((r) => ({
                value: r.id,
                label: r.identification,
            }))}
            loading={false}
            on:submit={handleSubmitRecolte}
            on:cancel={() => (showRecolteModal = false)}
        />
    </Modal>

    <ConfirmDialog
        open={showDeleteConfirm}
        title="Supprimer la ruche"
        message={`Êtes-vous sûr de vouloir supprimer ${selectedRuche?.identification} ? Cette action est irréversible.`}
        confirmVariant="danger"
        on:confirm={handleConfirmDelete}
        on:cancel={() => (showDeleteConfirm = false)}
    />
</div>
