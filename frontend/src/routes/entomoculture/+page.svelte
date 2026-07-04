<!-- src/routes/entomoculture/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { entomocultureApi } from "$lib/api/entomoculture";
    import EntomocultureTable from "$lib/components/tables/EntomocultureTable.svelte";
    import Button from "$lib/components/ui/Button.svelte";
    import Modal from "$lib/components/ui/Modal.svelte";
    import EntomocultureForm from "$lib/components/forms/EntomocultureForm.svelte";
    import ConfirmDialog from "$lib/components/ui/ConfirmDialog.svelte";

    let lots: any[] = [];
    let loading = true;
    let showModal = false;
    let showDeleteConfirm = false;
    let selectedLot: any = null;
    let isEdit = false;
    let enclosOptions: any[] = [];

    onMount(async () => {
        await loadData();
        await loadEnclos();
    });

    async function loadData() {
        loading = true;
        try {
            const response = await entomocultureApi.getLots({ limit: 100 });
            lots = response.items;
        } catch (error) {
            console.error("Failed to load lots:", error);
        } finally {
            loading = false;
        }
    }

    async function loadEnclos() {
        try {
            const { enclosApi } = await import("$lib/api/enclos");
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
        selectedLot = null;
        isEdit = false;
        showModal = true;
    }

    function handleEdit(lot: any) {
        selectedLot = lot;
        isEdit = true;
        showModal = true;
    }

    function handleDelete(lot: any) {
        selectedLot = lot;
        showDeleteConfirm = true;
    }

    function handleView(lot: any) {
        window.location.href = `/entomoculture/${lot.id}`;
    }

    async function handleSubmit(formData: any) {
        try {
            if (isEdit && selectedLot) {
                await entomocultureApi.updateLot(selectedLot.id, formData);
            } else {
                await entomocultureApi.createLot(formData);
            }
            showModal = false;
            await loadData();
        } catch (error) {
            console.error("Failed to save lot:", error);
        }
    }

    async function handleConfirmDelete() {
        if (selectedLot) {
            try {
                await entomocultureApi.deleteLot(selectedLot.id);
                showDeleteConfirm = false;
                await loadData();
            } catch (error) {
                console.error("Failed to delete lot:", error);
            }
        }
    }
</script>

<div class="space-y-6">
    <div class="flex justify-between items-center">
        <div>
            <h1 class="text-2xl font-bold text-gray-900">Entomoculture</h1>
            <p class="text-sm text-gray-500 mt-1">
                Gérez vos élevages d'insectes
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
            Nouveau lot
        </Button>
    </div>

    <EntomocultureTable
        {lots}
        {loading}
        on:view={handleView}
        on:edit={handleEdit}
        on:delete={handleDelete}
        on:add={handleAdd}
    />

    <Modal
        open={showModal}
        title={isEdit ? "Modifier le lot" : "Ajouter un lot"}
        on:close={() => (showModal = false)}
        size="lg"
    >
        <EntomocultureForm
            formData={selectedLot || {
                identification: "",
                espece: "",
                stade_actuel: "larve",
                date_arrivee: new Date().toISOString().split("T")[0],
                type_production: "larves",
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
        title="Supprimer le lot"
        message={`Êtes-vous sûr de vouloir supprimer ${selectedLot?.identification} ? Cette action est irréversible.`}
        confirmVariant="danger"
        on:confirm={handleConfirmDelete}
        on:cancel={() => (showDeleteConfirm = false)}
    />
</div>
