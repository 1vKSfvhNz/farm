<!-- src/routes/avicoles/[id]/edit/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { avicolesApi } from "$lib/api/avicoles";
    import { enclosApi } from "$lib/api/enclos";
    import { permissionsStore } from "$lib/stores/permissions";
    import AvicoleForm from "$lib/components/forms/AvicoleForm.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";
    import Card from "$lib/components/ui/Card.svelte";

    let id: number;
    let avicole: any = null;
    let enclosOptions: any[] = [];
    let loading = true;
    let saving = false;

    const canEdit = permissionsStore.canEditEspece("avicoles");

    onMount(async () => {
        const path = window.location.pathname;
        const match = path.match(/\/avicoles\/(\d+)\/edit/);
        if (match) {
            id = parseInt(match[1]);
            await loadData();
            await loadEnclos();
        }
    });

    async function loadData() {
        loading = true;
        try {
            avicole = await avicolesApi.getAvicole(id);
        } catch (error) {
            console.error("Failed to load avicole:", error);
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

    async function handleSubmit(formData: any) {
        saving = true;
        try {
            await avicolesApi.updateAvicole(id, formData);
            window.location.href = `/avicoles/${id}`;
        } catch (error) {
            console.error("Failed to update avicole:", error);
        } finally {
            saving = false;
        }
    }

    if (!canEdit) {
        window.location.href = "/avicoles";
    }
</script>

<div class="max-w-3xl mx-auto">
    {#if loading}
        <div class="flex justify-center items-center h-64">
            <Spinner size="lg" />
        </div>
    {:else if avicole}
        <Card title="Modifier l'avicole">
            <AvicoleForm
                formData={avicole}
                {enclosOptions}
                loading={saving}
                isEdit={true}
                on:submit={handleSubmit}
                on:cancel={() => (window.location.href = `/avicoles/${id}`)}
            />
        </Card>
    {/if}
</div>
