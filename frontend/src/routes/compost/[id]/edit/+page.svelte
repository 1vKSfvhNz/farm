<!-- src/routes/compost/[id]/edit/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { compostApi } from "$lib/api/compost";
    import { permissionsStore } from "$lib/stores/permissions";
    import CompostForm from "$lib/components/forms/CompostForm.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";
    import Card from "$lib/components/ui/Card.svelte";

    let id: number;
    let compost: any = null;
    let loading = true;
    let saving = false;

    const canEdit = permissionsStore.hasPermission("can_edit_compost");

    onMount(async () => {
        const path = window.location.pathname;
        const match = path.match(/\/compost\/(\d+)\/edit/);
        if (match) {
            id = parseInt(match[1]);
            await loadData();
        }
    });

    async function loadData() {
        loading = true;
        try {
            compost = await compostApi.getCompost(id);
        } catch (error) {
            console.error("Failed to load compost:", error);
        } finally {
            loading = false;
        }
    }

    async function handleSubmit(formData: any) {
        saving = true;
        try {
            await compostApi.updateCompost(id, formData);
            window.location.href = `/compost/${id}`;
        } catch (error) {
            console.error("Failed to update compost:", error);
        } finally {
            saving = false;
        }
    }

    if (!canEdit) {
        window.location.href = "/compost";
    }
</script>

<div class="max-w-3xl mx-auto">
    {#if loading}
        <div class="flex justify-center items-center h-64">
            <Spinner size="lg" />
        </div>
    {:else if compost}
        <Card title="Modifier le compost">
            <CompostForm
                formData={compost}
                loading={saving}
                isEdit={true}
                on:submit={handleSubmit}
                on:cancel={() => (window.location.href = `/compost/${id}`)}
            />
        </Card>
    {/if}
</div>
