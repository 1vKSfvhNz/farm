<!-- src/routes/piscicoles/[id]/edit/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { piscicolesApi } from "$lib/api/piscicoles";
    import { enclosApi } from "$lib/api/enclos";
    import { permissionsStore } from "$lib/stores/permissions";
    import PiscicoleForm from "$lib/components/forms/PiscicoleForm.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";
    import Card from "$lib/components/ui/Card.svelte";

    let id: number;
    let piscicole: any = null;
    let enclosOptions: any[] = [];
    let loading = true;
    let saving = false;

    const canEdit = permissionsStore.canEditEspece("piscicoles");

    onMount(async () => {
        const path = window.location.pathname;
        const match = path.match(/\/piscicoles\/(\d+)\/edit/);
        if (match) {
            id = parseInt(match[1]);
            await loadData();
            await loadEnclos();
        }
    });

    async function loadData() {
        loading = true;
        try {
            piscicole = await piscicolesApi.getPiscicole(id);
        } catch (error) {
            console.error("Failed to load piscicole:", error);
        } finally {
            loading = false;
        }
    }

    async function loadEnclos() {
        try {
            const response = await enclosApi.getEnclos({ limit: 100 });
            enclosOptions = response.items
                .filter((e: any) => e.type === "bassin")
                .map((e: any) => ({ value: e.id, label: e.name }));
        } catch (error) {
            console.error("Failed to load enclos:", error);
        }
    }

    async function handleSubmit(formData: any) {
        saving = true;
        try {
            await piscicolesApi.updatePiscicole(id, formData);
            window.location.href = `/piscicoles/${id}`;
        } catch (error) {
            console.error("Failed to update piscicole:", error);
        } finally {
            saving = false;
        }
    }

    if (!canEdit) {
        window.location.href = "/piscicoles";
    }
</script>

<div class="max-w-3xl mx-auto">
    {#if loading}
        <div class="flex justify-center items-center h-64">
            <Spinner size="lg" />
        </div>
    {:else if piscicole}
        <Card title="Modifier le piscicole">
            <PiscicoleForm
                formData={piscicole}
                {enclosOptions}
                loading={saving}
                isEdit={true}
                on:submit={handleSubmit}
                on:cancel={() => (window.location.href = `/piscicoles/${id}`)}
            />
        </Card>
    {/if}
</div>
