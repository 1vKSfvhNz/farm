<!-- src/routes/entomoculture/[id]/edit/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { entomocultureApi } from "$lib/api/entomoculture";
    import { enclosApi } from "$lib/api/enclos";
    import { permissionsStore } from "$lib/stores/permissions";
    import EntomocultureForm from "$lib/components/forms/EntomocultureForm.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";
    import Card from "$lib/components/ui/Card.svelte";

    let id: number;
    let lot: any = null;
    let enclosOptions: any[] = [];
    let loading = true;
    let saving = false;

    const canEdit = permissionsStore.canEditEspece("entomoculture");

    onMount(async () => {
        const path = window.location.pathname;
        const match = path.match(/\/entomoculture\/(\d+)\/edit/);
        if (match) {
            id = parseInt(match[1]);
            await loadData();
            await loadEnclos();
        }
    });

    async function loadData() {
        loading = true;
        try {
            lot = await entomocultureApi.getLot(id);
        } catch (error) {
            console.error("Failed to load lot:", error);
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
            await entomocultureApi.updateLot(id, formData);
            window.location.href = `/entomoculture/${id}`;
        } catch (error) {
            console.error("Failed to update lot:", error);
        } finally {
            saving = false;
        }
    }

    if (!canEdit) {
        window.location.href = "/entomoculture";
    }
</script>

<div class="max-w-3xl mx-auto">
    {#if loading}
        <div class="flex justify-center items-center h-64">
            <Spinner size="lg" />
        </div>
    {:else if lot}
        <Card title="Modifier le lot">
            <EntomocultureForm
                formData={lot}
                {enclosOptions}
                loading={saving}
                isEdit={true}
                on:submit={handleSubmit}
                on:cancel={() =>
                    (window.location.href = `/entomoculture/${id}`)}
            />
        </Card>
    {/if}
</div>
