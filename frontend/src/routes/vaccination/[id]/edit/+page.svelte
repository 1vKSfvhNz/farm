<!-- src/routes/vaccinations/[id]/edit/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { vaccinationApi } from "$lib/api/vaccination";
    import { bovinsApi } from "$lib/api/bovins";
    import { permissionsStore } from "$lib/stores/permissions";
    import VaccinationForm from "$lib/components/forms/VaccinationForm.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";
    import Card from "$lib/components/ui/Card.svelte";

    let id: number;
    let vaccination: any = null;
    let maladieOptions: any[] = [];
    let vaccinOptions: any[] = [];
    let loading = true;
    let saving = false;

    const canEdit = permissionsStore.hasPermission("can_edit_vaccinations");

    onMount(async () => {
        const path = window.location.pathname;
        const match = path.match(/\/vaccinations\/(\d+)\/edit/);
        if (match) {
            id = parseInt(match[1]);
            await loadData();
            await loadOptions();
        }
    });

    async function loadData() {
        loading = true;
        try {
            vaccination = await vaccinationApi.getVaccination(id);
        } catch (error) {
            console.error("Failed to load vaccination:", error);
        } finally {
            loading = false;
        }
    }

    async function loadOptions() {
        try {
            const [maladies, vaccins] = await Promise.all([
                vaccinationApi.getMaladies(),
                vaccinationApi.getVaccins(),
                bovinsApi.getBovins({ limit: 1000 }),
            ]);
            maladieOptions = maladies.map((m: any) => ({
                value: m.id,
                label: m.nom,
            }));
            vaccinOptions = vaccins.map((v: any) => ({
                value: v.id,
                label: v.nom,
            }));
        } catch (error) {
            console.error("Failed to load options:", error);
        }
    }

    async function handleSubmit(formData: any) {
        saving = true;
        try {
            await vaccinationApi.updateVaccination(id, formData);
            window.location.href = `/vaccinations`;
        } catch (error) {
            console.error("Failed to update vaccination:", error);
        } finally {
            saving = false;
        }
    }

    if (!canEdit) {
        window.location.href = "/vaccinations";
    }
</script>

<div class="max-w-3xl mx-auto">
    {#if loading}
        <div class="flex justify-center items-center h-64">
            <Spinner size="lg" />
        </div>
    {:else if vaccination}
        <Card title="Modifier la vaccination">
            <VaccinationForm
                formData={vaccination}
                {maladieOptions}
                {vaccinOptions}
                loading={saving}
                isEdit={true}
                on:submit={handleSubmit}
                on:cancel={() => (window.location.href = "/vaccinations")}
            />
        </Card>
    {/if}
</div>
