<!-- src/routes/alertes/+page.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { alertStore } from "$stores/alerts";
    import { permissionsStore } from "$stores/permissions";
    import AlertBadge from "$components/ui/AlertBadge.svelte";
    import Button from "$components/ui/Button.svelte";
    import Pagination from "$components/ui/Pagination.svelte";
    import Card from "$components/ui/Card.svelte";
    import Select from "$components/ui/Select.svelte";

    let alerts: any[] = [];
    let loading = true;
    let currentPage = 1;
    let itemsPerPage = 20;
    let total = 0;
    let filterNiveau: string = "";
    let filterLu: string = "";
    let isAdmin = false;

    const niveauOptions = [
        { value: "", label: "Tous" },
        { value: "info", label: "Info" },
        { value: "warning", label: "Attention" },
        { value: "critical", label: "Critique" },
    ];

    const luOptions = [
        { value: "", label: "Tous" },
        { value: "false", label: "Non lus" },
        { value: "true", label: "Lus" },
    ];

    onMount(async () => {
        isAdmin = permissionsStore.isAdmin();
        await loadAlerts();
    });

    async function loadAlerts() {
        loading = true;
        try {
            await alertStore.loadAlerts();
            const state = await new Promise<any>((resolve) => {
                const unsubscribe = alertStore.subscribe((s) => {
                    unsubscribe();
                    resolve(s);
                });
            });
            alerts = state.alerts;
            total = state.pagination.total;
        } catch (error) {
            console.error("Failed to load alerts:", error);
        } finally {
            loading = false;
        }
    }

    function formatDate(dateStr: string): string {
        return new Date(dateStr).toLocaleString("fr-FR");
    }

    async function handleMarkAsRead(id: number) {
        await alertStore.markAsRead(id);
        await loadAlerts();
    }

    async function handleMarkAllAsRead() {
        await alertStore.markAllAsRead();
        await loadAlerts();
    }

    async function handleResolveAlert(id: number) {
        if (isAdmin) {
            await alertStore.resolveAlert(id);
            await loadAlerts();
        }
    }
</script>

<div class="space-y-6">
    <div class="flex justify-between items-center">
        <div>
            <h1 class="text-2xl font-bold text-gray-900">Alertes</h1>
            <p class="text-sm text-gray-500 mt-1">
                Historique des alertes système
            </p>
        </div>
        <Button on:click={handleMarkAllAsRead} variant="outline" size="sm">
            Tout marquer comme lu
        </Button>
    </div>

    <!-- Filtres -->
    <Card>
        <div class="flex flex-wrap gap-4">
            <Select
                label="Niveau"
                bind:value={filterNiveau}
                options={niveauOptions}
                on:change={loadAlerts}
                className="w-40"
            />
            <Select
                label="Statut"
                bind:value={filterLu}
                options={luOptions}
                on:change={loadAlerts}
                className="w-40"
            />
        </div>
    </Card>

    <!-- Liste des alertes -->
    <Card>
        {#if loading}
            <div class="flex justify-center items-center py-12">
                <div
                    class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"
                ></div>
            </div>
        {:else if alerts.length === 0}
            <div class="text-center py-12">
                <svg
                    class="w-12 h-12 text-gray-300 mx-auto mb-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                </svg>
                <p class="text-gray-500">Aucune alerte</p>
            </div>
        {:else}
            <div class="divide-y divide-gray-100">
                {#each alerts as alert}
                    <div class="p-4 hover:bg-gray-50 transition-colors">
                        <div class="flex items-start gap-3">
                            <AlertBadge niveau={alert.niveau} size="sm" />
                            <div class="flex-1 min-w-0">
                                <p class="text-sm text-gray-800">
                                    {alert.message}
                                </p>
                                <div
                                    class="flex flex-wrap items-center gap-3 mt-1"
                                >
                                    <span class="text-xs text-gray-400"
                                        >{formatDate(alert.created_at)}</span
                                    >
                                    {#if alert.entite_type}
                                        <span class="text-xs text-gray-400"
                                            >• {alert.entite_type}</span
                                        >
                                    {/if}
                                    {#if alert.est_traitee}
                                        <span class="text-xs text-green-600"
                                            >✓ Traitée</span
                                        >
                                    {/if}
                                </div>
                            </div>
                            <div class="flex items-center gap-2">
                                {#if !alert.est_lue}
                                    <button
                                        on:click={() =>
                                            handleMarkAsRead(alert.id)}
                                        class="text-xs text-primary-600 hover:text-primary-700 transition-colors"
                                    >
                                        Marquer lu
                                    </button>
                                {/if}
                                {#if isAdmin && !alert.est_traitee}
                                    <button
                                        on:click={() =>
                                            handleResolveAlert(alert.id)}
                                        class="text-xs text-green-600 hover:text-green-700 transition-colors"
                                    >
                                        Traiter
                                    </button>
                                {/if}
                            </div>
                        </div>
                    </div>
                {/each}
            </div>

            <div class="border-t border-gray-200 pt-4 mt-2">
                <Pagination
                    {currentPage}
                    totalPages={Math.ceil(total / itemsPerPage)}
                    on:pageChange={(e) => {
                        currentPage = e.detail.page;
                        loadAlerts();
                    }}
                />
            </div>
        {/if}
    </Card>
</div>
