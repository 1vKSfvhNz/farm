<!-- lib/components/dashboard/AlertList.svelte -->
<script lang="ts">
  import { onMount } from "svelte";
  import { alertStore } from "../../stores/alerts";
  import AlertBadge from "../../components/ui/AlertBadge.svelte";

  export let limit: number = 5;
  export let showViewAll: boolean = true;

  type AlertLevel = "info" | "warning" | "critical";
  function getAlertLevel(level: string): AlertLevel {
    return level as AlertLevel;
  }

  let alerts: Array<{
    id: number;
    message: string;
    niveau: string;
    est_lue: boolean;
    created_at: string;
  }> = [];

  onMount(() => {
    (async () => {
      await alertStore.loadAlerts();
    })();
    const unsubscribe = alertStore.subscribe((state: { alerts: any[] }) => {
      alerts = state.alerts.slice(0, limit);
    });
    return unsubscribe;
  });

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "À l'instant";
    if (diffMins < 60) return `Il y a ${diffMins} min`;
    if (diffHours < 24) return `Il y a ${diffHours} h`;
    if (diffDays === 1) return "Hier";
    return date.toLocaleDateString("fr-FR");
  }

  function markAsRead(id: number) {
    alertStore.markAsRead(id);
  }
</script>

<div class="bg-white rounded-xl border border-gray-200 shadow-sm">
  <div class="p-5 border-b border-gray-200">
    <div class="flex items-center justify-between">
      <h3 class="text-lg font-semibold text-gray-900">Alertes récentes</h3>
      {#if showViewAll}
        <a
          href="/alertes"
          class="text-sm text-primary-600 hover:text-primary-700 transition-colors"
        >
          Voir tout
        </a>
      {/if}
    </div>
  </div>

  <div class="divide-y divide-gray-100">
    {#if alerts.length === 0}
      <div class="p-8 text-center">
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
      {#each alerts as alert}
        <div class="p-4 hover:bg-gray-50 transition-colors">
          <div class="flex items-start gap-3">
            <AlertBadge niveau={getAlertLevel(alert.niveau)} size="sm" />
            <div class="flex-1 min-w-0">
              <p class="text-sm text-gray-800">{alert.message}</p>
              <div class="flex items-center gap-2 mt-1">
                <span class="text-xs text-gray-400"
                  >{formatDate(alert.created_at)}</span
                >
                {#if !alert.est_lue}
                  <span class="w-1.5 h-1.5 bg-primary-500 rounded-full"></span>
                  <span class="text-xs text-primary-600">Nouveau</span>
                {/if}
              </div>
            </div>
            {#if !alert.est_lue}
              <button
                on:click={() => markAsRead(alert.id)}
                class="text-xs text-gray-400 hover:text-primary-600 transition-colors"
              >
                Marquer lu
              </button>
            {/if}
          </div>
        </div>
      {/each}
    {/if}
  </div>
</div>
