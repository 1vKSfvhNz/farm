<!-- src/routes/settings/+page.svelte -->
<script lang="ts">
  import { uiStore } from "$lib/stores/ui";
  import { permissionsStore } from "$lib/stores/permissions";
  import Card from "$lib/components/ui/Card.svelte";
  import { authStore } from "$lib/stores/auth";

  let notifications = true;
  let sound = false;
  let autoRefresh = true;
  let theme = "system";
  let isAdmin = false;

  // S'abonner aux changements de permissions
  permissionsStore.subscribe((state) => {
    isAdmin = state.role === "admin";
  });

  $: uiStore.setNotificationsEnabled(notifications);
  $: uiStore.setSoundEnabled(sound);
  $: uiStore.setAutoRefresh(autoRefresh);
  $: uiStore.setTheme(theme as "light" | "dark" | "system");

  async function handleLogout() {
    await authStore.logout();
  }
</script>

<div class="space-y-6">
  <h1 class="text-2xl font-bold text-gray-900">Paramètres</h1>

  <Card title="Préférences générales">
    <div class="space-y-4">
      <div class="flex items-center justify-between py-2">
        <div>
          <p class="font-medium text-gray-800">Notifications</p>
          <p class="text-sm text-gray-500">
            Recevoir des alertes et notifications
          </p>
        </div>
        <label class="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            bind:checked={notifications}
            class="sr-only peer"
          />
          <div
            class="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"
          ></div>
        </label>
      </div>

      <div class="flex items-center justify-between py-2">
        <div>
          <p class="font-medium text-gray-800">Sons</p>
          <p class="text-sm text-gray-500">Jouer des sons pour les alertes</p>
        </div>
        <label class="relative inline-flex items-center cursor-pointer">
          <input type="checkbox" bind:checked={sound} class="sr-only peer" />
          <div
            class="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"
          ></div>
        </label>
      </div>

      <div class="flex items-center justify-between py-2">
        <div>
          <p class="font-medium text-gray-800">Rafraîchissement automatique</p>
          <p class="text-sm text-gray-500">
            Actualiser automatiquement les données
          </p>
        </div>
        <label class="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            bind:checked={autoRefresh}
            class="sr-only peer"
          />
          <div
            class="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"
          ></div>
        </label>
      </div>

      <div class="flex items-center justify-between py-2">
        <div>
          <p class="font-medium text-gray-800">Thème</p>
          <p class="text-sm text-gray-500">Apparence de l'application</p>
        </div>
        <select
          bind:value={theme}
          class="rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
        >
          <option value="light">Clair</option>
          <option value="dark">Sombre</option>
          <option value="system">Système</option>
        </select>
      </div>
    </div>
  </Card>

  {#if isAdmin}
    <Card title="Administration">
      <div class="space-y-3">
        <button
          on:click={() => (window.location.href = "/utilisateurs")}
          class="w-full text-left px-4 py-2 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors flex items-center gap-3"
        >
          <svg
            class="w-5 h-5 text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M12 4.5v15m7.5-7.5h-15"
            />
          </svg>
          Gérer les utilisateurs
        </button>
        <button
          on:click={() => (window.location.href = "/systeme")}
          class="w-full text-left px-4 py-2 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors flex items-center gap-3"
        >
          <svg
            class="w-5 h-5 text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75"
            />
          </svg>
          Paramètres système
        </button>
      </div>
    </Card>
  {/if}

  <Card title="Sécurité">
    <div class="space-y-3">
      <button
        on:click={() => (window.location.href = "/profile")}
        class="w-full text-left px-4 py-2 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors flex items-center gap-3"
      >
        <svg
          class="w-5 h-5 text-gray-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
          />
        </svg>
        Mon profil
      </button>
      <button
        on:click={handleLogout}
        class="w-full text-left px-4 py-2 rounded-lg bg-red-50 hover:bg-red-100 transition-colors flex items-center gap-3 text-red-700"
      >
        <svg
          class="w-5 h-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
          />
        </svg>
        Déconnexion
      </button>
    </div>
  </Card>

  <Card title="À propos">
    <div class="space-y-2 text-sm text-gray-600">
      <p><span class="font-medium">Version:</span> 2.0.0</p>
      <p>
        <span class="font-medium">Farm Manager</span> - Application de gestion d'élevage
        intelligent
      </p>
      <p>
        Fonctionnalités: Suivi des animaux, gestion des enclos, comptabilité,
        prédictions IA, mode expérimental
      </p>
      <hr class="my-2" />
      <p class="text-xs text-gray-400">
        © 2024 Farm Manager - Tous droits réservés
      </p>
    </div>
  </Card>
</div>
