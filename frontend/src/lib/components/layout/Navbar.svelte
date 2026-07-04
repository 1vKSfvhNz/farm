<!-- lib/components/layout/Navbar.svelte -->
<script lang="ts">
  import { authStore } from "../../stores/auth";
  import { uiStore } from "../../stores/ui";
  import { notificationStore } from "../../stores/notifications";

  let showUserMenu = false;
  let showNotifications = false;

  function toggleSidebar() {
    uiStore.toggleSidebar();
  }

  async function handleLogout() {
    await authStore.logout();
  }
</script>

<nav
  class="fixed top-0 left-0 right-0 z-30 bg-white border-b border-gray-200 shadow-sm"
>
  <div class="px-4 h-16 flex items-center justify-between">
    <div class="flex items-center gap-4">
      <button
        on:click={toggleSidebar}
        class="p-2 rounded-lg hover:bg-gray-100 transition-colors"
      >
        <svg
          class="w-6 h-6 text-gray-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M4 6h16M4 12h16M4 18h16"
          />
        </svg>
      </button>

      <a href="/" class="flex items-center gap-2">
        <div
          class="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center"
        >
          <span class="text-white font-bold text-lg">F</span>
        </div>
        <span class="font-semibold text-gray-800 hidden sm:inline"
          >Farm Manager</span
        >
      </a>
    </div>

    <div class="flex items-center gap-2">
      <!-- Notifications -->
      <div class="relative">
        <button
          on:click={() => (showNotifications = !showNotifications)}
          class="p-2 rounded-lg hover:bg-gray-100 transition-colors relative"
        >
          <svg
            class="w-6 h-6 text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
            />
          </svg>
          {#if $notificationStore.unreadCount > 0}
            <span class="absolute top-1 right-1 w-3 h-3 bg-red-500 rounded-full"
            ></span>
          {/if}
        </button>

        {#if showNotifications}
          <div
            class="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden z-50"
          >
            <div class="p-3 border-b border-gray-200">
              <h3 class="font-semibold">Notifications</h3>
            </div>
            <div class="max-h-96 overflow-y-auto">
              {#if $notificationStore.notifications.length === 0}
                <p class="p-4 text-center text-gray-500">Aucune notification</p>
              {:else}
                {#each $notificationStore.notifications as notif}
                  <div
                    class="p-3 border-b border-gray-100 hover:bg-gray-50 transition-colors"
                  >
                    <p class="text-sm">{notif.message}</p>
                    <span class="text-xs text-gray-400"
                      >{new Date(notif.created_at).toLocaleString()}</span
                    >
                  </div>
                {/each}
              {/if}
            </div>
          </div>
        {/if}
      </div>

      <!-- User Menu -->
      <div class="relative">
        <button
          on:click={() => (showUserMenu = !showUserMenu)}
          class="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-100 transition-colors"
        >
          <div
            class="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center"
          >
            <span class="text-gray-600 font-medium">
              {$authStore.user?.full_name?.charAt(0) || "U"}
            </span>
          </div>
          <span class="hidden md:inline text-sm text-gray-700">
            {$authStore.user?.full_name}
          </span>
          <svg
            class="w-4 h-4 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </button>

        {#if showUserMenu}
          <div
            class="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden z-50"
          >
            <a
              href="/profile"
              class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors"
            >
              Mon profil
            </a>
            <a
              href="/settings"
              class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors"
            >
              Paramètres
            </a>
            <hr class="my-1" />
            <button
              on:click={handleLogout}
              class="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
            >
              Déconnexion
            </button>
          </div>
        {/if}
      </div>
    </div>
  </div>
</nav>
