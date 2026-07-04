<!-- lib/components/layout/Sidebar.svelte -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<!-- svelte-ignore a11y-click-events-have-key-events -->
<script lang="ts">
  import { uiStore } from "$lib/stores/ui";
  import { permissionsStore } from "$lib/stores/permissions";
  import { page } from "$lib/stores/page";
  import { menuItems } from "$lib/types/roles";
  import { onMount } from "svelte";

  let isOpen = true;
  let activePath = "";
  let isMobile = false;
  let filteredMenu: any[] = [];
  let currentRole: string | null = null;

  function filterMenuByPermissions() {
    filteredMenu = menuItems.filter((item) => {
      // Toujours afficher le tableau de bord et les paramètres
      if (
        item.href === "/" ||
        item.href === "/parametres" ||
        item.href === "/alertes"
      ) {
        return true;
      }
      // Pour les autres menus, vérifier la permission
      if (item.requiredPermission) {
        return permissionsStore.hasPermission(item.requiredPermission as any);
      }
      return true;
    });
  }

  onMount(() => {
    const checkMobile = () => {
      isMobile = window.innerWidth < 768;
      if (isMobile) {
        isOpen = false;
      } else {
        isOpen = true;
      }
    };

    checkMobile();
    window.addEventListener("resize", checkMobile);

    // S'abonner aux changements de page
    const unsubscribePage = page.subscribe((state: { path: string }) => {
      activePath = state.path;
    });

    // S'abonner aux changements de permissions
    const unsubscribePermissions = permissionsStore.subscribe(() => {
      filterMenuByPermissions();
      currentRole = permissionsStore.getRole();
    });

    permissionsStore.init();
    filterMenuByPermissions();
    currentRole = permissionsStore.getRole();

    return () => {
      window.removeEventListener("resize", checkMobile);
      unsubscribePage();
      unsubscribePermissions();
      permissionsStore.cleanup();
    };
  });

  function toggleSidebar() {
    isOpen = !isOpen;
    uiStore.setSidebarOpen(isOpen);
  }

  function closeSidebar() {
    if (isMobile) {
      isOpen = false;
      uiStore.setSidebarOpen(false);
    }
  }

  function getRoleLabel(role: string | null): string {
    switch (role) {
      case "admin":
        return "Administrateur";
      case "veterinaire":
        return "Vétérinaire";
      case "responsable_enclos":
        return "Responsable enclos";
      case "technicien":
        return "Technicien";
      case "observateur":
        return "Observateur";
      default:
        return "Chargement...";
    }
  }
</script>

<!-- Overlay mobile -->
{#if isMobile && isOpen}
  <div
    class="fixed inset-0 bg-black/50 z-40 transition-opacity duration-300"
    on:click={closeSidebar}
  />
{/if}

<!-- Sidebar -->
<aside
  class="fixed top-16 left-0 h-full bg-gradient-to-b from-gray-900 to-gray-800 text-white z-40 transition-all duration-300 shadow-xl overflow-y-auto"
  class:translate-x-0={isOpen}
  class:-translate-x-full={!isOpen}
  style="width: 280px"
>
  <div class="flex flex-col h-full">
    <!-- Navigation principale -->
    <nav class="flex-1 py-4">
      {#each filteredMenu as item}
        <a
          href={item.href}
          on:click={closeSidebar}
          class={`flex items-center gap-3 px-4 py-3 mx-2 rounded-lg transition-all duration-200
                      ${activePath === item.href ? "bg-gray-700 text-white" : "text-gray-300 hover:bg-gray-700/50"}
                  `}
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
              d={item.icon}
            />
          </svg>
          <span class="text-sm font-medium">{item.label}</span>

          {#if activePath === item.href}
            <div class="ml-auto w-1 h-6 bg-primary-500 rounded-full"></div>
          {/if}
        </a>
      {/each}
    </nav>

    <!-- Footer sidebar -->
    <div class="p-4 border-t border-gray-700">
      <div class="flex items-center gap-3 px-2 py-2 rounded-lg bg-gray-800/50">
        <div class="flex-1">
          <p class="text-xs text-gray-400">Rôle</p>
          <p class="text-sm font-medium">{getRoleLabel(currentRole)}</p>
        </div>
        <button
          on:click={toggleSidebar}
          class="p-1 rounded-lg hover:bg-gray-700 transition-colors"
        >
          <svg
            class="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d={isOpen
                ? "M11 19l-7-7 7-7m8 14l-7-7 7-7"
                : "M13 5l7 7-7 7M5 5l7 7-7 7"}
            />
          </svg>
        </button>
      </div>
    </div>
  </div>
</aside>
