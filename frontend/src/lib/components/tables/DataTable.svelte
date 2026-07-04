<!-- lib/components/tables/DataTable.svelte - Version sans slot dynamique -->
<script lang="ts">
  import { createEventDispatcher } from "svelte";

  export let columns: Array<{
    key: string;
    label: string;
    sortable?: boolean;
    width?: string;
  }> = [];
  export let data: Array<Record<string, any>> = [];
  export let loading: boolean = false;
  export let selectable: boolean = false;
  export let selectedIds: Set<string | number> = new Set();
  export let sortKey: string | null = null;
  export let sortDirection: "asc" | "desc" = "asc";
  export let customRenderers: Record<string, (value: any, row: any) => string> =
    {};

  const dispatch = createEventDispatcher();

  let currentPage = 1;
  let itemsPerPage = 10;
  let searchQuery = "";

  $: filteredData = data.filter((item) => {
    if (!searchQuery) return true;
    return Object.values(item).some((value) =>
      String(value).toLowerCase().includes(searchQuery.toLowerCase()),
    );
  });

  $: sortedData = [...filteredData].sort((a, b) => {
    if (!sortKey) return 0;
    const aVal = a[sortKey];
    const bVal = b[sortKey];
    if (aVal < bVal) return sortDirection === "asc" ? -1 : 1;
    if (aVal > bVal) return sortDirection === "asc" ? 1 : -1;
    return 0;
  });

  $: paginatedData = sortedData.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage,
  );
  $: totalPages = Math.ceil(sortedData.length / itemsPerPage);

  function handleSort(key: string) {
    if (sortKey === key) {
      sortDirection = sortDirection === "asc" ? "desc" : "asc";
    } else {
      sortKey = key;
      sortDirection = "asc";
    }
    dispatch("sort", { key, direction: sortDirection });
  }

  function handleSelectAll(e: Event) {
    const checked = (e.target as HTMLInputElement).checked;
    if (checked) {
      paginatedData.forEach((item) => selectedIds.add(item.id));
    } else {
      paginatedData.forEach((item) => selectedIds.delete(item.id));
    }
    selectedIds = selectedIds;
    dispatch("select", { selectedIds: Array.from(selectedIds) });
  }

  function handleSelectRow(id: string | number, e: Event) {
    const checked = (e.target as HTMLInputElement).checked;
    if (checked) {
      selectedIds.add(id);
    } else {
      selectedIds.delete(id);
    }
    selectedIds = selectedIds;
    dispatch("select", { selectedIds: Array.from(selectedIds) });
  }

  function goToPage(page: number) {
    if (page >= 1 && page <= totalPages) {
      currentPage = page;
      dispatch("pageChange", { page });
    }
  }

  function getCellValue(row: Record<string, any>, key: string): any {
    return row[key] ?? "-";
  }

  function renderCell(
    row: Record<string, any>,
    col: { key: string; label: string; sortable?: boolean; width?: string },
  ): string {
    const value = getCellValue(row, col.key);

    // Vérifier s'il y a un renderer personnalisé
    if (customRenderers[col.key]) {
      return customRenderers[col.key](value, row);
    }

    // Rendu par défaut
    if (col.key === "statut") {
      if (value === "vivant") {
        return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">Vivant</span>';
      } else if (value === "vendu") {
        return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">Vendu</span>';
      } else if (value === "decede") {
        return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">Décédé</span>';
      } else {
        return `<span class="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800">${value}</span>`;
      }
    }

    if (col.key === "sexe") {
      if (value === "male") {
        return '<span class="inline-flex items-center gap-1"><span class="w-2 h-2 bg-blue-500 rounded-full"></span> Mâle</span>';
      } else if (value === "femelle") {
        return '<span class="inline-flex items-center gap-1"><span class="w-2 h-2 bg-pink-500 rounded-full"></span> Femelle</span>';
      } else {
        return `<span class="inline-flex items-center gap-1"><span class="w-2 h-2 bg-purple-500 rounded-full"></span> ${value}</span>`;
      }
    }

    if (col.key === "taux_occupation") {
      const rate = value;
      const colorClass =
        rate >= 90
          ? "text-red-600"
          : rate >= 75
            ? "text-yellow-600"
            : "text-green-600";
      return `<div class="flex items-center gap-2">
                <div class="w-20 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                    <div class="h-full rounded-full transition-all ${rate >= 90 ? "bg-red-500" : rate >= 75 ? "bg-yellow-500" : "bg-green-500"}" style="width: ${rate}%"></div>
                </div>
                <span class="${colorClass}">${rate}%</span>
            </div>`;
    }

    if (col.key === "montant" && row.type) {
      const sign = row.type === "depense" ? "-" : "+";
      const colorClass =
        row.type === "depense" ? "text-red-600" : "text-green-600 font-medium";
      return `<span class="${colorClass}">${sign}${value.toLocaleString("fr-FR")} €</span>`;
    }

    if (col.key === "date" && value && value.includes("-")) {
      return new Date(value).toLocaleDateString("fr-FR");
    }

    if (col.key === "surface") {
      return `${value} m²`;
    }

    if (col.key === "age_jours") {
      if (!value) return "-";
      const years = Math.floor(value / 365);
      const months = Math.floor((value % 365) / 30);
      if (years > 0) return `${years} an${years > 1 ? "s" : ""}`;
      if (months > 0) return `${months} mois`;
      return `${value} jour${value > 1 ? "s" : ""}`;
    }

    if (col.key === "dernier_poids") {
      return value ? `${value} kg` : "-";
    }

    if (col.key === "occupation") {
      const rate = row.taux_occupation;
      const colorClass =
        rate >= 90
          ? "text-red-600"
          : rate >= 75
            ? "text-yellow-600"
            : "text-green-600";
      return `<span class="${colorClass}">${row.occupation_actuelle} / ${row.capacite_maximale}</span>`;
    }

    return value;
  }
</script>

<div
  class="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden"
>
  <!-- Barre de recherche -->
  <div class="p-4 border-b border-gray-200 bg-gray-50">
    <div class="flex flex-col sm:flex-row gap-3 justify-between">
      <div class="relative max-w-sm">
        <input
          type="text"
          placeholder="Rechercher..."
          bind:value={searchQuery}
          class="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-2 focus:ring-primary-100 transition-all"
        />
        <svg
          class="absolute left-3 top-2.5 w-4 h-4 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
          />
        </svg>
      </div>
      <div class="flex items-center gap-2">
        <slot name="actions" />
      </div>
    </div>
  </div>

  <!-- Tableau -->
  <div class="overflow-x-auto">
    <table class="w-full">
      <thead class="bg-gray-50 border-b border-gray-200">
        <tr>
          {#if selectable}
            <th class="w-10 px-4 py-3">
              <input
                type="checkbox"
                checked={paginatedData.length > 0 &&
                  paginatedData.every((item) => selectedIds.has(item.id))}
                on:change={handleSelectAll}
                class="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
            </th>
          {/if}
          {#each columns as col}
            <th
              class="px-4 py-3 text-left text-sm font-semibold text-gray-700"
              style="width: {col.width || 'auto'}"
            >
              {#if col.sortable}
                <button
                  on:click={() => handleSort(col.key)}
                  class="flex items-center gap-1 hover:text-primary-600 transition-colors"
                >
                  {col.label}
                  {#if sortKey === col.key}
                    <svg
                      class="w-3 h-3"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      {#if sortDirection === "asc"}
                        <path
                          stroke-linecap="round"
                          stroke-linejoin="round"
                          stroke-width="2"
                          d="M5 15l7-7 7 7"
                        />
                      {:else}
                        <path
                          stroke-linecap="round"
                          stroke-linejoin="round"
                          stroke-width="2"
                          d="M19 9l-7 7-7-7"
                        />
                      {/if}
                    </svg>
                  {/if}
                </button>
              {:else}
                {col.label}
              {/if}
            </th>
          {/each}
          {#if $$slots["actions-row"]}
            <th class="w-20 px-4 py-3"></th>
          {/if}
        </tr>
      </thead>
      <tbody class="divide-y divide-gray-100">
        {#if loading}
          <tr>
            <td
              colspan={columns.length +
                (selectable ? 1 : 0) +
                ($$slots["actions-row"] ? 1 : 0)}
              class="px-4 py-12 text-center"
            >
              <div class="flex justify-center">
                <div
                  class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"
                ></div>
              </div>
            </td>
          </tr>
        {:else if paginatedData.length === 0}
          <tr>
            <td
              colspan={columns.length +
                (selectable ? 1 : 0) +
                ($$slots["actions-row"] ? 1 : 0)}
              class="px-4 py-12 text-center text-gray-500"
            >
              Aucune donnée disponible
            </td>
          </tr>
        {:else}
          {#each paginatedData as row, index}
            <tr class="hover:bg-gray-50 transition-colors">
              {#if selectable}
                <td class="px-4 py-3">
                  <input
                    type="checkbox"
                    checked={selectedIds.has(row.id)}
                    on:change={(e) => handleSelectRow(row.id, e)}
                    class="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                </td>
              {/if}
              {#each columns as col}
                <td class="px-4 py-3 text-sm text-gray-600">
                  {@html renderCell(row, col)}
                </td>
              {/each}
              {#if $$slots["actions-row"]}
                <td class="px-4 py-3 text-right">
                  <slot name="actions-row" {row} {index} />
                </td>
              {/if}
            </tr>
          {/each}
        {/if}
      </tbody>
    </table>
  </div>

  <!-- Pagination -->
  {#if totalPages > 1}
    <div
      class="px-4 py-3 border-t border-gray-200 bg-gray-50 flex items-center justify-between"
    >
      <div class="text-sm text-gray-500">
        {(currentPage - 1) * itemsPerPage + 1} - {Math.min(
          currentPage * itemsPerPage,
          filteredData.length,
        )} sur {filteredData.length}
      </div>
      <div class="flex items-center gap-1">
        <button
          on:click={() => goToPage(currentPage - 1)}
          disabled={currentPage === 1}
          class="p-2 rounded-lg hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
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
              d="M15 19l-7-7 7-7"
            />
          </svg>
        </button>
        <span class="px-3 py-1 text-sm">{currentPage} / {totalPages}</span>
        <button
          on:click={() => goToPage(currentPage + 1)}
          disabled={currentPage === totalPages}
          class="p-2 rounded-lg hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
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
              d="M9 5l7 7-7 7"
            />
          </svg>
        </button>
      </div>
      <select
        bind:value={itemsPerPage}
        class="text-sm border border-gray-300 rounded-lg px-2 py-1"
      >
        <option value={10}>10 par page</option>
        <option value={25}>25 par page</option>
        <option value={50}>50 par page</option>
        <option value={100}>100 par page</option>
      </select>
    </div>
  {/if}
</div>
