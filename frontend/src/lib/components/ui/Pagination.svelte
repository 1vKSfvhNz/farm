<!-- lib/components/ui/Pagination.svelte -->
<script lang="ts">
    export let currentPage: number = 1;
    export let totalPages: number = 1;
    export let maxVisible: number = 5;

    let visiblePages: number[] = [];

    $: {
        const half = Math.floor(maxVisible / 2);
        let start = Math.max(1, currentPage - half);
        let end = Math.min(totalPages, start + maxVisible - 1);

        if (end - start + 1 < maxVisible) {
            start = Math.max(1, end - maxVisible + 1);
        }

        visiblePages = [];
        for (let i = start; i <= end; i++) {
            visiblePages.push(i);
        }
    }

    function goToPage(page: number) {
        if (page >= 1 && page <= totalPages && page !== currentPage) {
            currentPage = page;
        }
    }
</script>

{#if totalPages > 1}
    <nav class="flex items-center justify-center gap-1" aria-label="Pagination">
        <button
            on:click={() => goToPage(currentPage - 1)}
            disabled={currentPage === 1}
            class="p-2 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
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
                    d="M15 19l-7-7 7-7"
                />
            </svg>
        </button>

        {#if visiblePages[0] > 1}
            <button
                on:click={() => goToPage(1)}
                class="px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors"
                >1</button
            >
            {#if visiblePages[0] > 2}
                <span class="px-3 py-2">...</span>
            {/if}
        {/if}

        {#each visiblePages as page}
            <button
                on:click={() => goToPage(page)}
                class="px-3 py-2 rounded-lg transition-colors {page ===
                currentPage
                    ? 'bg-primary-600 text-white'
                    : 'hover:bg-gray-100'}"
            >
                {page}
            </button>
        {/each}

        {#if visiblePages[visiblePages.length - 1] < totalPages}
            {#if visiblePages[visiblePages.length - 1] < totalPages - 1}
                <span class="px-3 py-2">...</span>
            {/if}
            <button
                on:click={() => goToPage(totalPages)}
                class="px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors"
                >{totalPages}</button
            >
        {/if}

        <button
            on:click={() => goToPage(currentPage + 1)}
            disabled={currentPage === totalPages}
            class="p-2 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
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
                    d="M9 5l7 7-7 7"
                />
            </svg>
        </button>
    </nav>
{/if}
