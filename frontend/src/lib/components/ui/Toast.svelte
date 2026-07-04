<!-- lib/components/ui/Toast.svelte -->
<script lang="ts">
  import { onDestroy } from "svelte";

  export let message: string = "";
  export let type: "success" | "error" | "warning" | "info" = "info";
  export let duration: number = 5000;
  export let closable: boolean = true;

  let visible: boolean = true;
  let timeout: ReturnType<typeof setTimeout>;

  const icons = {
    success: "✓",
    error: "✕",
    warning: "⚠",
    info: "ℹ",
  };

  const colors = {
    success: "bg-green-50 border-green-200 text-green-800",
    error: "bg-red-50 border-red-200 text-red-800",
    warning: "bg-yellow-50 border-yellow-200 text-yellow-800",
    info: "bg-blue-50 border-blue-200 text-blue-800",
  };

  function close() {
    visible = false;
  }

  function startTimer() {
    if (duration > 0) {
      timeout = setTimeout(() => {
        close();
      }, duration);
    }
  }

  function resetTimer() {
    if (timeout) clearTimeout(timeout);
    startTimer();
  }

  onDestroy(() => {
    if (timeout) clearTimeout(timeout);
  });

  startTimer();
</script>

{#if visible}
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div
    class="fixed bottom-4 right-4 z-50 flex items-center gap-3 px-4 py-3 rounded-lg border shadow-lg animate-in slide-in-from-right-5 fade-in duration-300 {colors[
      type
    ]}"
    on:mouseenter={resetTimer}
    on:mouseleave={resetTimer}
  >
    <span class="font-bold text-lg">{icons[type]}</span>
    <span class="flex-1 text-sm">{message}</span>
    {#if closable}
      <button
        on:click={close}
        class="ml-4 text-gray-400 hover:text-gray-600 transition-colors"
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
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      </button>
    {/if}
  </div>
{/if}
