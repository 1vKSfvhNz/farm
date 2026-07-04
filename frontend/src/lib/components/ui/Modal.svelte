<!-- lib/components/ui/Modal.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";

    export let open: boolean = false;
    export let title: string = "";
    export let size: "sm" | "md" | "lg" | "xl" | "full" = "md";
    export let closeOnOutsideClick: boolean = true;
    export let closeOnEscape: boolean = true;

    const dispatch = createEventDispatcher();

    let modalRef: HTMLDivElement;

    const sizes = {
        sm: "max-w-md",
        md: "max-w-lg",
        lg: "max-w-2xl",
        xl: "max-w-4xl",
        full: "max-w-[90vw]",
    };

    function handleClose() {
        open = false;
        dispatch("close");
    }

    function handleBackdropClick(e: MouseEvent) {
        if (closeOnOutsideClick && e.target === modalRef) {
            handleClose();
        }
    }

    function handleKeydown(e: KeyboardEvent) {
        if (closeOnEscape && e.key === "Escape" && open) {
            handleClose();
        }
    }

    $: {
        if (typeof window !== "undefined") {
            if (open) {
                document.body.style.overflow = "hidden";
                window.addEventListener("keydown", handleKeydown);
            } else {
                document.body.style.overflow = "";
                window.removeEventListener("keydown", handleKeydown);
            }
        }
    }
</script>

{#if open}
    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div
        bind:this={modalRef}
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm transition-all duration-200"
        on:click={handleBackdropClick}
    >
        <div
            class="bg-white rounded-xl shadow-xl {sizes[
                size
            ]} w-full mx-4 max-h-[90vh] flex flex-col animate-in fade-in zoom-in-95 duration-200"
        >
            {#if title}
                <div
                    class="flex justify-between items-center p-5 border-b border-gray-200"
                >
                    <h2 class="text-xl font-semibold text-gray-900">{title}</h2>
                    <button
                        on:click={handleClose}
                        class="text-gray-400 hover:text-gray-600 transition-colors"
                    >
                        <svg
                            class="w-6 h-6"
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
                </div>
            {/if}

            <div class="flex-1 overflow-y-auto p-5">
                <slot />
            </div>

            {#if $$slots.footer}
                <div
                    class="p-5 border-t border-gray-200 bg-gray-50 rounded-b-xl"
                >
                    <slot name="footer" />
                </div>
            {/if}
        </div>
    </div>
{/if}
