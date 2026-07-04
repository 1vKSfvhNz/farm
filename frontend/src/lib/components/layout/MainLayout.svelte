<!-- lib/components/layout/MainLayout.svelte -->
<script lang="ts">
    import { uiStore } from "../../stores/ui";
    import { onMount } from "svelte";
    import Navbar from "./Navbar.svelte";
    import Sidebar from "./Sidebar.svelte";
    import Footer from "./Footer.svelte";
    import Toast from "../../components/ui/Toast.svelte";
    import Spinner from "../../components/ui/Spinner.svelte";

    let toastMessage = "";
    let toastType: "success" | "error" | "warning" | "info" = "info";
    let showToast = false;
    let isLoading = false;

    onMount(() => {
        // Écouter les événements de toast
        window.addEventListener("show-toast", ((e: CustomEvent) => {
            toastMessage = e.detail.message;
            toastType = e.detail.type || "info";
            showToast = true;

            setTimeout(() => {
                showToast = false;
            }, 5000);
        }) as EventListener);

        // Écouter les événements de chargement
        window.addEventListener("loading-start", (() => {
            isLoading = true;
        }) as EventListener);

        window.addEventListener("loading-end", (() => {
            isLoading = false;
        }) as EventListener);
    });
</script>

<div class="min-h-screen bg-gray-50">
    <Navbar />
    <Sidebar />

    <!-- Main content -->
    <main
        class="pt-16 transition-all duration-300"
        class:ml-0={$uiStore.sidebarOpen === false}
        style="margin-left: {typeof window !== 'undefined' &&
        window.innerWidth >= 768 &&
        $uiStore.sidebarOpen
            ? '280px'
            : '0'}"
    >
        <div class="p-4 md:p-6">
            <slot />
        </div>
    </main>

    <Footer />

    <!-- Global loading spinner -->
    {#if isLoading}
        <div
            class="fixed inset-0 bg-black/50 z-50 flex items-center justify-center"
        >
            <div class="bg-white rounded-xl p-6 shadow-xl">
                <Spinner size="lg" color="primary" />
                <p class="mt-3 text-gray-600">Chargement...</p>
            </div>
        </div>
    {/if}

    <!-- Toast notifications -->
    {#if showToast}
        <Toast message={toastMessage} type={toastType} />
    {/if}
</div>
