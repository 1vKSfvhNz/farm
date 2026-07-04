<!-- src/routes/videos/+page.svelte -->
<script lang="ts">
  import { onMount } from "svelte";
  import { videoApi } from "$lib/api/video";
  import { permissionsStore } from "$lib/stores/permissions";
  import LiveView from "$lib/components/video/LiveView.svelte";
  import ReplaySearch from "$lib/components/video/ReplaySearch.svelte";
  import Button from "$lib/components/ui/Button.svelte";
  import Card from "$lib/components/ui/Card.svelte";

  let cameras: any[] = [];
  let loading = true;
  let activeTab: "live" | "replay" = "live";
  let selectedCamera: any = null;

  const canViewVideo = permissionsStore.hasPermission("can_view_video");

  if (!canViewVideo) {
    window.location.href = "/";
  }

  onMount(async () => {
    await loadCameras();
  });

  async function loadCameras() {
    loading = true;
    try {
      const response = await videoApi.getCameras({ limit: 100 });
      cameras = response.items;
      if (cameras.length > 0) {
        selectedCamera = cameras[0];
      }
    } catch (error) {
      console.error("Failed to load cameras:", error);
    } finally {
      loading = false;
    }
  }
</script>

<div class="space-y-6">
  <div class="flex justify-between items-center">
    <div>
      <h1 class="text-2xl font-bold text-gray-900">Vidéosurveillance</h1>
      <p class="text-sm text-gray-500 mt-1">
        Surveillance en temps réel des enclos
      </p>
    </div>
    <div class="flex gap-2">
      <Button
        on:click={() => (activeTab = "live")}
        variant={activeTab === "live" ? "primary" : "outline"}
      >
        <svg
          class="w-4 h-4 mr-1"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
          />
        </svg>
        Live
      </Button>
      <Button
        on:click={() => (activeTab = "replay")}
        variant={activeTab === "replay" ? "primary" : "outline"}
      >
        <svg
          class="w-4 h-4 mr-1"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          />
        </svg>
        Replay
      </Button>
    </div>
  </div>

  {#if loading}
    <div class="flex justify-center items-center h-64">
      <div
        class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"
      ></div>
    </div>
  {:else if cameras.length === 0}
    <Card>
      <div class="text-center py-12">
        <svg
          class="w-16 h-16 text-gray-300 mx-auto mb-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
          />
        </svg>
        <p class="text-gray-500">Aucune caméra configurée</p>
        {#if permissionsStore.isAdmin()}
          <p class="text-sm text-gray-400 mt-2">
            Contactez votre administrateur pour ajouter des caméras
          </p>
        {/if}
      </div>
    </Card>
  {:else if activeTab === "live"}
    <!-- Vue Live -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div class="lg:col-span-2">
        {#if selectedCamera}
          <LiveView
            streamUrl={selectedCamera.url}
            cameraName={selectedCamera.name}
            autoReconnect={true}
          />
        {/if}
      </div>
      <div>
        <Card title="Caméras disponibles">
          <div class="space-y-2">
            {#each cameras as camera}
              <button
                on:click={() => (selectedCamera = camera)}
                class="w-full text-left p-3 rounded-lg transition-colors"
                class:bg-primary-50={selectedCamera?.id === camera.id}
                class:bg-gray-50={selectedCamera?.id !== camera.id}
                class:border-primary-200={selectedCamera?.id === camera.id}
                class:border-gray-200={selectedCamera?.id !== camera.id}
                class:border={true}
              >
                <div class="flex items-center gap-3">
                  <div
                    class="w-2 h-2 rounded-full"
                    class:bg-green-500={camera.is_active}
                    class:bg-red-500={!camera.is_active}
                  ></div>
                  <div>
                    <p class="font-medium text-gray-800">{camera.name}</p>
                    <p class="text-xs text-gray-500">
                      {camera.is_active ? "En ligne" : "Hors ligne"}
                    </p>
                  </div>
                </div>
              </button>
            {/each}
          </div>
        </Card>
      </div>
    </div>
  {:else}
    <!-- Vue Replay -->
    <ReplaySearch
      cameraOptions={cameras
        .filter((c) => c.is_active)
        .map((c) => ({ value: c.id, label: c.name }))}
      actionTypes={[
        { value: "animal_detected", label: "Animal détecté" },
        { value: "movement", label: "Mouvement" },
        { value: "alarm", label: "Alarme" },
      ]}
      on:play={(e) => {
        console.log("Play video:", e.detail);
      }}
    />
  {/if}
</div>
