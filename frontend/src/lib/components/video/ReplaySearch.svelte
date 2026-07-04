<!-- lib/components/video/ReplaySearch.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import DatePicker from "../ui/DatePicker.svelte";
    import Button from "../ui/Button.svelte";
    import Select from "../ui/Select.svelte";
    import Modal from "../ui/Modal.svelte";
    import VideoPlayer from "./VideoPlayer.svelte";

    export let cameraOptions: Array<{ value: number; label: string }> = [];
    export let actionTypes: Array<{ value: string; label: string }> = [];

    const dispatch = createEventDispatcher();

    let filters = {
        camera_id: 0,
        start_date: "",
        end_date: "",
        action_type: "",
    };

    let events: Array<{
        id: number;
        timestamp: string;
        camera_name: string;
        action_type: string;
        thumbnail_url?: string;
    }> = [];

    let loading = false;
    let selectedEvent: any = null;

    async function search() {
        loading = true;
        // Simulation de recherche - à remplacer par appel API réel
        setTimeout(() => {
            events = [
                {
                    id: 1,
                    timestamp: "2024-01-15T10:30:00",
                    camera_name: "Caméra 1",
                    action_type: "animal_detected",
                    thumbnail_url: "",
                },
                {
                    id: 2,
                    timestamp: "2024-01-15T11:45:00",
                    camera_name: "Caméra 2",
                    action_type: "movement",
                    thumbnail_url: "",
                },
                {
                    id: 3,
                    timestamp: "2024-01-15T14:20:00",
                    camera_name: "Caméra 1",
                    action_type: "animal_detected",
                    thumbnail_url: "",
                },
            ];
            loading = false;
        }, 1000);
    }

    function playEvent(event: any) {
        selectedEvent = event;
        dispatch("play", event);
    }

    function formatDate(dateStr: string): string {
        return new Date(dateStr).toLocaleString("fr-FR");
    }
</script>

<div class="bg-white rounded-xl border border-gray-200 shadow-sm">
    <div class="p-5 border-b border-gray-200">
        <h3 class="text-lg font-semibold text-gray-900">
            Recherche d'enregistrements
        </h3>
    </div>

    <div class="p-5 space-y-4">
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Select
                label="Caméra"
                bind:value={filters.camera_id}
                options={[
                    { value: 0, label: "Toutes les caméras" },
                    ...cameraOptions,
                ]}
            />
            <DatePicker label="Date de début" bind:value={filters.start_date} />
            <DatePicker label="Date de fin" bind:value={filters.end_date} />
            <Select
                label="Type d'action"
                bind:value={filters.action_type}
                options={[
                    { value: "", label: "Tous les types" },
                    ...actionTypes,
                ]}
            />
        </div>

        <div class="flex justify-end">
            <Button on:click={search} {loading} variant="primary">
                Rechercher
            </Button>
        </div>
    </div>

    {#if events.length > 0}
        <div class="border-t border-gray-200">
            <div
                class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-5"
            >
                {#each events as event}
                    <!-- svelte-ignore a11y-no-static-element-interactions -->
                    <!-- svelte-ignore a11y-click-events-have-key-events -->
                    <div
                        class="bg-gray-50 rounded-lg border border-gray-200 overflow-hidden cursor-pointer hover:shadow-md transition-all"
                        on:click={() => playEvent(event)}
                    >
                        <div
                            class="aspect-video bg-gray-800 flex items-center justify-center"
                        >
                            {#if event.thumbnail_url}
                                <img
                                    src={event.thumbnail_url}
                                    alt="Thumbnail"
                                    class="w-full h-full object-cover"
                                />
                            {:else}
                                <svg
                                    class="w-12 h-12 text-gray-600"
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
                            {/if}
                        </div>
                        <div class="p-3">
                            <div class="flex items-center justify-between">
                                <span class="text-sm font-medium text-gray-800"
                                    >{event.camera_name}</span
                                >
                                <span class="text-xs text-gray-500"
                                    >{formatDate(event.timestamp)}</span
                                >
                            </div>
                            <p class="text-xs text-gray-500 mt-1">
                                {event.action_type}
                            </p>
                        </div>
                    </div>
                {/each}
            </div>
        </div>
    {:else if !loading}
        <div class="p-8 text-center border-t border-gray-200">
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
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
            </svg>
            <p class="text-gray-500">Aucun enregistrement trouvé</p>
            <p class="text-sm text-gray-400 mt-1">
                Modifiez vos critères de recherche
            </p>
        </div>
    {/if}
</div>

{#if selectedEvent}
    <Modal
        open={true}
        title="Lecture vidéo"
        size="xl"
        on:close={() => (selectedEvent = null)}
    >
        <VideoPlayer
            src={`/api/video/records/${selectedEvent.id}/stream`}
            controls={true}
        />
    </Modal>
{/if}
