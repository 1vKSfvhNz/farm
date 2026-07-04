<script lang="ts">
    import { onMount } from "svelte";
    import { experimentalStore } from "../../stores/experimental";

    type ModeType = "complet" | "hybride" | "experimental";

    interface ModeStatus {
        mode: ModeType;
        confiance_moyenne: number;
        jours_collecte: number;
        nombre_donnees_par_espece?: Record<string, number>;
        recommandations?: string[];
    }

    let modeStatus: ModeStatus | null = null;

    const modeColors: Record<ModeType, string> = {
        complet: "bg-green-100 border-green-200 text-green-800",
        hybride: "bg-blue-100 border-blue-200 text-blue-800",
        experimental: "bg-purple-100 border-purple-200 text-purple-800",
    };

    const modeIcons: Record<ModeType, string> = {
        complet: "M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z",

        hybride:
            "M2.25 12l8.954-8.955c.44-.439 1.152-.439 1.591 0L21.75 12M4.5 9.75v10.125c0 .621.504 1.125 1.125 1.125H9.75v-4.875c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21h4.125c.621 0 1.125-.504 1.125-1.125V9.75M8.25 21h8.25",

        experimental:
            "M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.75 18.75L21 21",
    };

    function getModeColor(mode: ModeType): string {
        return modeColors[mode];
    }

    function getModeIcon(mode: ModeType): string {
        return modeIcons[mode];
    }

    onMount(() => {
        experimentalStore.loadModeStatus();

        const unsubscribe = experimentalStore.subscribe((state: any) => {
            modeStatus = state.modeStatus ?? null;
        });

        return unsubscribe;
    });
</script>

{#if modeStatus && modeStatus.mode !== "complet"}
    <div
        class={`rounded-xl border-2 ${getModeColor(modeStatus.mode)} bg-white/50 backdrop-blur-sm p-4`}
    >
        <div class="flex items-start gap-3">
            <div
                class="w-10 h-10 rounded-lg bg-white flex items-center justify-center shadow-sm"
            >
                <svg
                    class="w-5 h-5 text-purple-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="2"
                        d={getModeIcon(modeStatus.mode)}
                    />
                </svg>
            </div>

            <div class="flex-1">
                <div class="flex items-center gap-2">
                    <h4 class="font-semibold text-gray-900">
                        Mode {modeStatus.mode === "hybride"
                            ? "Hybride"
                            : "Expérimental"}
                    </h4>

                    <span
                        class="text-xs px-2 py-0.5 rounded-full bg-gray-200 text-gray-700"
                    >
                        Confiance: {modeStatus.confiance_moyenne}%
                    </span>
                </div>

                <p class="text-sm text-gray-600 mt-1">
                    L'application utilise des prédictions basées sur
                    {modeStatus.nombre_donnees_par_espece
                        ? Object.keys(modeStatus.nombre_donnees_par_espece)
                              .length
                        : 0}
                    espèces.
                    {modeStatus.jours_collecte} jours de données collectées.
                </p>

                {#if modeStatus.recommandations?.length}
                    <div class="mt-2">
                        <p class="text-xs text-gray-500">Recommandations :</p>

                        <ul
                            class="list-disc list-inside text-xs text-gray-600 mt-1"
                        >
                            {#each modeStatus.recommandations as rec}
                                <li>{rec}</li>
                            {/each}
                        </ul>
                    </div>
                {/if}
            </div>

            <a
                href="/experimental"
                class="text-sm text-primary-600 hover:text-primary-700 font-medium whitespace-nowrap"
            >
                Voir détails →
            </a>
        </div>
    </div>
{/if}
