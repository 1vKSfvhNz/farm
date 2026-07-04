<!-- src/routes/enclos/[id]/+page.svelte -->
<script lang="ts">
  import { onMount } from "svelte";
  import { enclosApi } from "$lib/api/enclos";
  import { bovinsApi } from "$lib/api/bovins";
  import { ovinsApi } from "$lib/api/ovins";
  import { caprinsApi } from "$lib/api/caprins";
  import { avicolesApi } from "$lib/api/avicoles";
  import { piscicolesApi } from "$lib/api/piscicoles";
  import { apiaryApi } from "$lib/api/apiary";
  import { entomocultureApi } from "$lib/api/entomoculture";
  import { permissionsStore } from "$lib/stores/permissions";
  import Card from "$lib/components/ui/Card.svelte";
  import Button from "$lib/components/ui/Button.svelte";
  import Spinner from "$lib/components/ui/Spinner.svelte";
  import AlertBadge from "$lib/components/ui/AlertBadge.svelte";
  import type {
    EnclosResponse,
    AvicoleResponse,
    BovinResponse,
    CaprinResponse,
    EntomocultureLotResponse,
    OvinResponse,
    PiscicoleResponse,
    RucheResponse
  } from "$lib";
  import { getFullPhotoUrl } from "$lib/utils/media";

  let id: number;
  let enclos: EnclosResponse | null = null;
  let loading = true;

  // Données par espèce
  let bovins: BovinResponse[] = [];
  let ovins: OvinResponse[] = [];
  let caprins: CaprinResponse[] = [];
  let avicoles: AvicoleResponse[] = [];
  let piscicoles: PiscicoleResponse[] = [];
  let ruches: RucheResponse[] = [];
  let lots: EntomocultureLotResponse[] = [];

  const canEdit = permissionsStore.canEditEnclos;

  // Configuration des espèces avec leurs couleurs et icônes
  const especesConfig = {
    bovins: {
      label: "Bovins",
      icon: "🐄",
      color: "blue",
      bgColor: "bg-blue-50",
      borderColor: "border-blue-400",
      textColor: "text-blue-700",
      data: bovins,
      route: "bovins",
      count: () => bovins.length,
      stats: () => {
        const males = bovins.filter(a => a.sexe === "male").length;
        const femelles = bovins.filter(a => a.sexe === "femelle").length;
        const lait = bovins.filter(a => a.production_laitiere).length;
        const viande = bovins.filter(a => a.production_viande).length;
        const reproduction = bovins.filter(a => a.production_reproduction).length;
        return { males, femelles, lait, viande, reproduction };
      },
      renderStats: (stats: any) => `
        <div class="flex items-center gap-1"><span class="text-blue-600">♂</span><span class="text-gray-600">${stats.males} mâles</span></div>
        <div class="flex items-center gap-1"><span class="text-pink-600">♀</span><span class="text-gray-600">${stats.femelles} femelles</span></div>
        <div class="flex items-center gap-1"><span class="text-blue-400">🥛</span><span class="text-gray-600">${stats.lait} en lactation</span></div>
        <div class="flex items-center gap-1"><span class="text-red-400">🥩</span><span class="text-gray-600">${stats.viande} viande</span></div>
        <div class="flex items-center gap-1"><span class="text-green-400">🐣</span><span class="text-gray-600">${stats.reproduction} reproduction</span></div>
      `
    },
    ovins: {
      label: "Ovins",
      icon: "🐑",
      color: "green",
      bgColor: "bg-green-50",
      borderColor: "border-green-400",
      textColor: "text-green-700",
      data: ovins,
      route: "ovins",
      count: () => ovins.length,
      stats: () => {
        const males = ovins.filter(a => a.sexe === "male").length;
        const femelles = ovins.filter(a => a.sexe === "femelle").length;
        const viande = ovins.filter(a => a.production_viande).length;
        const reproduction = ovins.filter(a => a.production_reproduction).length;
        const laine = ovins.filter(a => a.production_laine).length;
        return { males, femelles, viande, reproduction, laine };
      },
      renderStats: (stats: any) => `
        <div class="flex items-center gap-1"><span class="text-green-600">♂</span><span class="text-gray-600">${stats.males} mâles</span></div>
        <div class="flex items-center gap-1"><span class="text-pink-600">♀</span><span class="text-gray-600">${stats.femelles} femelles</span></div>
        <div class="flex items-center gap-1"><span class="text-red-400">🥩</span><span class="text-gray-600">${stats.viande} viande</span></div>
        <div class="flex items-center gap-1"><span class="text-green-400">🐣</span><span class="text-gray-600">${stats.reproduction} reproduction</span></div>
        <div class="flex items-center gap-1"><span class="text-yellow-400">🧶</span><span class="text-gray-600">${stats.laine} laine</span></div>
      `
    },
    caprins: {
      label: "Caprins",
      icon: "🐐",
      color: "amber",
      bgColor: "bg-amber-50",
      borderColor: "border-amber-400",
      textColor: "text-amber-700",
      data: caprins,
      route: "caprins",
      count: () => caprins.length,
      stats: () => {
        const males = caprins.filter(a => a.sexe === "male").length;
        const femelles = caprins.filter(a => a.sexe === "femelle").length;
        const viande = caprins.filter(a => a.production_viande).length;
        const reproduction = caprins.filter(a => a.production_reproduction).length;
        return { males, femelles, viande, reproduction };
      },
      renderStats: (stats: any) => `
        <div class="flex items-center gap-1"><span class="text-amber-600">♂</span><span class="text-gray-600">${stats.males} mâles</span></div>
        <div class="flex items-center gap-1"><span class="text-pink-600">♀</span><span class="text-gray-600">${stats.femelles} femelles</span></div>
        <div class="flex items-center gap-1"><span class="text-red-400">🥩</span><span class="text-gray-600">${stats.viande} viande</span></div>
        <div class="flex items-center gap-1"><span class="text-green-400">🐣</span><span class="text-gray-600">${stats.reproduction} reproduction</span></div>
      `
    },
    avicoles: {
      label: "Avicoles",
      icon: "🐔",
      color: "yellow",
      bgColor: "bg-yellow-50",
      borderColor: "border-yellow-400",
      textColor: "text-yellow-700",
      data: avicoles,
      route: "avicoles",
      count: () => avicoles.length,
      stats: () => {
        const males = avicoles.filter(a => a.sexe === "male").length;
        const femelles = avicoles.filter(a => a.sexe === "femelle").length;
        const ponte = avicoles.filter(a => a.production_ponte).length;
        const viande = avicoles.filter(a => a.production_viande).length;
        const reproduction = avicoles.filter(a => a.production_reproduction).length;
        const totalOeufs = avicoles.reduce((sum, a) => sum + (a.oeufs_pondus_total || 0), 0);
        const totalPoidsOeufs = avicoles.reduce((sum, a) => sum + (a.poids_oeufs_total || 0), 0);
        return { males, femelles, ponte, viande, reproduction, totalOeufs, totalPoidsOeufs };
      },
      renderStats: (stats: any) => `
        <div class="flex items-center gap-1"><span class="text-yellow-600">♂</span><span class="text-gray-600">${stats.males} mâles</span></div>
        <div class="flex items-center gap-1"><span class="text-pink-600">♀</span><span class="text-gray-600">${stats.femelles} femelles</span></div>
        <div class="flex items-center gap-1"><span class="text-yellow-400">🥚</span><span class="text-gray-600">${stats.ponte} pondeuses</span></div>
        <div class="flex items-center gap-1"><span class="text-red-400">🥩</span><span class="text-gray-600">${stats.viande} viande</span></div>
        <div class="flex items-center gap-1"><span class="text-green-400">🐣</span><span class="text-gray-600">${stats.reproduction} reproduction</span></div>
        <div class="flex items-center gap-1 col-span-2 text-xs text-gray-500">🥚 ${stats.totalOeufs} œufs pondus (${stats.totalPoidsOeufs.toFixed(1)} kg)</div>
      `
    },
    piscicoles: {
      label: "Piscicoles",
      icon: "🐟",
      color: "cyan",
      bgColor: "bg-cyan-50",
      borderColor: "border-cyan-400",
      textColor: "text-cyan-700",
      data: piscicoles,
      route: "piscicoles",
      count: () => piscicoles.length,
      stats: () => {
        const viande = piscicoles.filter(a => a.production_viande).length;
        const reproduction = piscicoles.filter(a => a.production_reproduction).length;
        const tailles = piscicoles.map(a => a.taille_moyenne).filter((t): t is number => t !== undefined);
        const tailleMoy = tailles.length > 0 ? tailles.reduce((a, b) => a + b, 0) / tailles.length : 0;
        const poids = piscicoles.map(a => a.poids_moyen).filter((p): p is number => p !== undefined);
        const poidsMoy = poids.length > 0 ? poids.reduce((a, b) => a + b, 0) / poids.length : 0;
        return { viande, reproduction, tailleMoyenne: tailleMoy, poidsMoyen: poidsMoy };
      },
      renderStats: (stats: any) => `
        <div class="flex items-center gap-1"><span class="text-red-400">🥩</span><span class="text-gray-600">${stats.viande} viande</span></div>
        <div class="flex items-center gap-1"><span class="text-green-400">🐣</span><span class="text-gray-600">${stats.reproduction} reproduction</span></div>
        <div class="flex items-center gap-1 text-xs text-gray-500">📏 Taille: ${stats.tailleMoyenne > 0 ? stats.tailleMoyenne.toFixed(1) + ' cm' : 'N/A'}</div>
        <div class="flex items-center gap-1 text-xs text-gray-500">⚖️ Poids: ${stats.poidsMoyen > 0 ? stats.poidsMoyen.toFixed(1) + ' kg' : 'N/A'}</div>
      `
    },
    ruches: {
      label: "Ruches",
      icon: "🐝",
      color: "orange",
      bgColor: "bg-orange-50",
      borderColor: "border-orange-400",
      textColor: "text-orange-700",
      data: ruches,
      route: "ruches",
      count: () => ruches.length,
      stats: () => {
        const actives = ruches.filter(r => r.statut === "active").length;
        const orphelines = ruches.filter(r => r.statut === "orpheline").length;
        const essaimage = ruches.filter(r => r.statut === "en_essaimage").length;
        const mortes = ruches.filter(r => r.statut === "morte").length;
        return { actives, orphelines, essaimage, mortes };
      },
      renderStats: (stats: any) => `
        <div class="flex items-center gap-1"><span class="text-green-500">✅</span><span class="text-gray-600">${stats.actives} actives</span></div>
        <div class="flex items-center gap-1"><span class="text-red-500">❌</span><span class="text-gray-600">${stats.mortes} mortes</span></div>
        <div class="flex items-center gap-1"><span class="text-orange-500">👑</span><span class="text-gray-600">${stats.orphelines} orphelines</span></div>
        <div class="flex items-center gap-1"><span class="text-yellow-500">✈️</span><span class="text-gray-600">${stats.essaimage} essaimage</span></div>
      `
    },
    lots: {
      label: "Nids",
      icon: "🪹",
      color: "purple",
      bgColor: "bg-purple-50",
      borderColor: "border-purple-400",
      textColor: "text-purple-700",
      data: lots,
      route: "lots",
      count: () => lots.length,
      stats: () => {
        const larves = lots.filter(l => l.type_production === "larves").length;
        const reproduction = lots.filter(l => l.type_production === "reproduction").length;
        const oeufs = lots.filter(l => l.type_production === "oeufs").length;
        const stades: Record<string, number> = {};
        lots.forEach(l => {
          stades[l.stade_actuel] = (stades[l.stade_actuel] || 0) + 1;
        });
        const totalQuantite = lots.reduce((sum, l) => sum + (l.quantite_estimative || 0), 0);
        return { larves, reproduction, oeufs, stades, totalQuantite };
      },
      renderStats: (stats: any) => `
        <div class="flex items-center gap-1"><span class="text-purple-500">🐛</span><span class="text-gray-600">${stats.larves} larves</span></div>
        <div class="flex items-center gap-1"><span class="text-green-500">🔄</span><span class="text-gray-600">${stats.reproduction} reproduction</span></div>
        <div class="flex items-center gap-1"><span class="text-yellow-500">🥚</span><span class="text-gray-600">${stats.oeufs} œufs</span></div>
        <div class="flex items-center gap-1 col-span-2 text-xs text-gray-500">📊 Stades: ${Object.entries(stats.stades || {}).map(([stade, nb]) => `${stade}: ${nb}`).join(', ')}</div>
        <div class="flex items-center gap-1 col-span-2 text-xs text-gray-500">📊 Quantité estimée: ${stats.totalQuantite} individus</div>
      `
    }
  };

  // Récupérer les espèces présentes avec leurs données
  function getEspecesPresentes() {
    const presentes: Array<{ key: string; config: any; data: any[]; stats: any; count: number }> = [];
    for (const [key, config] of Object.entries(especesConfig)) {
      const count = config.count();
      if (count > 0) {
        presentes.push({
          key,
          config,
          data: config.data,
          stats: config.stats(),
          count
        });
      }
    }
    return presentes;
  }

  onMount(async () => {
    const path = window.location.pathname;
    const match = path.match(/\/enclos\/(\d+)/);
    if (match) {
      id = parseInt(match[1]);
      await loadData();
    }
  });

  async function loadData() {
    loading = true;
    try {
      const [enclosData, bovinsData, ovinsData, caprinsData, avicolesData, piscicolesData, ruchesData, lotsData] =
        await Promise.all([
          enclosApi.getEnclosById(id),
          bovinsApi.getBovins({ enclos_id: id, limit: 100 }),
          ovinsApi.getOvins({ enclos_id: id, limit: 100 }),
          caprinsApi.getCaprins({ enclos_id: id, limit: 100 }),
          avicolesApi.getAvicoles({ enclos_id: id, limit: 100 }),
          piscicolesApi.getPiscicoles({ enclos_id: id, limit: 100 }),
          apiaryApi.getRuches({ emplacement: "", limit: 100 }),
          entomocultureApi.getLots({ enclos_id: id, limit: 100 }),
        ]);

      enclos = enclosData;
      bovins = bovinsData.items || [];
      ovins = ovinsData.items || [];
      caprins = caprinsData.items || [];
      avicoles = avicolesData.items || [];
      piscicoles = piscicolesData.items || [];
      ruches = ruchesData.items || [];
      lots = lotsData.items || [];
    } catch (error) {
      console.error("Failed to load data:", error);
    } finally {
      loading = false;
    }
  }

  function getStatusBadge(statut: string) {
    const statusMap: Record<string, { niveau: "info" | "warning" | "critical"; label: string }> = {
      vivant: { niveau: "info", label: "Vivant" },
      vendu: { niveau: "warning", label: "Vendu" },
      decede: { niveau: "critical", label: "Décédé" },
      transfere: { niveau: "info", label: "Transféré" }
    };
    return statusMap[statut] || { niveau: "info", label: statut };
  }

  function getSexeEmoji(sexe: string): string {
    return sexe === "male" ? "♂" : sexe === "femelle" ? "♀" : "⚥";
  }

  function getDensite(): { value: number; unite: string; niveau: "normal" | "warning" | "critical" } {
    const total = bovins.length + ovins.length + caprins.length + avicoles.length +
                  piscicoles.length + ruches.length + lots.length;

    if (total === 0 || !enclos) {
      return { value: 0, unite: "animaux/m²", niveau: "normal" };
    }

    const surface = enclos.surface;
    if (surface === 0) {
      return { value: 0, unite: "animaux/m²", niveau: "normal" };
    }

    const densite = total / surface;
    let niveau: "normal" | "warning" | "critical" = "normal";

    if (enclos.type === "bassin" || enclos.type === "bac") {
      const volume = enclos.volume || surface * (enclos.hauteur || 1);
      const densiteVol = total / volume;
      if (densiteVol >= 10) niveau = "critical";
      else if (densiteVol >= 5) niveau = "warning";
      return { value: densiteVol, unite: "animaux/m³", niveau };
    }

    if (densite >= 1) niveau = "critical";
    else if (densite >= 0.5) niveau = "warning";
    return { value: densite, unite: "animaux/m²", niveau };
  }

  function getDensiteColor(niveau: "normal" | "warning" | "critical"): string {
    return {
      normal: "text-green-600",
      warning: "text-yellow-600",
      critical: "text-red-600"
    }[niveau];
  }

  // Récupérer les 3 premiers animaux d'une espèce pour l'aperçu
  function getPreviewAnimals(data: any[], limit: number = 3) {
    return data.slice(0, limit);
  }
</script>

<div class="min-h-screen bg-gray-50">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    {#if loading}
      <div class="flex justify-center items-center h-64">
        <Spinner size="lg" />
      </div>
    {:else if !enclos}
      <div class="text-center py-12">
        <div class="text-6xl mb-4">🏡</div>
        <p class="text-gray-500">Enclos non trouvé</p>
        <Button on:click={() => (window.location.href = "/enclos")} variant="primary" className="mt-4">
          Retour à la liste
        </Button>
      </div>
    {:else}
      <!-- En-tête -->
      <div class="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4 mb-8">
        <div>
          <div class="flex items-center gap-3">
            <div class="w-14 h-14 bg-gray-100 rounded-xl flex items-center justify-center text-3xl">
              {enclos.type === "bassin" ? "💧" : enclos.type === "pâturage" ? "🌿" : "🏡"}
            </div>
            <div>
              <h1 class="text-3xl font-bold text-gray-900">{enclos.name}</h1>
              <p class="text-sm text-gray-500 mt-1">Type: {enclos.type}</p>
            </div>
          </div>
        </div>
        <div class="flex gap-2">
          {#if canEdit()}
            <Button on:click={() => (window.location.href = `/enclos/${id}/edit`)} variant="outline">
              <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              Modifier
            </Button>
          {/if}
          <Button on:click={() => (window.location.href = "/enclos")} variant="outline">
            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Retour
          </Button>
        </div>
      </div>

      <!-- Grille d'informations de l'enclos -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <!-- Dimensions -->
        <Card title="📐 Dimensions" padding="md" bordered={true}>
          <div class="space-y-3 text-sm">
            <div class="flex justify-between py-1 border-b border-gray-100">
              <span class="text-gray-500">Longueur:</span>
              <span class="font-medium">{enclos.longueur} m</span>
            </div>
            <div class="flex justify-between py-1 border-b border-gray-100">
              <span class="text-gray-500">Largeur:</span>
              <span class="font-medium">{enclos.largeur} m</span>
            </div>
            <div class="flex justify-between py-1 border-b border-gray-100">
              <span class="text-gray-500">Surface:</span>
              <span class="font-medium text-lg">{enclos.surface} m²</span>
            </div>
            {#if enclos.hauteur}
              <div class="flex justify-between py-1 border-b border-gray-100">
                <span class="text-gray-500">Hauteur:</span>
                <span class="font-medium">{enclos.hauteur} m</span>
              </div>
            {/if}
            {#if enclos.volume}
              <div class="flex justify-between py-1">
                <span class="text-gray-500">Volume:</span>
                <span class="font-medium">{enclos.volume} m³</span>
              </div>
            {/if}
          </div>
        </Card>

        <!-- Localisation -->
        <Card title="📍 Localisation" padding="md" bordered={true}>
          <div class="space-y-3 text-sm">
            {#if enclos.zone}
              <div class="flex justify-between py-1 border-b border-gray-100">
                <span class="text-gray-500">Zone:</span>
                <span class="font-medium">{enclos.zone}</span>
              </div>
            {/if}
            {#if enclos.localisation_gps}
              <div class="flex justify-between py-1">
                <span class="text-gray-500">Coordonnées GPS:</span>
                <span class="font-mono text-xs">{enclos.localisation_gps}</span>
              </div>
            {/if}
            {#if !enclos.zone && !enclos.localisation_gps}
              <p class="text-gray-400 text-center py-4">Aucune information de localisation</p>
            {/if}
          </div>
        </Card>

        <!-- Statistiques de l'enclos -->
        <Card title="📊 Statistiques" padding="md" bordered={true}>
          <div class="space-y-2 text-sm">
            <!-- Total par espèce -->
            {#each getEspecesPresentes() as { key, config, count }}
              <div class="flex justify-between py-1 border-b border-gray-100">
                <span>{config.icon} {config.label}:</span>
                <span class="font-medium">{count}</span>
              </div>
            {/each}
            
            <!-- Total général -->
            <div class="flex justify-between pt-2 border-t-2 border-gray-200">
              <span class="font-semibold">Total:</span>
              <span class="font-bold text-lg">
                {bovins.length + ovins.length + caprins.length + avicoles.length +
                 piscicoles.length + ruches.length + lots.length}
              </span>
            </div>
            
            <!-- Densité -->
             {#if enclos}
              {@const densite = getDensite()}
              <div class="flex justify-between pt-2 border-t border-gray-100">
                <span class="text-gray-500">Densité:</span>
                <span class={`font-medium ${getDensiteColor(densite.niveau)}`}>
                  {densite.value.toFixed(2)} {densite.unite}
                </span>
              </div>
            {/if}
          </div>
        </Card>
      </div>

      <!-- ============================================ -->
      <!-- CADRANS PAR ESPÈCE PRÉSENTE DANS L'ENCLOS    -->
      <!-- ============================================ -->

      {#if getEspecesPresentes().length > 0}
        <h2 class="text-xl font-semibold text-gray-800 mb-4">🐾 Animaux présents</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
          {#each getEspecesPresentes() as { key, config, data, stats, count }}
            <div class={`rounded-xl border-2 ${config.borderColor} ${config.bgColor} p-5 shadow-sm hover:shadow-md transition-shadow`}>
              <!-- En-tête du cadran -->
              <div class="flex items-center justify-between mb-3">
                <div class="flex items-center gap-2">
                  <span class="text-3xl">{config.icon}</span>
                  <div>
                    <h3 class={`font-bold text-lg ${config.textColor}`}>{config.label}</h3>
                    <p class="text-xs text-gray-500">{count} animal{count > 1 ? 's' : ''}</p>
                  </div>
                </div>
                <a
                  href={`/${config.route}?enclos_id=${id}`}
                  class="text-xs text-primary-600 hover:text-primary-700 font-medium"
                >
                  Voir tout →
                </a>
              </div>

              <!-- Statistiques selon l'espèce -->
              <div class="grid grid-cols-2 gap-1 text-sm mb-3">
                {@html config.renderStats(stats)}
              </div>

              <!-- Aperçu des animaux -->
              {#if data.length > 0}
                <div class="mt-3 pt-3 border-t border-gray-200/50">
                  <p class="text-xs text-gray-500 mb-2">Aperçu ({Math.min(data.length, 3)} affichés):</p>
                  <div class="space-y-1.5">
                    {#each getPreviewAnimals(data) as animal}
                      <div class="flex items-center justify-between text-xs bg-white/50 rounded-lg px-3 py-1.5">
                        <div class="flex items-center gap-2">
                          <span>{getSexeEmoji(animal.sexe)}</span>
                          <span class="font-medium text-gray-700">
                            {animal.identification || 'Sans ID'}
                          </span>
                        </div>
                        <div class="flex items-center gap-2">
                          <span class="text-gray-500">{animal.race}</span>
                          {#if animal.statut}
                            <span class={`px-1.5 py-0.5 rounded-full text-[10px] font-medium
                              ${animal.statut === 'vivant' ? 'bg-green-100 text-green-700' :
                                animal.statut === 'vendu' ? 'bg-blue-100 text-blue-700' :
                                animal.statut === 'decede' ? 'bg-red-100 text-red-700' :
                                'bg-gray-100 text-gray-700'}`}>
                              {animal.statut}
                            </span>
                          {/if}
                        </div>
                      </div>
                    {/each}
                    {#if data.length > 3}
                      <div class="text-xs text-gray-400 text-center">
                        +{data.length - 3} autres
                      </div>
                    {/if}
                  </div>
                </div>
              {/if}

              <!-- Bouton d'action -->
              <div class="mt-3 pt-3 border-t border-gray-200/50 flex justify-end">
                <a
                  href={`/${config.route}?enclos_id=${id}`}
                  class="text-xs px-3 py-1.5 rounded-lg bg-white/60 hover:bg-white text-gray-700 border border-gray-200 transition-colors"
                >
                  Gérer les {config.label.toLowerCase()}
                </a>
              </div>
            </div>
          {/each}
        </div>
      {/if}

      <!-- Message si aucun animal -->
      {#if bovins.length === 0 && ovins.length === 0 && caprins.length === 0 &&
        avicoles.length === 0 && piscicoles.length === 0 && ruches.length === 0 && lots.length === 0}
        <Card padding="lg" bordered={true}>
          <div class="text-center py-8">
            <div class="text-4xl mb-2">🐄</div>
            <p class="text-gray-500">Aucun animal présent dans cet enclos</p>
            <a href="/animaux/add?enclos_id={id}" class="text-primary-600 hover:text-primary-700 text-sm font-medium mt-2 inline-block">
              + Ajouter un animal
            </a>
          </div>
        </Card>
      {/if}

      <!-- Description -->
      {#if enclos.description}
        <Card title="📝 Description" padding="md" bordered={true} className="mt-6">
          <p class="text-gray-600 whitespace-pre-wrap">{enclos.description}</p>
        </Card>
      {/if}
    {/if}
  </div>
</div>