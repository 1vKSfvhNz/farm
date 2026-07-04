<!-- src/routes/ovins/[id]/+page.svelte -->
<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import { goto } from "$app/navigation";
    import { ovinsStore } from "$lib/stores/ovins";
    import { ovinsApi } from "$lib/api/ovins";
    import { peseeApi } from "$lib/api/pesee";
    import { permissionsStore } from "$lib/stores/permissions";
    import Card from "$lib/components/ui/Card.svelte";
    import Button from "$lib/components/ui/Button.svelte";
    import Spinner from "$lib/components/ui/Spinner.svelte";
    import LineChart from "$lib/components/charts/LineChart.svelte";
    import Modal from "$lib/components/ui/Modal.svelte";
    import PeseeForm from "$lib/components/forms/PeseeForm.svelte";
    import OvinForm from "$lib/components/forms/OvinForm.svelte";
    import AlertBadge from "$lib/components/ui/AlertBadge.svelte";
    import AnimalVenteForm from "$lib/components/forms/AnimalVenteForm.svelte";
    import AnimalVenteInfo from "$lib/components/ui/AnimalVenteInfo.svelte";
    import type { OvinResponse } from "$lib/types/ovin";
    import type { PeseeResponse } from "$lib/types/pesee";
    import { calculateAge, getStatutBadge } from "$lib/stores/animal";
    import { API_URL } from "$lib/api/client";

    // ✅ Récupérer les données depuis le loader (+page.ts)
    export let data: {
        id: string;
        ovin: OvinResponse;
    };
    
    let id: number = parseInt(data.id);
    let ovin: OvinResponse = data.ovin;
    let pesees: PeseeResponse[] = [];
    let loading = true;
    let showPeseeModal = false;
    let showEditModal = false;
    let showVenteModal = false;

    const canEdit = permissionsStore.canEditEspece("ovins");

    // ✅ S'abonner au store pour les mises à jour
    const unsubscribe = ovinsStore.subscribe((state) => {
        if (state && typeof state === 'object' && 'id' in state) {
            const found = state as OvinResponse;
            if (found.id === id) {
                ovin = found;
                console.log("📦 Ovin mis à jour depuis le store:", ovin);
            }
        }
    });

    onMount(async () => {
        await loadData();
    });

    async function loadData() {
        loading = true;
        console.log("📊 Chargement des pesées pour l'ID:", id);
        
        try {
            const peseesData = await peseeApi.getPesees({ animal_id: id, limit: 100 });
            pesees = peseesData.items || [];
            
            console.log("✅ Ovin:", ovin);
            console.log("✅ Pesées chargées:", pesees.length);
        } catch (error: any) {
            console.error("❌ Failed to load pesees:", error);
        } finally {
            loading = false;
        }
    }

    async function handleAddPesee(event: CustomEvent) {
        const formData = event.detail;
        console.log("Données reçues:", formData);

        try {
            await peseeApi.createPesee({ ...formData, animal_id: id });
            showPeseeModal = false;
            await loadData();
        } catch (error) {
            console.error("❌ Failed to add pesee:", error);
        }
    }

    async function handleEditSubmit(event: CustomEvent) {
        const formData = event.detail;
        console.log("📝 Données d'édition reçues:", formData);
        
        try {
            const updated = await ovinsApi.updateOvin(id, formData);
            ovinsStore.setOvin(updated);
            ovin = updated;
            showEditModal = false;
            console.log("✅ Ovin mis à jour avec succès");
        } catch (error) {
            console.error("❌ Failed to update ovin:", error);
        }
    }

    // === FONCTION POUR ENREGISTRER UNE VENTE ===
    async function handleVenteSubmit(event: CustomEvent) {
        const formData = event.detail;
        console.log("💰 Données de vente reçues:", formData);
        
        try {
            const updated = await ovinsApi.enregistrerVente(id, formData);
            ovinsStore.setOvin(updated);
            ovin = updated;
            showVenteModal = false;
            console.log("✅ Vente enregistrée avec succès");
        } catch (error: any) {
            console.error("❌ Erreur lors de l'enregistrement:", error);
            alert(error?.message || "Erreur lors de l'enregistrement de la vente");
        }
    }

    function handleEdit() {
        showEditModal = true;
    }

    function handleVente() {
        showVenteModal = true;
    }

    function formatDate(date: string): string {
        return new Date(date).toLocaleDateString("fr-FR");
    }

    function formatCurrency(amount: number): string {
        return new Intl.NumberFormat("fr-FR", {
            style: "currency",
            currency: "XOF",
            minimumFractionDigits: 0,
        }).format(amount);
    }

    function getPhotoUrl(photoUrl: string | null | undefined): string | null {
        if (!photoUrl) return null;
        if (photoUrl.startsWith('http://') || photoUrl.startsWith('https://')) {
            return photoUrl;
        }
        return `${API_URL.replace('/api/v1', '')}${photoUrl}`;
    }

    function handleImageError(event: Event) {
        const target = event.currentTarget as HTMLImageElement;
        target.style.display = 'none';
        const fallback = target.nextElementSibling as HTMLElement;
        if (fallback) {
            fallback.classList.remove('hidden');
        }
    }

    // ✅ Préparer les données du graphique
    $: chartLabels = pesees.map((p) => formatDate(p.date_pesee));
    $: chartData = pesees.map((p) => p.poids);

    // ✅ Vérifier si l'animal est vendu
    $: isVendu = ovin?.statut === 'vendu';

    // ✅ Nettoyer l'abonnement
    onDestroy(() => {
        unsubscribe();
    });
</script>

<svelte:head>
    <title>Ovin {id} - Farm Manager</title>
</svelte:head>

<div class="space-y-6">
    {#if loading}
        <div class="flex justify-center items-center h-64">
            <Spinner size="lg" />
        </div>
    {:else if !ovin}
        <div class="text-center py-12">
            <p class="text-gray-500">Ovin non trouvé</p>
            <Button on:click={() => goto("/ovins")} variant="primary" className="mt-4">
                Retour à la liste
            </Button>
        </div>
    {:else}
        <!-- En-tête -->
        <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div>
                <h1 class="text-2xl font-bold text-gray-900 flex items-center gap-3">
                    {ovin.identification}
                    {#if ovin.photo_url}
                        <img 
                            src={getPhotoUrl(ovin.photo_url)} 
                            alt={ovin.identification}
                            class="w-12 h-12 rounded-full object-cover border-2 border-gray-200"
                            on:error={handleImageError}
                        />
                    {:else}
                        <div class="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center text-2xl">
                            🐑
                        </div>
                    {/if}
                </h1>
                <p class="text-sm text-gray-500 mt-1">
                    Race: {ovin.race || "Non renseignée"} | 
                    Sexe: {ovin.sexe === "male" ? "♂ Bélier" : ovin.sexe === "femelle" ? "♀ Brebis" : "⚥ Hermaphrodite"}
                </p>
            </div>
            <div class="flex gap-2 flex-wrap">
                {#if canEdit && !isVendu}
                    <Button on:click={handleEdit} variant="outline">
                        ✏️ Modifier
                    </Button>
                    <Button on:click={handleVente} variant="success" className="bg-green-600 hover:bg-green-700">
                        💰 Vendre
                    </Button>
                {/if}
                <Button on:click={() => goto("/ovins")} variant="outline">
                    ← Retour
                </Button>
            </div>
        </div>

        <!-- Informations générales -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card title="📋 Informations générales">
                <div class="space-y-3 text-sm">
                    <div class="flex justify-between items-center">
                        <span class="text-gray-500">Statut:</span>
                        {#if ovin}
                            {@const badge = getStatutBadge(ovin.statut)}
                            <AlertBadge niveau={badge.niveau} label={badge.label} size="sm" />
                        {/if}
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Âge:</span>
                        <span class="font-medium">{calculateAge(ovin.date_naissance)}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Date d'arrivée:</span>
                        <span class="font-medium">{formatDate(ovin.date_arrivee)}</span>
                    </div>
                    {#if ovin.date_naissance}
                        <div class="flex justify-between">
                            <span class="text-gray-500">Date naissance:</span>
                            <span class="font-medium">{formatDate(ovin.date_naissance)}</span>
                        </div>
                    {/if}
                    {#if ovin.provenance}
                        <div class="flex justify-between">
                            <span class="text-gray-500">Provenance:</span>
                            <span class="font-medium">{ovin.provenance}</span>
                        </div>
                    {/if}
                    {#if ovin.prix_achat}
                        <div class="flex justify-between">
                            <span class="text-gray-500">Prix d'achat:</span>
                            <span class="font-medium">{formatCurrency(ovin.prix_achat)}</span>
                        </div>
                    {/if}
                    {#if ovin.enclos_id}
                        <div class="flex justify-between">
                            <span class="text-gray-500">Enclos:</span>
                            <span class="font-medium">{ovin.enclos_id}</span>
                        </div>
                    {/if}
                    {#if ovin.poids_initial}
                        <div class="flex justify-between">
                            <span class="text-gray-500">Poids initial:</span>
                            <span class="font-medium">{ovin.poids_initial} kg</span>
                        </div>
                    {/if}
                    {#if isVendu}
                        <div class="mt-2 pt-2 border-t border-gray-200">
                            <span class="text-sm text-green-600 font-semibold">✅ Animal vendu</span>
                        </div>
                    {/if}
                </div>
            </Card>

            <Card title="🏭 Production">
                <div class="space-y-3 text-sm">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Production viande:</span>
                        <span class="font-medium">{ovin.production_viande ? "✅ Oui" : "❌ Non"}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Production laine:</span>
                        <span class="font-medium">{ovin.production_laine ? "✅ Oui" : "❌ Non"}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Reproduction:</span>
                        <span class="font-medium">{ovin.production_reproduction ? "✅ Oui" : "❌ Non"}</span>
                    </div>
                    {#if ovin.qualite_laine}
                        <div class="flex justify-between">
                            <span class="text-gray-500">Qualité laine:</span>
                            <span class="font-medium">{ovin.qualite_laine}</span>
                        </div>
                    {/if}
                </div>
            </Card>

            <!-- Carte conditionnelle : Vente ou Dernière pesée -->
            {#if isVendu}
                <AnimalVenteInfo animal={ovin} espece="ovin" showMarge={true} />
            {:else}
                <Card title="⚖️ Dernière pesée">
                    <div class="space-y-3 text-sm">
                        {#if pesees.length > 0}
                            {@const dernierePesee = pesees[pesees.length - 1]}
                            <div class="flex justify-between items-baseline">
                                <span class="text-gray-500">Poids:</span>
                                <span class="font-medium text-lg">{dernierePesee.poids} kg</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-500">Date:</span>
                                <span class="font-medium">{formatDate(dernierePesee.date_pesee)}</span>
                            </div>
                            {#if dernierePesee.methode}
                                <div class="flex justify-between">
                                    <span class="text-gray-500">Méthode:</span>
                                    <span class="font-medium">{dernierePesee.methode}</span>
                                </div>
                            {/if}
                            {#if dernierePesee.notes}
                                <div class="flex justify-between">
                                    <span class="text-gray-500">Notes:</span>
                                    <span class="font-medium">{dernierePesee.notes}</span>
                                </div>
                            {/if}
                            {#if pesees.length > 1}
                                <div class="flex justify-between pt-2 border-t border-gray-100">
                                    <span class="text-gray-500">Nombre total:</span>
                                    <span class="font-medium">{pesees.length} pesées</span>
                                </div>
                            {/if}
                        {:else}
                            <p class="text-gray-500 text-center py-4">Aucune pesée enregistrée</p>
                        {/if}
                        {#if canEdit && !isVendu}
                            <div class="mt-4 pt-2">
                                <Button
                                    on:click={() => (showPeseeModal = true)}
                                    variant="primary"
                                    size="sm"
                                    fullWidth
                                >
                                    ➕ Ajouter une pesée
                                </Button>
                            </div>
                        {/if}
                    </div>
                </Card>
            {/if}
        </div>

        <!-- Graphique de croissance - Masqué si vendu -->
        {#if pesees.length > 1 && !isVendu}
            <Card title="📈 Courbe de croissance">
                <LineChart
                    labels={chartLabels}
                    datasets={[{ label: "Poids (kg)", data: chartData, color: "#3b82f6" }]}
                    yAxisLabel="Poids (kg)"
                    height={350}
                />
            </Card>
        {/if}

        <!-- Historique des pesées - Masqué si vendu -->
        {#if pesees.length > 0 && !isVendu}
            <Card title="📋 Historique des pesées">
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-4 py-3 text-left font-medium text-gray-500">Date</th>
                                <th class="px-4 py-3 text-left font-medium text-gray-500">Poids (kg)</th>
                                <th class="px-4 py-3 text-left font-medium text-gray-500">Méthode</th>
                                <th class="px-4 py-3 text-left font-medium text-gray-500">Notes</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-100">
                            {#each [...pesees].reverse() as pesee}
                                <tr class="hover:bg-gray-50">
                                    <td class="px-4 py-3">{formatDate(pesee.date_pesee)}</td>
                                    <td class="px-4 py-3 font-semibold">{pesee.poids} kg</td>
                                    <td class="px-4 py-3">{pesee.methode || "-"}</td>
                                    <td class="px-4 py-3 text-gray-500">{pesee.notes || "-"}</td>
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                </div>
            </Card>
        {/if}

        <!-- Notes -->
        {#if ovin.notes}
            <Card title="📝 Notes">
                <p class="text-gray-600 whitespace-pre-wrap">{ovin.notes}</p>
            </Card>
        {/if}
    {/if}

    <!-- Modal ajout pesée -->
    <Modal
        open={showPeseeModal}
        title="⚖️ Ajouter une pesée"
        on:close={() => (showPeseeModal = false)}
        size="md"
    >
        <PeseeForm
            formData={{
                animal_id: id,
                date_pesee: new Date().toISOString().split("T")[0]
            }}
            loading={false}
            on:submit={handleAddPesee}
            on:cancel={() => (showPeseeModal = false)}
        />
    </Modal>

    <!-- Modal édition ovin -->
    <Modal
        open={showEditModal}
        title="✏️ Modifier l'ovin"
        on:close={() => (showEditModal = false)}
        size="lg"
    >
        <OvinForm
            formData={ovin ? {
                type_espece: 'ovin',
                race: ovin.race,
                sexe: ovin.sexe,
                date_naissance: ovin.date_naissance || null,
                date_arrivee: ovin.date_arrivee,
                provenance: ovin.provenance,
                prix_achat: ovin.prix_achat || undefined,
                enclos_id: ovin.enclos_id || 0,
                statut: ovin.statut,
                production_viande: ovin.production_viande,
                production_reproduction: ovin.production_reproduction,
                production_laine: ovin.production_laine,
                qualite_laine: ovin.qualite_laine || "",
                notes: ovin.notes || undefined,
                photo_url: ovin.photo_url || null
            } : {
                type_espece: "ovin",
                race: "",
                sexe: "male",
                date_naissance: null,
                date_arrivee: new Date().toISOString().split("T")[0],
                provenance: "",
                prix_achat: undefined,
                enclos_id: 0,
                statut: "vivant",
                production_viande: false,
                production_reproduction: false,
                production_laine: false,
                qualite_laine: "",
                notes: "",
                photo_url: null
            }}
            loading={false}
            isEdit={true}
            on:submit={handleEditSubmit}
            on:cancel={() => (showEditModal = false)}
        />
    </Modal>

    <!-- Modal vente -->
    <Modal
        open={showVenteModal}
        title="💰 Enregistrer la vente"
        on:close={() => (showVenteModal = false)}
        size="md"
    >
        <AnimalVenteForm
            animal={ovin}
            espece="ovin"
            loading={false}
            on:submit={handleVenteSubmit}
            on:cancel={() => (showVenteModal = false)}
        />
    </Modal>
</div>