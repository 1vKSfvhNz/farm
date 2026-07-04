<!-- src/lib/components/ui/AnimalVenteInfo.svelte -->
<script lang="ts">
    import Card from "$lib/components/ui/Card.svelte";
    import type { AnimalBase } from "$lib/types/animal";

    export let animal: AnimalBase & { 
        identification: string; 
        prix_achat?: number | null;
        prix_vente?: number | null;
        date_vente?: string | null;
        client_acheteur?: string | null;
        note_vente?: string | null;
    };
    export const espece: string = 'animal';
    export let showMarge: boolean = true;

    const emojis: Record<string, string> = {
        bovin: "🐄",
        ovin: "🐑",
        caprin: "🐐"
    };

    function formatDate(date: string | null | undefined): string {
        if (!date) return "-";
        return new Date(date).toLocaleDateString("fr-FR", {
            day: "2-digit",
            month: "2-digit",
            year: "numeric"
        });
    }

    function formatCurrency(amount: number | null | undefined): string {
        if (!amount && amount !== 0) return "-";
        return new Intl.NumberFormat("fr-FR", {
            style: "currency",
            currency: "XOF",
            minimumFractionDigits: 0,
        }).format(amount);
    }

    $: marge = (animal?.prix_vente || 0) - (animal?.prix_achat || 0);
    $: isMargePositive = marge > 0;
    $: hasVenteInfo = animal?.prix_vente || animal?.date_vente || animal?.client_acheteur;
</script>

{#if hasVenteInfo}
    <Card title="💰 Informations de vente">
        <div class="space-y-3 text-sm">
            {#if animal.prix_vente}
                <div class="flex justify-between">
                    <span class="text-gray-500">Prix de vente:</span>
                    <span class="font-medium text-green-700">{formatCurrency(animal.prix_vente)}</span>
                </div>
            {/if}

            {#if animal.date_vente}
                <div class="flex justify-between">
                    <span class="text-gray-500">Date de vente:</span>
                    <span class="font-medium">{formatDate(animal.date_vente)}</span>
                </div>
            {/if}

            {#if animal.client_acheteur}
                <div class="flex justify-between">
                    <span class="text-gray-500">Client:</span>
                    <span class="font-medium">{animal.client_acheteur}</span>
                </div>
            {/if}

            {#if animal.note_vente}
                <div class="mt-2 pt-2 border-t border-gray-100">
                    <span class="text-gray-500 text-xs block mb-1">📝 Note :</span>
                    <p class="text-gray-700 text-sm whitespace-pre-wrap">{animal.note_vente}</p>
                </div>
            {/if}

            {#if showMarge && animal.prix_achat && animal.prix_vente}
                <div class="flex justify-between pt-2 border-t border-gray-100">
                    <span class="text-gray-500">Marge:</span>
                    <span class="font-semibold {isMargePositive ? 'text-green-600' : 'text-red-600'}">
                        {formatCurrency(marge)}
                        {#if isMargePositive}
                            <span class="text-xs">✅</span>
                        {:else}
                            <span class="text-xs">⚠️</span>
                        {/if}
                    </span>
                </div>
                {#if isMargePositive && animal.prix_achat > 0}
                    <div class="flex justify-between text-xs text-gray-500">
                        <span>Rentabilité:</span>
                        <span>{(marge / animal.prix_achat * 100).toFixed(1)}%</span>
                    </div>
                {/if}
            {/if}
        </div>
    </Card>
{/if}