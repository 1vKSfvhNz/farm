<!-- src/lib/components/forms/AnimalVenteForm.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Button from "$lib/components/ui/Button.svelte";
    import Input from "$lib/components/ui/Input.svelte";
    import Textarea from "$lib/components/ui/Textarea.svelte";
    import type { AnimalBase } from "$lib/types/animal";
    import DatePicker from "../ui/DatePicker.svelte";

    export let animal: AnimalBase & { identification: string; };
    export let loading: boolean = false;
    export let espece: string = "animal"; // bovin, ovin, caprin

    const dispatch = createEventDispatcher();

    // Emojis par espèce
    const emojis: Record<string, string> = {
        bovin: "🐄",
        ovin: "🐑",
        caprin: "🐐"
    };

    let formData = {
        prix_vente: animal?.prix_vente || 0,
        date_vente: new Date().toISOString().split("T")[0],
        client_acheteur: animal?.client_acheteur || "",
        note_vente: animal?.note_vente || "",
        statut: "vendu" as const
    };

    // Validation
    let errors: Record<string, string> = {};
    let isSubmitting = false;

    function validate(): boolean {
        errors = {};
        
        if (formData.prix_vente <= 0) {
            errors.prix_vente = "Le prix de vente doit être supérieur à 0";
        }
        
        if (!formData.date_vente) {
            errors.date_vente = "La date de vente est requise";
        }
        
        return Object.keys(errors).length === 0;
    }

    async function handleSubmit() {
        if (isSubmitting) return;
        
        if (!validate()) {
            // Focus sur le premier champ en erreur
            const firstError = document.querySelector('[aria-invalid="true"]');
            if (firstError) {
                (firstError as HTMLElement).focus();
            }
            return;
        }

        isSubmitting = true;
        try {
            await dispatch("submit", formData);
        } finally {
            isSubmitting = false;
        }
    }

    function handleCancel() {
        dispatch("cancel");
    }

    // Calcul de la marge
    $: marge = formData.prix_vente - (animal?.prix_achat || 0);
    $: isMargePositive = marge > 0;
    $: isDisabled = loading || isSubmitting;
</script>

<div class="space-y-4">
    <!-- En-tête avec emoji -->
    <div class="flex items-center gap-2">
        <span class="text-2xl">{emojis[espece] || "🐄"}</span>
        <p class="text-sm text-gray-600">
            Enregistrement de la vente pour <strong>{animal?.identification}</strong>
        </p>
    </div>

    <!-- Prix de vente -->
    <Input
        label="💰 Prix de vente"
        inputType="number"
        step="100"
        min="0"
        bind:value={formData.prix_vente}
        required
        placeholder="Exemple: 250000"
        error={errors.prix_vente}
        disabled={isDisabled}
    />

    <!-- Date de vente -->
    <DatePicker
        label="📅 Date de vente"
        bind:value={formData.date_vente}
        required
        error={errors.date_vente}
        disabled={isDisabled}
    />

    <!-- Client acheteur -->
    <Input
        label="👤 Client acheteur"
        inputType="text"
        bind:value={formData.client_acheteur}
        placeholder="Nom du client ou de l'entreprise"
        disabled={isDisabled}
    />

    <!-- Notes - Utilisation du composant Textarea -->
    <Textarea
        label="📝 Note sur la vente"
        bind:value={formData.note_vente}
        placeholder="Informations complémentaires sur la vente..."
        rows={3}
        helpText="Optionnel - Conditions de vente, observations..."
        disabled={isDisabled}
    />

    <!-- Statistiques rapides -->
    {#if animal?.prix_achat}
        <div class="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <h4 class="text-sm font-medium text-gray-700 mb-2">📊 Résumé financier</h4>
            <div class="space-y-2 text-sm">
                <div class="flex justify-between">
                    <span class="text-gray-500">Prix d'achat:</span>
                    <span class="font-medium">{animal.prix_achat}</span>
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-500">Prix de vente:</span>
                    <span class="font-medium text-green-600">{formData.prix_vente}</span>
                </div>
                <div class="flex justify-between pt-2 border-t border-gray-200">
                    <span class="text-gray-500">Marge:</span>
                    <span class="font-semibold {isMargePositive ? 'text-green-600' : 'text-red-600'}">
                        {marge}
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
            </div>
        </div>
    {/if}

    <!-- Message d'avertissement -->
    <div class="bg-yellow-50 border-l-4 border-yellow-400 p-3 text-sm text-yellow-800 rounded-r">
        <p class="flex items-center gap-2">
            <span class="text-lg">⚠️</span>
            <span>Cette action marquera l'animal comme <strong>vendu</strong> et ne pourra plus être modifié.</span>
        </p>
    </div>

    <!-- Boutons -->
    <div class="flex justify-end gap-3 pt-2">
        <Button
            variant="outline"
            on:click={handleCancel}
            disabled={isDisabled}
        >
            Annuler
        </Button>
        <Button
            variant="success"
            on:click={handleSubmit}
            disabled={isDisabled}
            className="bg-green-600 hover:bg-green-700 disabled:opacity-50"
        >
            {isDisabled ? "Enregistrement..." : "💾 Enregistrer la vente"}
        </Button>
    </div>
</div>