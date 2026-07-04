<!-- src/lib/components/users/Infos.svelte -->
<!-- svelte-ignore a11y-label-has-associated-control -->
<script lang="ts">
    import { onMount } from "svelte";
    import { createEventDispatcher } from "svelte";
    import Card from "$lib/components/ui/Card.svelte";
    import Button from "$lib/components/ui/Button.svelte";
    import { usersApi } from "$lib/api/users";
    import { ROLE_LABELS } from "$lib/types/roles";
    import type { UserResponse, EmployeeStatus, EmployeeType } from "$lib/types/users";

    export let userId: number | null = null;
    export let user: UserResponse | null = null;

    const dispatch = createEventDispatcher<{
        close: void;
        edit: number;
        deactivate: number;
        activate: number;
    }>();

    let loading = true;
    let error = "";
    let internalUser: UserResponse | null = null;

    // Options pour les labels
    const employeeStatusLabels: Record<string, string> = {
        actif: "Actif",
        conge: "Congé",
        maladie: "Maladie",
        suspendu: "Suspendu",
        licencie: "Licencié",
        retraite: "Retraité",
        stagiaire: "Stagiaire"
    };

    const employeeTypeLabels: Record<string, string> = {
        permanent: "Permanent",
        stagiaire: "Stagiaire",
        contractuel: "Contractuel",
        saisonnier: "Saisonnier",
        consultant: "Consultant"
    };

    function getRoleLabel(role: string): string {
        return ROLE_LABELS[role as keyof typeof ROLE_LABELS] || role;
    }

    function formatDate(dateStr: string | null | undefined): string {
        if (!dateStr) return "-";
        try {
            return new Date(dateStr).toLocaleDateString('fr-FR', {
                day: '2-digit',
                month: '2-digit',
                year: 'numeric'
            });
        } catch {
            return dateStr;
        }
    }

    function formatDateTime(dateStr: string | null | undefined): string {
        if (!dateStr) return "-";
        try {
            return new Date(dateStr).toLocaleString('fr-FR', {
                day: '2-digit',
                month: '2-digit',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch {
            return dateStr;
        }
    }

    function formatCurrency(amount: number | null | undefined): string {
        if (!amount) return "-";
        return `${amount.toLocaleString('fr-FR')} FCFA`;
    }

    async function loadUser() {
        if (!userId) return;
        
        loading = true;
        error = "";
        try {
            const response = await usersApi.getUser(userId);
            internalUser = response;
            user = response;
        } catch (err: any) {
            error = err.message || "Erreur lors du chargement de l'utilisateur";
            console.error("Load user error:", err);
        } finally {
            loading = false;
        }
    }

    function handleClose() {
        dispatch('close');
    }

    function handleEdit() {
        if (internalUser) {
            dispatch('edit', internalUser.id);
        }
    }

    function handleToggleStatus() {
        if (internalUser) {
            if (internalUser.is_active) {
                dispatch('deactivate', internalUser.id);
            } else {
                dispatch('activate', internalUser.id);
            }
        }
    }

    $: {
        if (userId && !user) {
            loadUser();
        } else if (user) {
            internalUser = user;
            loading = false;
        }
    }
</script>

{#if loading}
    <div class="flex justify-center items-center py-12">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        <span class="ml-3 text-gray-600">Chargement...</span>
    </div>
{:else if error}
    <div class="bg-red-50 border border-red-200 rounded-lg p-4">
        <p class="text-red-700">❌ {error}</p>
        <Button on:click={loadUser} variant="outline" className="mt-2">
            Réessayer
        </Button>
    </div>
{:else if internalUser}
    <div class="space-y-4">
        <!-- En-tête -->
        <div class="flex items-start justify-between">
            <div>
                <h2 class="text-xl font-bold text-gray-900">{internalUser.full_name}</h2>
                <p class="text-sm text-gray-500">
                    {internalUser.employee_id ? `Matricule: ${internalUser.employee_id}` : 'Aucun matricule'}
                </p>
            </div>
            <div class="flex items-center gap-2">
                <span class="px-3 py-1 rounded-full text-sm font-medium {internalUser.is_active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                    {internalUser.is_active ? 'Actif' : 'Inactif'}
                </span>
                <Button on:click={handleClose} variant="ghost" size="sm">
                    ✕
                </Button>
            </div>
        </div>

        <!-- Informations de base -->
        <Card title="Informations personnelles" className="p-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Email</label>
                    <p class="text-gray-900">{internalUser.email}</p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Téléphone</label>
                    <p class="text-gray-900">{internalUser.phone || "-"}</p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Nom d'utilisateur</label>
                    <p class="text-gray-900">{internalUser.username}</p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Rôles</label>
                    <div class="flex flex-wrap gap-1 mt-1">
                        {#if internalUser.roles && internalUser.roles.length > 0}
                            {#each internalUser.roles as role}
                                <span class="px-2 py-0.5 rounded-full text-xs bg-gray-100 text-gray-700">
                                    {getRoleLabel(role)}
                                </span>
                            {/each}
                        {:else}
                            <span class="text-gray-500">-</span>
                        {/if}
                    </div>
                </div>
            </div>
        </Card>

        <!-- Informations professionnelles -->
        <Card title="Informations professionnelles" className="p-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Département</label>
                    <p class="text-gray-900">{internalUser.department || "-"}</p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Date d'embauche</label>
                    <p class="text-gray-900">{formatDate(internalUser.hire_date)}</p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Statut employé</label>
                    <p class="text-gray-900">
                        {internalUser.employee_status ? employeeStatusLabels[internalUser.employee_status] || internalUser.employee_status : "-"}
                    </p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Type d'employé</label>
                    <p class="text-gray-900">
                        {internalUser.employee_type ? employeeTypeLabels[internalUser.employee_type] || internalUser.employee_type : "-"}
                    </p>
                </div>
            </div>
        </Card>

        <!-- Salaire -->
        <Card title="Salaire et rémunération" className="p-4">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Salaire de base</label>
                    <p class="text-gray-900">{formatCurrency(internalUser.base_salary)}</p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Devise</label>
                    <p class="text-gray-900">{internalUser.salary_currency || "XOF"}</p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Prime/Bonus</label>
                    <p class="text-gray-900">{formatCurrency(internalUser.bonus)}</p>
                </div>
            </div>
        </Card>

        <!-- Informations bancaires -->
        <Card title="Informations bancaires" className="p-4">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Banque</label>
                    <p class="text-gray-900">{internalUser.bank_name || "-"}</p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Numéro de compte</label>
                    <p class="text-gray-900">{internalUser.bank_account || "-"}</p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">RIB</label>
                    <p class="text-gray-900">{internalUser.rib || "-"}</p>
                </div>
            </div>
        </Card>

        <!-- Informations administratives -->
        <Card title="Informations administratives" className="p-4">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">CNI/Passeport</label>
                    <p class="text-gray-900">{internalUser.national_id || "-"}</p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Sécurité sociale</label>
                    <p class="text-gray-900">{internalUser.social_security_number || "-"}</p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Numéro fiscal</label>
                    <p class="text-gray-900">{internalUser.tax_id || "-"}</p>
                </div>
            </div>
        </Card>

        <!-- Contact d'urgence -->
        <Card title="Contact d'urgence" className="p-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Nom</label>
                    <p class="text-gray-900">{internalUser.emergency_contact_name || "-"}</p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Téléphone</label>
                    <p class="text-gray-900">{internalUser.emergency_contact_phone || "-"}</p>
                </div>
            </div>
        </Card>

        <!-- Observations -->
        {#if internalUser.observations}
            <Card title="Observations" className="p-4">
                <p class="text-gray-900 whitespace-pre-wrap">{internalUser.observations}</p>
            </Card>
        {/if}

        <!-- Métadonnées -->
        <Card title="Métadonnées" className="p-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Créé le</label>
                    <p class="text-gray-900">{formatDateTime(internalUser.created_at)}</p>
                </div>
                <div>
                    <label class="text-xs font-medium text-gray-500 uppercase tracking-wider">Modifié le</label>
                    <p class="text-gray-900">{formatDateTime(internalUser.updated_at)}</p>
                </div>
            </div>
        </Card>

        <!-- Actions -->
        <div class="flex items-center gap-3 pt-4 border-t">
            <Button on:click={handleEdit} variant="primary">
                ✏️ Modifier
            </Button>
            <Button 
                on:click={handleToggleStatus}
                variant={internalUser.is_active ? "warning" : "success"}
            >
                {internalUser.is_active ? '🔒 Désactiver' : '🔓 Réactiver'}
            </Button>
            <Button on:click={handleClose} variant="ghost">
                Fermer
            </Button>
        </div>
    </div>
{:else}
    <div class="text-center py-12 text-gray-500">
        <p>Aucun utilisateur sélectionné</p>
    </div>
{/if}