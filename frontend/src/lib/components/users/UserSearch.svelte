<!-- src/lib/components/users/UserSearch.svelte -->
<!-- svelte-ignore a11y-label-has-associated-control -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Input from "$lib/components/ui/Input.svelte";
    import Button from "$lib/components/ui/Button.svelte";
    import Card from "$lib/components/ui/Card.svelte";
    import { ROLE_GROUPS, ROLE_LABELS } from "$lib/types/roles";
    import type { UserSearchParams } from "$lib/types/users";
    import DatePicker from "../ui/DatePicker.svelte";

    export let loading = false;
    export const searchParams: UserSearchParams = {};

    const dispatch = createEventDispatcher<{
        search: UserSearchParams;
        reset: void;
    }>();

    // Options pour les selects
    const employeeStatusOptions = [
        { value: "actif", label: "Actif" },
        { value: "conge", label: "Congé" },
        { value: "maladie", label: "Maladie" },
        { value: "suspendu", label: "Suspendu" },
        { value: "licencie", label: "Licencié" },
        { value: "retraite", label: "Retraité" },
        { value: "stagiaire", label: "Stagiaire" }
    ];

    const employeeTypeOptions = [
        { value: "permanent", label: "Permanent" },
        { value: "stagiaire", label: "Stagiaire" },
        { value: "contractuel", label: "Contractuel" },
        { value: "saisonnier", label: "Saisonnier" },
        { value: "consultant", label: "Consultant" }
    ];

    const departmentOptions = [
        "Élevage", "Administration", "Comptabilité", "Vétérinaire",
        "Maintenance", "Recherche", "Formation", "Logistique"
    ];

    // État local des filtres
    let filters = {
        search: "",
        roles: [] as string[],
        is_active: undefined as boolean | undefined,
        employee_id: "",
        department: "",
        employee_status: [] as string[],
        employee_type: [] as string[],
        hire_date_from: "",
        hire_date_to: "",
        salary_min: "",
        salary_max: "",
        created_from: "",
        created_to: "",
        order_by: "created_at",
        order_direction: "desc" as "asc" | "desc"
    };

    // Groupes de rôles pour le filtrage
    const roleGroups = [
        { label: "Administration", roles: ROLE_GROUPS.admin },
        { label: "Techniciens", roles: ROLE_GROUPS.technician },
        { label: "Observateurs", roles: ROLE_GROUPS.observer },
        { label: "Rôles transverses", roles: ROLE_GROUPS.transverse },
    ];

    function getRoleLabel(role: string): string {
        return ROLE_LABELS[role as keyof typeof ROLE_LABELS] || role;
    }

    // Appliquer les filtres
    function applyFilters() {
        const params: UserSearchParams = {
            search: filters.search || undefined,
            roles: filters.roles.length > 0 ? filters.roles : undefined,
            is_active: filters.is_active,
            employee_id: filters.employee_id || undefined,
            department: filters.department || undefined,
            employee_status: filters.employee_status.length > 0 ? filters.employee_status : undefined,
            employee_type: filters.employee_type.length > 0 ? filters.employee_type : undefined,
            hire_date_from: filters.hire_date_from || undefined,
            hire_date_to: filters.hire_date_to || undefined,
            salary_min: filters.salary_min ? parseFloat(filters.salary_min) : undefined,
            salary_max: filters.salary_max ? parseFloat(filters.salary_max) : undefined,
            created_from: filters.created_from || undefined,
            created_to: filters.created_to || undefined,
            order_by: filters.order_by,
            order_direction: filters.order_direction
        };
        dispatch('search', params);
    }

    // Réinitialiser les filtres
    function resetFilters() {
        filters = {
            search: "",
            roles: [],
            is_active: undefined,
            employee_id: "",
            department: "",
            employee_status: [],
            employee_type: [],
            hire_date_from: "",
            hire_date_to: "",
            salary_min: "",
            salary_max: "",
            created_from: "",
            created_to: "",
            order_by: "created_at",
            order_direction: "desc"
        };
        dispatch('reset');
    }

    // Toggle un rôle
    function toggleRole(roleValue: string) {
        if (filters.roles.includes(roleValue)) {
            filters.roles = filters.roles.filter(r => r !== roleValue);
        } else {
            filters.roles = [...filters.roles, roleValue];
        }
    }

    // Toggle un statut employé
    function toggleEmployeeStatus(status: string) {
        if (filters.employee_status.includes(status)) {
            filters.employee_status = filters.employee_status.filter(s => s !== status);
        } else {
            filters.employee_status = [...filters.employee_status, status];
        }
    }

    // Toggle un type employé
    function toggleEmployeeType(type: string) {
        if (filters.employee_type.includes(type)) {
            filters.employee_type = filters.employee_type.filter(t => t !== type);
        } else {
            filters.employee_type = [...filters.employee_type, type];
        }
    }

    // Nombre de filtres actifs
    $: activeFilterCount = 
        (filters.search ? 1 : 0) +
        (filters.roles.length > 0 ? 1 : 0) +
        (filters.is_active !== undefined ? 1 : 0) +
        (filters.employee_id ? 1 : 0) +
        (filters.department ? 1 : 0) +
        (filters.employee_status.length > 0 ? 1 : 0) +
        (filters.employee_type.length > 0 ? 1 : 0) +
        (filters.hire_date_from ? 1 : 0) +
        (filters.hire_date_to ? 1 : 0) +
        (filters.salary_min ? 1 : 0) +
        (filters.salary_max ? 1 : 0) +
        (filters.created_from ? 1 : 0) +
        (filters.created_to ? 1 : 0);

    // Champ de tri
    const orderByOptions = [
        { value: "created_at", label: "Date de création" },
        { value: "updated_at", label: "Date de mise à jour" },
        { value: "full_name", label: "Nom complet" },
        { value: "email", label: "Email" },
        { value: "username", label: "Nom d'utilisateur" },
        { value: "employee_id", label: "Matricule" },
        { value: "department", label: "Département" },
        { value: "hire_date", label: "Date d'embauche" },
        { value: "base_salary", label: "Salaire" },
        { value: "is_active", label: "Statut" }
    ];

    // Fonction pour gérer la touche Entrée - CORRECTION AVEC CustomEvent
    function handleKeyDown(event: CustomEvent) {
        const keyboardEvent = event.detail as KeyboardEvent;
        if (keyboardEvent.key === "Enter") {
            applyFilters();
        }
    }

    // Exporter les fonctions pour le parent
    export { applyFilters, resetFilters };
</script>

<Card title="Recherche avancée">
    <div class="space-y-4">
        <!-- Barre de recherche principale -->
        <div class="flex gap-3">
            <div class="flex-1">
                <Input
                    bind:value={filters.search}
                    placeholder="🔍 Rechercher par nom, email, matricule..."
                    on:keydown={handleKeyDown}
                />
            </div>
            <Button on:click={applyFilters} loading={loading} variant="primary">
                Rechercher
            </Button>
            <Button on:click={resetFilters} variant="outline" disabled={activeFilterCount === 0}>
                Réinitialiser
            </Button>
        </div>

        <!-- Filtres avancés -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <!-- Colonne 1 -->
            <div class="space-y-3">
                <!-- Statut compte -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        Statut du compte
                    </label>
                    <div class="flex gap-2">
                        <label class="flex items-center space-x-2 cursor-pointer">
                            <input
                                type="radio"
                                checked={filters.is_active === true}
                                on:change={() => filters.is_active = true}
                                class="text-primary-600 focus:ring-primary-500"
                            />
                            <span class="text-sm">Actif</span>
                        </label>
                        <label class="flex items-center space-x-2 cursor-pointer">
                            <input
                                type="radio"
                                checked={filters.is_active === false}
                                on:change={() => filters.is_active = false}
                                class="text-primary-600 focus:ring-primary-500"
                            />
                            <span class="text-sm">Inactif</span>
                        </label>
                        <label class="flex items-center space-x-2 cursor-pointer">
                            <input
                                type="radio"
                                checked={filters.is_active === undefined}
                                on:change={() => filters.is_active = undefined}
                                class="text-primary-600 focus:ring-primary-500"
                            />
                            <span class="text-sm">Tous</span>
                        </label>
                    </div>
                </div>

                <!-- Matricule -->
                <Input
                    label="Matricule"
                    bind:value={filters.employee_id}
                    placeholder="EMP-001"
                />

                <!-- Département -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        Département
                    </label>
                    <select
                        bind:value={filters.department}
                        class="w-full rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
                    >
                        <option value="">Tous les départements</option>
                        {#each departmentOptions as dept}
                            <option value={dept}>{dept}</option>
                        {/each}
                    </select>
                </div>

                <!-- Date d'embauche -->
                <div class="grid grid-cols-2 gap-2">
                    <DatePicker
                        label="Embauche (de)"
                        bind:value={filters.hire_date_from}
                    />
                    <DatePicker
                        label="Embauche (à)"
                        bind:value={filters.hire_date_to}
                    />
                </div>

                <!-- Salaire -->
                <div class="grid grid-cols-2 gap-2">
                    <Input
                        label="Salaire min"
                        bind:value={filters.salary_min}
                        inputType="number"
                        placeholder="0"
                    />
                    <Input
                        label="Salaire max"
                        bind:value={filters.salary_max}
                        inputType="number"
                        placeholder="1000000"
                    />
                </div>
            </div>

            <!-- Colonne 2 -->
            <div class="space-y-3">
                <!-- Rôles -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        Rôles
                    </label>
                    <div class="max-h-40 overflow-y-auto p-2 border border-gray-300 rounded-lg space-y-2">
                        {#each roleGroups as group}
                            {#if group.roles.length > 0}
                                <div>
                                    <div class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
                                        {group.label}
                                    </div>
                                    <div class="space-y-1 pl-2">
                                        {#each group.roles as roleValue}
                                            <label class="flex items-center space-x-2 text-sm hover:bg-gray-50 p-1 rounded cursor-pointer">
                                                <input
                                                    type="checkbox"
                                                    checked={filters.roles.includes(roleValue)}
                                                    on:change={() => toggleRole(roleValue)}
                                                    class="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                                                />
                                                <span>{getRoleLabel(roleValue)}</span>
                                            </label>
                                        {/each}
                                    </div>
                                </div>
                            {/if}
                        {/each}
                    </div>
                </div>

                <!-- Statut employé -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        Statut employé
                    </label>
                    <div class="flex flex-wrap gap-2">
                        {#each employeeStatusOptions as opt}
                            <label class="flex items-center space-x-1 cursor-pointer text-sm">
                                <input
                                    type="checkbox"
                                    checked={filters.employee_status.includes(opt.value)}
                                    on:change={() => toggleEmployeeStatus(opt.value)}
                                    class="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                                />
                                <span>{opt.label}</span>
                            </label>
                        {/each}
                    </div>
                </div>

                <!-- Type employé -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        Type d'employé
                    </label>
                    <div class="flex flex-wrap gap-2">
                        {#each employeeTypeOptions as opt}
                            <label class="flex items-center space-x-1 cursor-pointer text-sm">
                                <input
                                    type="checkbox"
                                    checked={filters.employee_type.includes(opt.value)}
                                    on:change={() => toggleEmployeeType(opt.value)}
                                    class="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                                />
                                <span>{opt.label}</span>
                            </label>
                        {/each}
                    </div>
                </div>

                <!-- Dates de création -->
                <div class="grid grid-cols-2 gap-2">
                    <DatePicker
                        label="Créé (de)"
                        bind:value={filters.created_from}
                    />
                    <DatePicker
                        label="Créé (à)"
                        bind:value={filters.created_to}
                    />
                </div>
            </div>
        </div>

        <!-- Tri -->
        <div class="flex items-center gap-4 pt-3 border-t">
            <label class="text-sm font-medium text-gray-700">Trier par :</label>
            <select
                bind:value={filters.order_by}
                class="rounded-lg border border-gray-300 px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
                {#each orderByOptions as opt}
                    <option value={opt.value}>{opt.label}</option>
                {/each}
            </select>
            <select
                bind:value={filters.order_direction}
                class="rounded-lg border border-gray-300 px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
                <option value="asc">Croissant</option>
                <option value="desc">Décroissant</option>
            </select>
            <span class="text-sm text-gray-500 ml-auto">
                {activeFilterCount} filtre{activeFilterCount > 1 ? 's' : ''} actif{activeFilterCount > 1 ? 's' : ''}
            </span>
        </div>
    </div>
</Card>