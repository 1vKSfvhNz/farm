<!-- src/routes/utilisateurs/+page.svelte - Version avec recherche avancée et Infos -->
<!-- svelte-ignore a11y-label-has-associated-control -->
<script lang="ts">
    import { onMount } from "svelte";
    import { permissionsStore } from "$stores/permissions";
    import DataTable from "$components/tables/DataTable.svelte";
    import Button from "$components/ui/Button.svelte";
    import Modal from "$components/ui/Modal.svelte";
    import Input from "$components/ui/Input.svelte";
    import ConfirmDialog from "$components/ui/ConfirmDialog.svelte";
    import UserSearch from "$lib/components/users/UserSearch.svelte";
    import Infos from "$lib/components/users/Infos.svelte";
    import { usersApi } from "$lib/api/users";
    import { ROLE_LABELS, ROLE_GROUPS, isSuperAdmin } from "$lib/types/roles";
    import type { UserCreate, UserResponse, UserUpdate, EmployeeStatus, EmployeeType, UserSearchParams } from "$lib/types/users";
    import { currentUser } from "$lib/stores/auth";
    import DatePicker from "$lib/components/ui/DatePicker.svelte";

    // Types pour les colonnes du DataTable
    type Column = {
        key: string;
        label: string;
        sortable?: boolean;
        width?: string;
    };

    let users: UserResponse[] = [];
    let loading = true;
    let showModal = false;
    let showDeactivateConfirm = false;
    let showActivateConfirm = false;
    let showInfosModal = false;
    let selectedUser: UserResponse | null = null;
    let selectedUserId: number | null = null;
    let isEdit = false;
    let isSuperAdminUser = false;
    let totalUsers = 0;

    // State pour les filtres
    let searchParams: UserSearchParams = {
        limit: 50
    };

    // Formulaire complet avec champs employé
    type FormData = {
        email: string;
        phone: string;
        username: string;
        full_name: string;
        is_active: boolean;
        roles: string[];
        employee_id: string;
        department: string;
        hire_date: string;
        base_salary: string;
        salary_currency: string;
        salary_frequency: string;
        bonus: string;
        employee_status: EmployeeStatus;
        employee_type: EmployeeType;
        bank_name: string;
        bank_account: string;
        rib: string;
        national_id: string;
        social_security_number: string;
        tax_id: string;
        emergency_contact_name: string;
        emergency_contact_phone: string;
        observations: string;
    };

    let formData: FormData = {
        email: "",
        phone: "",
        username: "",
        full_name: "",
        is_active: true,
        roles: [],
        employee_id: "",
        department: "",
        hire_date: "",
        base_salary: "",
        salary_currency: "XOF",
        salary_frequency: "monthly",
        bonus: "",
        employee_status: "actif",
        employee_type: "permanent",
        bank_name: "",
        bank_account: "",
        rib: "",
        national_id: "",
        social_security_number: "",
        tax_id: "",
        emergency_contact_name: "",
        emergency_contact_phone: "",
        observations: ""
    };

    let errors: Record<string, string> = {};

    // Groupes de rôles pour l'affichage
    const roleGroups = [
        { label: "Administration", roles: ROLE_GROUPS.admin },
        { label: "Techniciens", roles: ROLE_GROUPS.technician },
        { label: "Observateurs", roles: ROLE_GROUPS.observer },
        { label: "Rôles transverses", roles: ROLE_GROUPS.transverse },
    ];

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

    const currencyOptions = [
        { value: "XOF", label: "FCFA" },
        { value: "EUR", label: "Euro" },
        { value: "USD", label: "Dollar US" }
    ];

    const salaryFrequencyOptions = [
        { value: "monthly", label: "Mensuel" },
        { value: "hourly", label: "Horaire" },
        { value: "daily", label: "Journalier" },
        { value: "weekly", label: "Hebdomadaire" },
        { value: "annual", label: "Annuel" }
    ];

    function getRoleLabel(role: string): string {
        return ROLE_LABELS[role as keyof typeof ROLE_LABELS] || role;
    }

    const columns: Column[] = [
        { key: "employee_id", label: "Matricule", sortable: true, width: "120px" },
        { key: "full_name", label: "Nom complet", sortable: true },
        { key: "email", label: "Email", sortable: true },
        { key: "department", label: "Département", sortable: true },
        { key: "roles", label: "Rôles", sortable: true },
        { key: "is_active", label: "Statut", sortable: true },
    ];

    const customRenderers: Record<string, (value: any, row: Record<string, any>) => string> = {
        is_active: (value: boolean) => {
            if (value) {
                return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">Actif</span>';
            }
            return '<span class="px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">Inactif</span>';
        },
        roles: (value: string[]) => {
            if (!value || value.length === 0) return "-";
            return value.map(r => getRoleLabel(r)).join(", ");
        },
        employee_id: (value: string) => value || "-",
        department: (value: string) => value || "-"
    };

    onMount(async () => {
        const userRoles = permissionsStore.getRoles();
        isSuperAdminUser = userRoles.some(role => isSuperAdmin(role));
        
        if (!isSuperAdminUser) {
            window.location.href = "/";
            return;
        }
        await loadUsers();
    });

    async function loadUsers(params?: UserSearchParams) {
        loading = true;
        try {
            const response = await usersApi.getUsers({ ...searchParams, ...params });
            users = response.items;
            totalUsers = response.total;
        } catch (error) {
            console.error("Failed to load users:", error);
        } finally {
            loading = false;
        }
    }

    function handleSearch(params: UserSearchParams) {
        searchParams = { ...searchParams, ...params };
        loadUsers(params);
    }

    function handleReset() {
        searchParams = { limit: 50 };
        loadUsers({ limit: 50 });
    }

    function resetForm() {
        formData = {
            email: "",
            phone: "",
            username: "",
            full_name: "",
            is_active: true,
            roles: [],
            employee_id: "",
            department: "",
            hire_date: "",
            base_salary: "",
            salary_currency: "XOF",
            salary_frequency: "monthly",
            bonus: "",
            employee_status: "actif",
            employee_type: "permanent",
            bank_name: "",
            bank_account: "",
            rib: "",
            national_id: "",
            social_security_number: "",
            tax_id: "",
            emergency_contact_name: "",
            emergency_contact_phone: "",
            observations: ""
        };
        errors = {};
    }

    function handleAdd() {
        selectedUser = null;
        isEdit = false;
        resetForm();
        showModal = true;
    }

    function handleEdit(user: UserResponse) {
        selectedUser = user;
        isEdit = true;

        formData = {
            email: user.email,
            phone: user.phone || "",
            username: user.username,
            full_name: user.full_name,
            is_active: user.is_active,
            roles: user.roles ? [...user.roles] : [],
            employee_id: user.employee_id || "",
            department: user.department || "",
            hire_date: user.hire_date || "",
            base_salary: user.base_salary?.toString() || "",
            salary_currency: user.salary_currency || "XOF",
            salary_frequency: user.salary_frequency || "monthly",
            bonus: user.bonus?.toString() || "",
            employee_status: user.employee_status || "actif",
            employee_type: user.employee_type || "permanent",
            bank_name: user.bank_name || "",
            bank_account: user.bank_account || "",
            rib: user.rib || "",
            national_id: user.national_id || "",
            social_security_number: user.social_security_number || "",
            tax_id: user.tax_id || "",
            emergency_contact_name: user.emergency_contact_name || "",
            emergency_contact_phone: user.emergency_contact_phone || "",
            observations: user.observations || ""
        };

        errors = {};
        showModal = true;
    }

    function handleViewUser(userId: number) {
        selectedUserId = userId;
        showInfosModal = true;
    }

    function validate(): boolean {
        errors = {};
        if (!formData.email) errors.email = "L'email est requis";
        if (!formData.username) errors.username = "Le nom d'utilisateur est requis";
        if (!formData.full_name) errors.full_name = "Le nom complet est requis";
        if (!formData.roles || formData.roles.length === 0) {
            errors.roles = "Au moins un rôle est requis";
        }
        return Object.keys(errors).length === 0;
    }

    function toNumber(value: string): number | undefined {
        if (!value || value.trim() === "") return undefined;
        const num = parseFloat(value);
        return isNaN(num) ? undefined : num;
    }

    async function handleSubmit() {
        if (!validate()) return;

        try {
            if (isEdit && selectedUser) {
                const updatePayload: UserUpdate = {
                    email: formData.email,
                    phone: formData.phone || null,
                    full_name: formData.full_name,
                    is_active: formData.is_active,
                    employee_id: formData.employee_id || undefined,
                    department: formData.department || undefined,
                    hire_date: formData.hire_date || undefined,
                    base_salary: toNumber(formData.base_salary),
                    salary_currency: formData.salary_currency || undefined,
                    salary_frequency: formData.salary_frequency || undefined,
                    bonus: toNumber(formData.bonus),
                    employee_status: formData.employee_status || undefined,
                    employee_type: formData.employee_type || undefined,
                    bank_name: formData.bank_name || undefined,
                    bank_account: formData.bank_account || undefined,
                    rib: formData.rib || undefined,
                    national_id: formData.national_id || undefined,
                    social_security_number: formData.social_security_number || undefined,
                    tax_id: formData.tax_id || undefined,
                    emergency_contact_name: formData.emergency_contact_name || undefined,
                    emergency_contact_phone: formData.emergency_contact_phone || undefined,
                    observations: formData.observations || undefined,
                    roles: formData.roles
                };
                await usersApi.updateUser(selectedUser.id, updatePayload);
            } else {
                const createPayload: UserCreate = {
                    email: formData.email,
                    phone: formData.phone || null,
                    username: formData.username,
                    full_name: formData.full_name,
                    is_active: formData.is_active,
                    roles: formData.roles || [],
                    employee_id: formData.employee_id || undefined,
                    department: formData.department || undefined,
                    hire_date: formData.hire_date || undefined,
                    base_salary: toNumber(formData.base_salary),
                    salary_currency: formData.salary_currency || undefined,
                    salary_frequency: formData.salary_frequency || undefined,
                    bonus: toNumber(formData.bonus),
                    employee_status: formData.employee_status || undefined,
                    employee_type: formData.employee_type || undefined,
                    bank_name: formData.bank_name || undefined,
                    bank_account: formData.bank_account || undefined,
                    rib: formData.rib || undefined,
                    national_id: formData.national_id || undefined,
                    social_security_number: formData.social_security_number || undefined,
                    tax_id: formData.tax_id || undefined,
                    emergency_contact_name: formData.emergency_contact_name || undefined,
                    emergency_contact_phone: formData.emergency_contact_phone || undefined,
                    observations: formData.observations || undefined
                };
                await usersApi.createUser(createPayload);
            }
            showModal = false;
            await loadUsers(searchParams);
        } catch (error) {
            console.error("Failed to save user:", error);
            alert("Erreur lors de la sauvegarde");
        }
    }

    async function handleDeactivate() {
        if (selectedUser) {
            try {
                await usersApi.updateUser(selectedUser.id, { is_active: false });
                showDeactivateConfirm = false;
                await loadUsers(searchParams);
            } catch (error) {
                console.error("Failed to deactivate user:", error);
                alert("Erreur lors de la désactivation");
            }
        }
    }

    async function handleActivate() {
        if (selectedUser) {
            try {
                await usersApi.updateUser(selectedUser.id, { is_active: true });
                showActivateConfirm = false;
                await loadUsers(searchParams);
            } catch (error) {
                console.error("Failed to activate user:", error);
                alert("Erreur lors de la réactivation");
            }
        }
    }

    function canToggleStatus(user: UserResponse): boolean {
        return $currentUser?.id !== user.id;        
    }

    function toggleRole(roleValue: string) {
        if (!formData.roles) {
            formData.roles = [];
        }
        if (formData.roles.includes(roleValue)) {
            formData.roles = formData.roles.filter(r => r !== roleValue);
        } else {
            formData.roles = [...formData.roles, roleValue];
        }
    }

    const asUser = (row: Record<string, any>) => row as UserResponse;

    function goBack() {
        history.back();
    }
</script>

<div class="space-y-6">
    <!-- En-tête -->
    <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
            <button
                on:click={goBack}
                class="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 hover:border-gray-400 transition-all duration-200 shadow-sm"
            >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
                <span>Retour</span>
            </button>

            <div>
                <h1 class="text-2xl font-bold text-gray-900">Employés</h1>
                <p class="text-sm text-gray-500 mt-1">
                    {totalUsers} employé{totalUsers > 1 ? 's' : ''} trouvé{totalUsers > 1 ? 's' : ''}
                </p>
            </div>
        </div>

        <Button on:click={handleAdd} variant="primary">
            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
            </svg>
            Nouvel employé
        </Button>
    </div>

    <!-- Composant de recherche -->
    <UserSearch
        {loading}
        on:search={(e) => handleSearch(e.detail)}
        on:reset={handleReset}
    />

    <!-- Tableau des utilisateurs -->
    <DataTable
        {columns}
        data={users}
        {loading}
        selectable={false}
        {customRenderers}
    >
        <div slot="actions-row" let:row>
            {@const user = asUser(row)}
            <div class="flex items-center gap-2 justify-end">
                <!-- Bouton Voir les détails -->
                <button
                    on:click={() => handleViewUser(user.id)}
                    class="p-1 text-gray-400 hover:text-blue-600 transition-colors"
                    title="Voir les détails"
                >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                </button>

                <!-- Bouton Modifier -->
                <button
                    on:click={() => handleEdit(user)}
                    class="p-1 text-gray-400 hover:text-blue-600 transition-colors"
                    title="Modifier"
                >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                </button>
                
                <!-- Bouton Activer/Désactiver -->
                {#if user.is_active}
                    <button
                        on:click={() => {
                            selectedUser = user;
                            showDeactivateConfirm = true;
                        }}
                        disabled={!canToggleStatus(user)}
                        class="p-1 text-gray-400 hover:text-orange-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Désactiver"
                    >
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                        </svg>
                    </button>
                {:else}
                    <button
                        on:click={() => {
                            selectedUser = user;
                            showActivateConfirm = true;
                        }}
                        disabled={!canToggleStatus(user)}
                        class="p-1 text-gray-400 hover:text-green-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Réactiver"
                    >
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                    </button>
                {/if}
            </div>
        </div>
    </DataTable>

    <!-- Modal d'ajout/modification -->
    <Modal
        open={showModal}
        title={isEdit ? "Modifier l'employé" : "Ajouter un employé"}
        on:close={() => (showModal = false)}
        size="xl"
    >
        <div class="space-y-4 max-h-[70vh] overflow-y-auto pr-2">
            <!-- Section 1: Informations de base -->
            <div class="border-b border-gray-200 pb-4">
                <h3 class="text-lg font-medium text-gray-900 mb-3">Informations de base</h3>
                <div class="grid grid-cols-2 gap-3">
                    <Input
                        label="Email"
                        bind:value={formData.email}
                        inputType="email"
                        required
                        error={errors.email}
                    />
                    <Input
                        label="Téléphone"
                        bind:value={formData.phone}
                        inputType="tel"
                        placeholder="+221 77 123 45 67"
                    />
                    <Input
                        label="Nom d'utilisateur"
                        bind:value={formData.username}
                        required
                        error={errors.username}
                    />
                    <Input
                        label="Nom complet"
                        bind:value={formData.full_name}
                        required
                        error={errors.full_name}
                    />
                </div>
            </div>

            <!-- Section 2: Informations professionnelles -->
            <div class="border-b border-gray-200 pb-4">
                <h3 class="text-lg font-medium text-gray-900 mb-3">Informations professionnelles</h3>
                <div class="grid grid-cols-2 gap-3">
                    <Input
                        label="Matricule"
                        bind:value={formData.employee_id}
                        placeholder="EMP-001"
                    />
                    <Input
                        label="Département"
                        bind:value={formData.department}
                        placeholder="Élevage"
                    />
                    <DatePicker
                        label="Date d'embauche"
                        bind:value={formData.hire_date}
                    />
                </div>
            </div>

            <!-- Section 3: Salaire et rémunération -->
            <div class="border-b border-gray-200 pb-4">
                <h3 class="text-lg font-medium text-gray-900 mb-3">Salaire et rémunération</h3>
                <div class="grid grid-cols-3 gap-3">
                    <Input
                        label="Salaire de base"
                        bind:value={formData.base_salary}
                        inputType="number"
                        step="1000"
                        placeholder="0"
                    />
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Devise</label>
                        <select
                            bind:value={formData.salary_currency}
                            class="w-full rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        >
                            {#each currencyOptions as opt}
                                <option value={opt.value}>{opt.label}</option>
                            {/each}
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Fréquence</label>
                        <select
                            bind:value={formData.salary_frequency}
                            class="w-full rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        >
                            {#each salaryFrequencyOptions as opt}
                                <option value={opt.value}>{opt.label}</option>
                            {/each}
                        </select>
                    </div>
                    <Input
                        label="Prime/Bonus"
                        bind:value={formData.bonus}
                        inputType="number"
                        step="1000"
                        placeholder="0"
                    />
                </div>
            </div>

            <!-- Section 4: Statut -->
            <div class="border-b border-gray-200 pb-4">
                <h3 class="text-lg font-medium text-gray-900 mb-3">Statut</h3>
                <div class="grid grid-cols-2 gap-3">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Statut employé</label>
                        <select
                            bind:value={formData.employee_status}
                            class="w-full rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        >
                            {#each employeeStatusOptions as opt}
                                <option value={opt.value}>{opt.label}</option>
                            {/each}
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Type d'employé</label>
                        <select
                            bind:value={formData.employee_type}
                            class="w-full rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        >
                            {#each employeeTypeOptions as opt}
                                <option value={opt.value}>{opt.label}</option>
                            {/each}
                        </select>
                    </div>
                    <div class="col-span-2">
                        <label class="flex items-center space-x-3 cursor-pointer">
                            <input
                                type="checkbox"
                                bind:checked={formData.is_active}
                                class="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                            />
                            <span class="text-sm text-gray-700">Compte actif</span>
                        </label>
                    </div>
                </div>
            </div>

            <!-- Section 5: Informations bancaires -->
            <div class="border-b border-gray-200 pb-4">
                <h3 class="text-lg font-medium text-gray-900 mb-3">Informations bancaires</h3>
                <div class="grid grid-cols-3 gap-3">
                    <Input label="Banque" bind:value={formData.bank_name} />
                    <Input label="Numéro de compte" bind:value={formData.bank_account} />
                    <Input label="RIB" bind:value={formData.rib} />
                </div>
            </div>

            <!-- Section 6: Informations administratives -->
            <div class="border-b border-gray-200 pb-4">
                <h3 class="text-lg font-medium text-gray-900 mb-3">Informations administratives</h3>
                <div class="grid grid-cols-3 gap-3">
                    <Input label="CNI/Passeport" bind:value={formData.national_id} />
                    <Input label="Sécurité sociale" bind:value={formData.social_security_number} />
                    <Input label="Numéro fiscal" bind:value={formData.tax_id} />
                </div>
            </div>

            <!-- Section 7: Contact d'urgence -->
            <div class="border-b border-gray-200 pb-4">
                <h3 class="text-lg font-medium text-gray-900 mb-3">Contact d'urgence</h3>
                <div class="grid grid-cols-2 gap-3">
                    <Input label="Nom" bind:value={formData.emergency_contact_name} />
                    <Input label="Téléphone" bind:value={formData.emergency_contact_phone} inputType="tel" />
                </div>
            </div>

            <!-- Section 8: Observations -->
            <div class="border-b border-gray-200 pb-4">
                <Input
                    label="Observations"
                    bind:value={formData.observations}
                    textarea={true}
                    rows={2}
                />
            </div>

            <!-- Section 9: Rôles -->
            <div class="border-b border-gray-200 pb-4">
                <label class="block text-sm font-medium text-gray-700 mb-1">
                    Rôles
                    <span class="text-red-500 ml-1">*</span>
                </label>
                
                <div class="space-y-3 max-h-60 overflow-y-auto p-2 border border-gray-300 rounded-lg">
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
                                                checked={(formData.roles || []).includes(roleValue)}
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
                
                {#if errors.roles}
                    <p class="mt-1 text-sm text-red-600">{errors.roles}</p>
                {/if}
                <p class="mt-1 text-xs text-gray-500">
                    Sélectionnez un ou plusieurs rôles pour définir les permissions de l'utilisateur
                </p>
            </div>

            <!-- Message informatif sur le mot de passe -->
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-700">
                <span class="font-medium">🔑 Information :</span>
                {#if isEdit}
                    Le mot de passe de l'utilisateur reste inchangé lors de la modification.
                {:else}
                    Un mot de passe temporaire sera généré et envoyé par email à l'utilisateur.
                {/if}
            </div>

            <!-- Boutons d'action -->
            <div class="flex justify-end gap-3 pt-4 border-t">
                <Button on:click={() => (showModal = false)} variant="outline">
                    Annuler
                </Button>
                <Button on:click={handleSubmit} variant="primary">
                    {isEdit ? "Mettre à jour" : "Créer"}
                </Button>
            </div>
        </div>
    </Modal>

    <!-- Modal des détails de l'utilisateur (Infos) -->
    <Modal
        open={showInfosModal}
        on:close={() => {
            showInfosModal = false;
            selectedUserId = null;
        }}
        size="lg"
        title="Détails de l'employé"
    >
        {#if showInfosModal}
            <Infos
                userId={selectedUserId}
                on:close={() => {
                    showInfosModal = false;
                    selectedUserId = null;
                }}
                on:edit={(e) => {
                    // Trouver l'utilisateur et ouvrir le formulaire d'édition
                    const user = users.find(u => u.id === e.detail);
                    if (user) {
                        handleEdit(user);
                        showInfosModal = false;
                    }
                }}
                on:deactivate={(e) => {
                    // Trouver l'utilisateur et ouvrir la confirmation de désactivation
                    const user = users.find(u => u.id === e.detail);
                    if (user) {
                        selectedUser = user;
                        showDeactivateConfirm = true;
                        showInfosModal = false;
                    }
                }}
                on:activate={(e) => {
                    // Trouver l'utilisateur et ouvrir la confirmation de réactivation
                    const user = users.find(u => u.id === e.detail);
                    if (user) {
                        selectedUser = user;
                        showActivateConfirm = true;
                        showInfosModal = false;
                    }
                }}
            />
        {/if}
    </Modal>

    <!-- Confirmations -->
    <ConfirmDialog
        open={showDeactivateConfirm}
        title="Désactiver l'employé"
        message={`Êtes-vous sûr de vouloir désactiver ${selectedUser?.full_name} ? L'utilisateur ne pourra plus se connecter jusqu'à sa réactivation.`}
        confirmVariant="danger"
        confirmText="Désactiver"
        on:confirm={handleDeactivate}
        on:cancel={() => (showDeactivateConfirm = false)}
    />

    <ConfirmDialog
        open={showActivateConfirm}
        title="Réactiver l'employé"
        message={`Êtes-vous sûr de vouloir réactiver ${selectedUser?.full_name} ? L'utilisateur pourra à nouveau se connecter.`}
        confirmVariant="primary"
        confirmText="Réactiver"
        on:confirm={handleActivate}
        on:cancel={() => (showActivateConfirm = false)}
    />
</div>