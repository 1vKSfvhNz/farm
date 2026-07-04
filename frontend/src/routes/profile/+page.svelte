<!-- src/routes/profile/+page.svelte -->
<!-- svelte-ignore a11y-label-has-associated-control -->
<script lang="ts">
    import { onMount } from "svelte";
    import { authStore, currentUser } from "$lib/stores/auth";
    import { authApi } from "$lib/api/auth";
    import { usersApi } from "$lib/api/users";
    import Input from "$lib/components/ui/Input.svelte";
    import Button from "$lib/components/ui/Button.svelte";
    import Card from "$lib/components/ui/Card.svelte";
    import type { EmployeeStatus, EmployeeType, UserResponse, UserUpdate } from "$lib/types/users";
    import { ROLE_LABELS } from "$lib/types/roles";
    import DatePicker from "$lib/components/ui/DatePicker.svelte";

    let user: UserResponse | null = null;
    let loading = false;
    let saving = false;
    let error = "";
    let successMessage = "";
    
    // Formulaire de mise à jour du profil - TOUTES LES VALEURS SONT DES STRINGS
    let formData = {
        full_name: "",
        email: "",
        phone: "",
        username: "",
        employee_id: "",
        department: "",
        hire_date: "",
        base_salary: "",
        salary_currency: "XOF",
        salary_frequency: "monthly",
        bonus: "",
        employee_status: "actif" as EmployeeStatus,
        employee_type: "permanent" as EmployeeType,
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

    // Formulaire de changement de mot de passe
    let oldPassword = "";
    let newPassword = "";
    let confirmPassword = "";
    let passwordError = "";
    let passwordSuccess = "";

    // États pour le mode édition
    let isEditing = false;

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

    // Fonction pour obtenir le label d'un rôle
    function getRoleLabel(role: string): string {
        return ROLE_LABELS[role as keyof typeof ROLE_LABELS] || role;
    }

    // Helper pour convertir les valeurs en string (jamais null ou undefined)
    function toStr(value: string | null | undefined): string {
        return value ?? "";
    }

    // Helper pour convertir un nombre en string
    function numToStr(value: number | null | undefined): string {
        if (value === null || value === undefined) return "";
        return value.toString();
    }

    // Helper pour convertir une string en number ou undefined
    function toNum(value: string): number | undefined {
        if (!value || value.trim() === "") return undefined;
        const num = parseFloat(value);
        return isNaN(num) ? undefined : num;
    }

    // S'abonner au store pour obtenir l'utilisateur
    const unsubscribe = currentUser.subscribe((value) => {
        user = value;
        if (user) {
            initForm();
        }
    });

    // Initialiser le formulaire avec les données utilisateur
    function initForm() {
        if (user) {
            formData = {
                full_name: toStr(user.full_name),
                email: toStr(user.email),
                phone: toStr(user.phone),
                username: toStr(user.username),
                employee_id: toStr(user.employee_id),
                department: toStr(user.department),
                hire_date: toStr(user.hire_date),
                base_salary: numToStr(user.base_salary),
                salary_currency: toStr(user.salary_currency) || "XOF",
                salary_frequency: toStr(user.salary_frequency) || "monthly",
                bonus: numToStr(user.bonus),
                employee_status: user.employee_status || "actif",
                employee_type: user.employee_type || "permanent",
                bank_name: toStr(user.bank_name),
                bank_account: toStr(user.bank_account),
                rib: toStr(user.rib),
                national_id: toStr(user.national_id),
                social_security_number: toStr(user.social_security_number),
                tax_id: toStr(user.tax_id),
                emergency_contact_name: toStr(user.emergency_contact_name),
                emergency_contact_phone: toStr(user.emergency_contact_phone),
                observations: toStr(user.observations)
            };
        }
    }

    onMount(() => {
        return unsubscribe;
    });

    // Mettre à jour le profil
    async function handleUpdateProfile() {
        if (!user) return;

        saving = true;
        error = "";
        successMessage = "";

        try {
            const updateData: UserUpdate = {
                full_name: formData.full_name || undefined,
                email: formData.email || undefined,
                phone: formData.phone || null,
                username: formData.username || undefined,
                employee_id: formData.employee_id || undefined,
                department: formData.department || undefined,
                hire_date: formData.hire_date || undefined,
                base_salary: toNum(formData.base_salary),
                salary_currency: formData.salary_currency || undefined,
                salary_frequency: formData.salary_frequency || undefined,
                bonus: toNum(formData.bonus),
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

            const updatedUser = await usersApi.updateUser(user.id, updateData);
            
            // Mettre à jour le store
            await authStore.refreshUser();
            
            successMessage = "Profil mis à jour avec succès !";
            isEditing = false;
            initForm();
        } catch (err: any) {
            error = err.message || "Erreur lors de la mise à jour du profil";
            console.error("Update profile error:", err);
        } finally {
            saving = false;
        }
    }

    // Changer le mot de passe
    async function handleChangePassword() {
        if (newPassword !== confirmPassword) {
            passwordError = "Les mots de passe ne correspondent pas";
            return;
        }

        if (newPassword.length < 6) {
            passwordError = "Le mot de passe doit contenir au moins 6 caractères";
            return;
        }

        if (newPassword === oldPassword) {
            passwordError = "Le nouveau mot de passe doit être différent de l'ancien";
            return;
        }

        loading = true;
        passwordError = "";
        passwordSuccess = "";

        try {
            await authApi.changePassword({
                old_password: oldPassword,
                new_password: newPassword
            });
            
            passwordSuccess = "Mot de passe modifié avec succès !";
            oldPassword = "";
            newPassword = "";
            confirmPassword = "";
        } catch (err: any) {
            passwordError = err.message || "Erreur lors de la modification du mot de passe";
        } finally {
            loading = false;
        }
    }

    // Annuler l'édition
    function cancelEdit() {
        isEditing = false;
        initForm();
        error = "";
        successMessage = "";
    }

    // Auto-fermeture des messages
    $: {
        if (successMessage) {
            setTimeout(() => { successMessage = ""; }, 5000);
        }
        if (passwordSuccess) {
            setTimeout(() => { passwordSuccess = ""; }, 5000);
        }
    }
</script>

<div class="max-w-4xl mx-auto space-y-6 p-4">
    <div class="flex items-center justify-between">
        <h1 class="text-2xl font-bold text-gray-900">Mon profil</h1>
        <div class="flex items-center gap-3">
            <span class="px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                {user?.is_active ? "Actif" : "Inactif"}
            </span>
            <Button
                variant={isEditing ? "outline" : "primary"}
                size="sm"
                on:click={() => {
                    if (isEditing) {
                        cancelEdit();
                    } else {
                        isEditing = true;
                        initForm();
                    }
                }}
            >
                {isEditing ? "Annuler" : "Modifier"}
            </Button>
        </div>
    </div>

    <!-- Messages -->
    {#if successMessage}
        <div class="bg-green-50 border border-green-200 rounded-lg p-3">
            <p class="text-green-700 text-sm">✅ {successMessage}</p>
        </div>
    {/if}
    {#if error}
        <div class="bg-red-50 border border-red-200 rounded-lg p-3">
            <p class="text-red-700 text-sm">❌ {error}</p>
        </div>
    {/if}

    <!-- Carte Informations personnelles -->
    <Card title="Informations personnelles">
        <div class="space-y-6">
            <!-- Section Informations de base -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        Nom complet
                    </label>
                    {#if isEditing}
                        <Input bind:value={formData.full_name} placeholder="Nom complet" required />
                    {:else}
                        <p class="text-gray-900 py-2">{toStr(user?.full_name) || "-"}</p>
                    {/if}
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        Email
                    </label>
                    {#if isEditing}
                        <Input bind:value={formData.email} inputType="email" placeholder="Email" required />
                    {:else}
                        <p class="text-gray-900 py-2">{toStr(user?.email) || "-"}</p>
                    {/if}
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        Nom d'utilisateur
                    </label>
                    {#if isEditing}
                        <Input bind:value={formData.username} placeholder="Nom d'utilisateur" required />
                    {:else}
                        <p class="text-gray-900 py-2">{toStr(user?.username) || "-"}</p>
                    {/if}
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        Téléphone
                    </label>
                    {#if isEditing}
                        <Input bind:value={formData.phone} inputType="tel" placeholder="Téléphone" />
                    {:else}
                        <p class="text-gray-900 py-2">{toStr(user?.phone) || "-"}</p>
                    {/if}
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        Rôles
                    </label>
                    <p class="text-gray-900 py-2">
                        <span class="inline-flex flex-wrap gap-1">
                            {#if user?.roles && user.roles.length > 0}
                                {#each user.roles as role}
                                    <span class="px-2 py-0.5 rounded-full text-xs bg-gray-100 text-gray-700">
                                        {getRoleLabel(role)}
                                    </span>
                                {/each}
                            {:else}
                                -
                            {/if}
                        </span>
                    </p>
                </div>
            </div>

            <!-- Section Informations professionnelles -->
            <div class="border-t border-gray-200 pt-4">
                <h3 class="text-md font-semibold text-gray-800 mb-3">Informations professionnelles</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Matricule
                        </label>
                        {#if isEditing}
                            <Input bind:value={formData.employee_id} placeholder="Matricule" />
                        {:else}
                            <p class="text-gray-900 py-2">{toStr(user?.employee_id) || "-"}</p>
                        {/if}
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Département
                        </label>
                        {#if isEditing}
                            <Input bind:value={formData.department} placeholder="Département" />
                        {:else}
                            <p class="text-gray-900 py-2">{toStr(user?.department) || "-"}</p>
                        {/if}
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Date d'embauche
                        </label>
                        {#if isEditing}
                            <DatePicker bind:value={formData.hire_date}/>
                        {:else}
                            <p class="text-gray-900 py-2">
                                {user?.hire_date ? new Date(user.hire_date).toLocaleDateString('fr-FR') : "-"}
                            </p>
                        {/if}
                    </div>
                </div>
            </div>

            <!-- Section Salaire -->
            <div class="border-t border-gray-200 pt-4">
                <h3 class="text-md font-semibold text-gray-800 mb-3">Salaire et rémunération</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Salaire de base
                        </label>
                        {#if isEditing}
                            <Input bind:value={formData.base_salary} inputType="number" step="1000" placeholder="0" />
                        {:else}
                            <p class="text-gray-900 py-2">
                                {user?.base_salary ? `${user.base_salary.toLocaleString()} FCFA` : "-"}
                            </p>
                        {/if}
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Devise
                        </label>
                        {#if isEditing}
                            <select
                                bind:value={formData.salary_currency}
                                class="w-full rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
                            >
                                {#each currencyOptions as opt}
                                    <option value={opt.value}>{opt.label}</option>
                                {/each}
                            </select>
                        {:else}
                            <p class="text-gray-900 py-2">{toStr(user?.salary_currency) || "XOF"}</p>
                        {/if}
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Prime/Bonus
                        </label>
                        {#if isEditing}
                            <Input bind:value={formData.bonus} inputType="number" step="1000" placeholder="0" />
                        {:else}
                            <p class="text-gray-900 py-2">
                                {user?.bonus ? `${user.bonus.toLocaleString()} FCFA` : "0 FCFA"}
                            </p>
                        {/if}
                    </div>
                </div>
            </div>

            <!-- Section Statut -->
            <div class="border-t border-gray-200 pt-4">
                <h3 class="text-md font-semibold text-gray-800 mb-3">Statut</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Statut employé
                        </label>
                        {#if isEditing}
                            <select
                                bind:value={formData.employee_status}
                                class="w-full rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
                            >
                                {#each employeeStatusOptions as opt}
                                    <option value={opt.value}>{opt.label}</option>
                                {/each}
                            </select>
                        {:else}
                            <p class="text-gray-900 py-2">
                                {employeeStatusOptions.find(opt => opt.value === user?.employee_status)?.label || "-"}
                            </p>
                        {/if}
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Type d'employé
                        </label>
                        {#if isEditing}
                            <select
                                bind:value={formData.employee_type}
                                class="w-full rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
                            >
                                {#each employeeTypeOptions as opt}
                                    <option value={opt.value}>{opt.label}</option>
                                {/each}
                            </select>
                        {:else}
                            <p class="text-gray-900 py-2">
                                {employeeTypeOptions.find(opt => opt.value === user?.employee_type)?.label || "-"}
                            </p>
                        {/if}
                    </div>
                </div>
            </div>

            <!-- Section Informations bancaires -->
            <div class="border-t border-gray-200 pt-4">
                <h3 class="text-md font-semibold text-gray-800 mb-3">Informations bancaires</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Banque
                        </label>
                        {#if isEditing}
                            <Input bind:value={formData.bank_name} placeholder="Nom de la banque" />
                        {:else}
                            <p class="text-gray-900 py-2">{toStr(user?.bank_name) || "-"}</p>
                        {/if}
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Compte
                        </label>
                        {#if isEditing}
                            <Input bind:value={formData.bank_account} placeholder="Numéro de compte" />
                        {:else}
                            <p class="text-gray-900 py-2">{toStr(user?.bank_account) || "-"}</p>
                        {/if}
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            RIB
                        </label>
                        {#if isEditing}
                            <Input bind:value={formData.rib} placeholder="RIB" />
                        {:else}
                            <p class="text-gray-900 py-2">{toStr(user?.rib) || "-"}</p>
                        {/if}
                    </div>
                </div>
            </div>

            <!-- Section Informations administratives -->
            <div class="border-t border-gray-200 pt-4">
                <h3 class="text-md font-semibold text-gray-800 mb-3">Informations administratives</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            CNI/Passeport
                        </label>
                        {#if isEditing}
                            <Input bind:value={formData.national_id} placeholder="CNI/Passeport" />
                        {:else}
                            <p class="text-gray-900 py-2">{toStr(user?.national_id) || "-"}</p>
                        {/if}
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Sécurité sociale
                        </label>
                        {#if isEditing}
                            <Input bind:value={formData.social_security_number} placeholder="Numéro de sécurité sociale" />
                        {:else}
                            <p class="text-gray-900 py-2">{toStr(user?.social_security_number) || "-"}</p>
                        {/if}
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Numéro fiscal
                        </label>
                        {#if isEditing}
                            <Input bind:value={formData.tax_id} placeholder="Numéro fiscal" />
                        {:else}
                            <p class="text-gray-900 py-2">{toStr(user?.tax_id) || "-"}</p>
                        {/if}
                    </div>
                </div>
            </div>

            <!-- Section Contact d'urgence -->
            <div class="border-t border-gray-200 pt-4">
                <h3 class="text-md font-semibold text-gray-800 mb-3">Contact d'urgence</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Nom
                        </label>
                        {#if isEditing}
                            <Input bind:value={formData.emergency_contact_name} placeholder="Nom du contact" />
                        {:else}
                            <p class="text-gray-900 py-2">{toStr(user?.emergency_contact_name) || "-"}</p>
                        {/if}
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">
                            Téléphone
                        </label>
                        {#if isEditing}
                            <Input bind:value={formData.emergency_contact_phone} inputType="tel" placeholder="Téléphone d'urgence" />
                        {:else}
                            <p class="text-gray-900 py-2">{toStr(user?.emergency_contact_phone) || "-"}</p>
                        {/if}
                    </div>
                </div>
            </div>

            <!-- Section Observations -->
            <div class="border-t border-gray-200 pt-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        Observations
                    </label>
                    {#if isEditing}
                        <Input bind:value={formData.observations} textarea={true} rows={3} placeholder="Observations..." />
                    {:else}
                        <p class="text-gray-900 py-2 whitespace-pre-wrap">{toStr(user?.observations) || "-"}</p>
                    {/if}
                </div>
            </div>

            <!-- Boutons d'action -->
            {#if isEditing}
                <div class="flex justify-end gap-3 pt-4 border-t">
                    <Button on:click={cancelEdit} variant="outline">
                        Annuler
                    </Button>
                    <Button
                        on:click={handleUpdateProfile}
                        loading={saving}
                        variant="primary"
                    >
                        {saving ? "Enregistrement..." : "Enregistrer les modifications"}
                    </Button>
                </div>
            {/if}
        </div>
    </Card>

    <!-- Carte Changement de mot de passe -->
    <Card title="Changer le mot de passe">
        <div class="space-y-4">
            {#if passwordSuccess}
                <div class="bg-green-50 border border-green-200 rounded-lg p-3">
                    <p class="text-green-700 text-sm">✅ {passwordSuccess}</p>
                </div>
            {/if}
            {#if passwordError}
                <div class="bg-red-50 border border-red-200 rounded-lg p-3">
                    <p class="text-red-700 text-sm">❌ {passwordError}</p>
                </div>
            {/if}

            <Input
                label="Mot de passe actuel"
                bind:value={oldPassword}
                inputType="password"
                placeholder="Mot de passe actuel"
                required
            />

            <Input
                label="Nouveau mot de passe"
                bind:value={newPassword}
                inputType="password"
                placeholder="Nouveau mot de passe (minimum 6 caractères)"
                required
                hint="Le mot de passe doit contenir au moins 6 caractères"
            />

            <Input
                label="Confirmer le mot de passe"
                bind:value={confirmPassword}
                inputType="password"
                placeholder="Confirmer le nouveau mot de passe"
                required
            />

            <div class="flex justify-end">
                <Button
                    on:click={handleChangePassword}
                    loading={loading}
                    variant="primary"
                    disabled={!oldPassword || !newPassword || !confirmPassword}
                >
                    {loading ? "Changement..." : "Changer le mot de passe"}
                </Button>
            </div>
        </div>
    </Card>
</div>