<!-- src/routes/+page.svelte -->
<script lang="ts">
    import { goto } from "$app/navigation";
    import { authStore, isSuperAdmin, userRoles } from "$stores/auth";
    import { apiClient } from "$lib/api/client";
    import type {
        AccountSummary,
        AvicoleStats,
        BovinStats,
        CaprinStats,
        EggProductionStats,
        EntomocultureStats,
        LengthResponse,
        OvinStats,
        PiscicoleStats,
        RecentActivitiesResponse,
        RucheStats,
    } from "$lib";

    // ✅ État pour la sidebar mobile
    let isSidebarOpen = false;

    // ✅ Fonction pour toggle la sidebar
    function toggleSidebar() {
        isSidebarOpen = !isSidebarOpen;
    }

    // ✅ Fermer la sidebar quand on clique sur un lien (navigation)
    function closeSidebar() {
        isSidebarOpen = false;
    }

    // ✅ Initialisation des stats avec des valeurs par défaut
    let stats: LengthResponse = {
        users_length: 0,
        enclos_length: 0,
        bovins_length: 0,
        ovins_length: 0,
        caprins_length: 0,
        avicoles_length: 0,
        piscicoles_length: 0,
        ruches_length: 0,
        nids_length: 0
    };

    let productionStats = {
        oeufs_jour: 0,
        lait_jour: 0,
        miel_kg: 0,
    };

    let financialStats = {
        ca_mois: 0,
        depenses_mois: 0,
        benefice_mois: 0,
    };

    interface Activity {
        icon?: string;
        description: string;
        entity_name?: string;
        created_at: string;
    }
    let recentActivities: Activity[] = [];

    interface Alert {
        niveau: string;
        date_alerte: string;
        message: string;
    }
    let alerts: Alert[] = [];

    let isLoading = true;

    // ✅ Utiliser les stores dérivés directement (réactif)
    $: canViewBovins = $userRoles.includes("super_admin") ||
        $userRoles.includes("bovin_admin") ||
        $userRoles.includes("bovin_technicien") ||
        $userRoles.includes("bovin_observateur");

    $: canViewOvins = $userRoles.includes("super_admin") ||
        $userRoles.includes("ovin_admin") ||
        $userRoles.includes("ovin_technicien") ||
        $userRoles.includes("ovin_observateur");

    $: canViewCaprins = $userRoles.includes("super_admin") ||
        $userRoles.includes("caprin_admin") ||
        $userRoles.includes("caprin_technicien") ||
        $userRoles.includes("caprin_observateur");

    // $: canViewAvicoles = $userRoles.includes("super_admin") ||
    //     $userRoles.includes("avicole_admin") ||
    //     $userRoles.includes("avicole_technicien") ||
    //     $userRoles.includes("avicole_observateur");

    // $: canViewPiscicoles = $userRoles.includes("super_admin") ||
    //     $userRoles.includes("piscicole_admin") ||
    //     $userRoles.includes("piscicole_technicien") ||
    //     $userRoles.includes("piscicole_observateur");

    // $: canViewApiculture = $userRoles.includes("super_admin") ||
    //     $userRoles.includes("apiculture_admin") ||
    //     $userRoles.includes("apiculture_technicien") ||
    //     $userRoles.includes("apiculture_observateur");

    // $: canViewEntomoculture = $userRoles.includes("super_admin") ||
    //     $userRoles.includes("entomoculture_admin") ||
    //     $userRoles.includes("entomoculture_technicien") ||
    //     $userRoles.includes("entomoculture_observateur");
        
    $: canViewAccounting = $userRoles.includes("super_admin") || 
        $userRoles.includes("responsable_account");
        
    $: canViewEnclos = $userRoles.includes("super_admin") ||
        $userRoles.includes("responsable_enclos");


    // ✅ Fonction pour attendre le token
    async function waitForToken(maxAttempts = 20): Promise<boolean> {
        for (let i = 0; i < maxAttempts; i++) {
            const token = localStorage.getItem('access_token');
            if (token) {
                console.log(`✅ Token trouvé après ${i} tentatives`);
                return true;
            }
            await new Promise(resolve => setTimeout(resolve, 50));
        }
        console.error('❌ Token non trouvé après attente');
        return false;
    }

    // ✅ Charger les données quand l'auth est prête ET que les rôles sont chargés
    $: if (!$authStore.isLoading && $authStore.isAuthenticated && $userRoles.length > 0) {
        loadDashboardData();
    }

    // ✅ Redirection si non authentifié
    $: if (!$authStore.isLoading && !$authStore.isAuthenticated) {
        goto("/auth/login");
    }

    // ✅ Fonction pour naviguer vers le profil
    function goToProfile() {
        closeSidebar();
        goto("/profile");
    }

    async function loadDashboardData() {
        // ✅ Attendre que le token soit disponible
        const hasToken = await waitForToken();
        if (!hasToken) {
            console.error('Token non disponible, impossible de charger les données');
            isLoading = false;
            return;
        }
        
        isLoading = true;
        console.log('Chargement des données...');
        console.log('Permissions:', {
            bovins: canViewBovins,
            enclos: canViewEnclos,
            roles: $userRoles
        });
        
        try {
            // ✅ CHARGEMENT DES LENGTHS - UNIQUE APPEL
            try {
                const lengthData = await apiClient.get<LengthResponse>('users/length');
                console.log('📊 Length data reçue:', lengthData);
                
                // Mettre à jour les stats avec les données reçues
                stats = {
                    users_length: lengthData.users_length || 0,
                    enclos_length: lengthData.enclos_length || 0,
                    bovins_length: lengthData.bovins_length || 0,
                    ovins_length: lengthData.ovins_length || 0,
                    caprins_length: lengthData.caprins_length || 0,
                    avicoles_length: lengthData.avicoles_length || 0,
                    piscicoles_length: lengthData.piscicoles_length || 0,
                    ruches_length: lengthData.ruches_length || 0,
                    nids_length: lengthData.nids_length || 0
                };
            } catch (error: any) {
                console.error('❌ Erreur chargement lengths:', error?.message);
                // Garder les valeurs par défaut
            }

            // ✅ Charger les statistiques détaillées par espèce (si nécessaire)
            // Note: Les lengths sont déjà chargés, mais vous pouvez charger des stats supplémentaires
            if (canViewBovins) {
                try {
                    const bovins = await apiClient.get<BovinStats>("/bovins/stats/global");
                    // Utiliser les données supplémentaires si besoin
                } catch (e) { console.error('Bovins error:', e); }
            }
            
            if (canViewOvins) {
                try {
                    const ovins = await apiClient.get<OvinStats>("/ovins/stats/global");
                } catch (e) { console.error('Ovins error:', e); }
            }
            
            if (canViewCaprins) {
                try {
                    const caprins = await apiClient.get<CaprinStats>("/caprins/stats/global");
                } catch (e) { console.error('Caprins error:', e); }
            }
            
            // if (canViewAvicoles) {
            //     try {
            //         const avicoles = await apiClient.get<AvicoleStats>("/avicoles/stats/global");
            //     } catch (e) { console.error('Avicoles error:', e); }
            // }
            
            // if (canViewPiscicoles) {
            //     try {
            //         const piscicoles = await apiClient.get<PiscicoleStats>("/piscicoles/stats/global");
            //     } catch (e) { console.error('Piscicoles error:', e); }
            // }
            
            // if (canViewApiculture) {
            //     try {
            //         const apiary = await apiClient.get<RucheStats>("/apiary/stats/ruches");
            //     } catch (e) { console.error('Apiculture error:', e); }
            // }
            
            // if (canViewEntomoculture) {
            //     try {
            //         const entomoculture = await apiClient.get<EntomocultureStats>("/entomoculture/stats/global");
            //     } catch (e) { console.error('Entomoculture error:', e); }
            // }

            // // ✅ Charger la production
            // if (canViewAvicoles) {
            //     try {
            //         const oeufs = await apiClient.get<EggProductionStats>("/avicoles/production/oeufs/stats", { params: { days: 30 } });
            //         productionStats.oeufs_jour = oeufs.moyenne_par_jour || 0;
            //     } catch (e) { console.error('Production error:', e); }
            // }

            // ✅ Charger les finances
            if (canViewAccounting) {
                try {
                    const finances = await apiClient.get<AccountSummary>("/accounting/summary");
                    financialStats.ca_mois = finances.total_recettes || 0;
                    financialStats.depenses_mois = finances.total_depenses || 0;
                    financialStats.benefice_mois = finances.benefice || 0;
                } catch (e) { console.error('Accounting error:', e); }
            }

            // ✅ Charger les activités récentes
            try {
                const activities = await apiClient.get<RecentActivitiesResponse>("/dashboard/recent-activities", { params: { limit: 5 } });
                const recentActivities = activities || [];
            } catch (e) { console.error('Activities error:', e); }

            // ✅ Charger les alertes
            try {
                const alertsData = await apiClient.get("/alerts", { params: { limit: 5, est_lue: false } });
                const alerts = alertsData || [];
            } catch (e) { console.error('Alerts error:', e); }
            
        } catch (error) {
            console.error("Failed to load dashboard data:", error);
        } finally {
            isLoading = false;
        }
    }

    function formatNumber(value: number): string {
        return value.toLocaleString("fr-FR");
    }

    function formatCurrency(value: number): string {
        return new Intl.NumberFormat("fr-FR", {
            style: "currency",
            currency: "XOF",
            minimumFractionDigits: 0,
        }).format(value);
    }

    function getNiveauClass(niveau: string): string {
        switch (niveau) {
            case "critical":
                return "bg-red-100 text-red-700";
            case "warning":
                return "bg-yellow-100 text-yellow-700";
            default:
                return "bg-blue-100 text-blue-700";
        }
    }

    function getNiveauIcon(niveau: string): string {
        switch (niveau) {
            case "critical":
                return "🔴";
            case "warning":
                return "⚠️";
            default:
                return "ℹ️";
        }
    }
</script>

{#if isLoading}
    <div
        class="min-h-screen flex items-center justify-center bg-gradient-to-br from-green-50 to-emerald-100"
    >
        <div class="text-center">
            <div
                class="animate-spin rounded-full h-16 w-16 border-b-4 border-green-600 mx-auto"
            ></div>
            <p class="mt-6 text-gray-600 font-medium text-lg">
                Chargement du tableau de bord...
            </p>
            <p class="text-sm text-gray-400 mt-2">Farm Manager Burkina Faso</p>
        </div>
    </div>
{:else if $authStore.isAuthenticated}
    <div class="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
        <!-- Overlay pour mobile quand sidebar est ouverte -->
        {#if isSidebarOpen}
            <!-- svelte-ignore a11y-click-events-have-key-events -->
            <!-- svelte-ignore a11y-no-static-element-interactions -->
            <div
                class="fixed inset-0 bg-black bg-opacity-50 z-30 md:hidden"
                on:click={toggleSidebar}
            ></div>
        {/if}

        <!-- Sidebar -->
        <aside
            class="fixed left-0 top-0 z-40 h-screen w-64 bg-white shadow-lg border-r border-gray-200 transition-transform duration-300 ease-in-out md:translate-x-0"
            class:translate-x-0={isSidebarOpen}
            class:-translate-x-full={!isSidebarOpen}
        >
            <div class="h-full px-3 py-4 overflow-y-auto">
                <div class="flex items-center justify-between mb-8 pt-4">
                    <div class="flex items-center">
                        <div
                            class="w-12 h-12 bg-gradient-to-r from-green-600 to-emerald-600 rounded-xl flex items-center justify-center shadow-md"
                        >
                            <span class="text-white font-bold text-2xl">🌾</span>
                        </div>
                        <div class="ml-3">
                            <h1 class="text-xl font-bold text-gray-800">
                                Farm Manager
                            </h1>
                            <p class="text-xs text-gray-500">Burkina Faso</p>
                        </div>
                    </div>
                    <!-- Bouton fermeture sidebar sur mobile -->
                    <button
                        on:click={toggleSidebar}
                        class="md:hidden text-gray-500 hover:text-gray-700"
                    >
                        <span class="text-2xl">✕</span>
                    </button>
                </div>

                <nav class="space-y-1">
                    <a
                        href="/"
                        class="flex items-center px-4 py-3 text-gray-700 rounded-lg bg-green-50"
                        on:click={closeSidebar}
                    >
                        <span class="text-xl mr-3">📊</span>
                        <span class="font-medium">Tableau de bord</span>
                    </a>
                    {#if $isSuperAdmin}
                        <a
                            href="/users"
                            class="flex items-center px-4 py-3 text-gray-600 rounded-lg hover:bg-gray-50 hover:text-green-600 transition-colors"
                            on:click={closeSidebar}
                        >
                            <span class="text-xl mr-3">👥</span>
                            <span>Gestion des utilisateurs</span>
                            <span class="ml-auto text-xs bg-gray-100 px-2 py-0.5 rounded-full">
                                {stats.users_length}
                            </span>
                        </a>
                    {/if}
                    {#if canViewEnclos}
                        <a
                            href="/enclos"
                            class="flex items-center px-4 py-3 text-gray-600 rounded-lg hover:bg-gray-50 hover:text-green-600 transition-colors"
                            on:click={closeSidebar}
                        >
                            <span class="text-xl mr-3">🏠</span>
                            <span>Enclos</span>
                            <span
                                class="ml-auto text-xs bg-gray-100 px-2 py-0.5 rounded-full"
                                >{stats.enclos_length}</span
                            >
                        </a>
                    {/if}
                    {#if canViewBovins}
                        <a
                            href="/bovins"
                            class="flex items-center px-4 py-3 text-gray-600 rounded-lg hover:bg-gray-50 hover:text-green-600 transition-colors"
                            on:click={closeSidebar}
                        >
                            <span class="text-xl mr-3">🐄</span>
                            <span>Bovins</span>
                            <span
                                class="ml-auto text-xs bg-gray-100 px-2 py-0.5 rounded-full"
                                >{stats.bovins_length}</span
                            >
                        </a>
                    {/if}
                    {#if canViewOvins}
                        <a
                            href="/ovins"
                            class="flex items-center px-4 py-3 text-gray-600 rounded-lg hover:bg-gray-50 hover:text-green-600 transition-colors"
                            on:click={closeSidebar}
                        >
                            <span class="text-xl mr-3">🐑</span>
                            <span>Ovins</span>
                            <span
                                class="ml-auto text-xs bg-gray-100 px-2 py-0.5 rounded-full"
                                >{stats.ovins_length}</span
                            >
                        </a>
                    {/if}
                    {#if canViewCaprins}
                        <a
                            href="/caprins"
                            class="flex items-center px-4 py-3 text-gray-600 rounded-lg hover:bg-gray-50 hover:text-green-600 transition-colors"
                            on:click={closeSidebar}
                        >
                            <span class="text-xl mr-3">🐐</span>
                            <span>Caprins</span>
                            <span
                                class="ml-auto text-xs bg-gray-100 px-2 py-0.5 rounded-full"
                                >{stats.caprins_length}</span
                            >
                        </a>
                    {/if}
                    <!-- {#if canViewAvicoles}
                        <a
                            href="/avicoles"
                            class="flex items-center px-4 py-3 text-gray-600 rounded-lg hover:bg-gray-50 hover:text-green-600 transition-colors"
                            on:click={closeSidebar}
                        >
                            <span class="text-xl mr-3">🐔</span>
                            <span>Avicoles</span>
                            <span
                                class="ml-auto text-xs bg-gray-100 px-2 py-0.5 rounded-full"
                                >{stats.avicoles_length}</span
                            >
                        </a>
                    {/if}
                    {#if canViewPiscicoles}
                        <a
                            href="/piscicoles"
                            class="flex items-center px-4 py-3 text-gray-600 rounded-lg hover:bg-gray-50 hover:text-green-600 transition-colors"
                            on:click={closeSidebar}
                        >
                            <span class="text-xl mr-3">🐟</span>
                            <span>Piscicoles</span>
                            <span
                                class="ml-auto text-xs bg-gray-100 px-2 py-0.5 rounded-full"
                                >{stats.piscicoles_length}</span
                            >
                        </a>
                    {/if}
                    {#if canViewApiculture}
                        <a
                            href="/apiary"
                            class="flex items-center px-4 py-3 text-gray-600 rounded-lg hover:bg-gray-50 hover:text-green-600 transition-colors"
                            on:click={closeSidebar}
                        >
                            <span class="text-xl mr-3">🍯</span>
                            <span>Apiculture</span>
                            <span
                                class="ml-auto text-xs bg-gray-100 px-2 py-0.5 rounded-full"
                                >{stats.ruches_length}</span
                            >
                        </a>
                    {/if}
                    {#if canViewEntomoculture}
                        <a
                            href="/entomoculture"
                            class="flex items-center px-4 py-3 text-gray-600 rounded-lg hover:bg-gray-50 hover:text-green-600 transition-colors"
                            on:click={closeSidebar}
                        >
                            <span class="text-xl mr-3">🐛</span>
                            <span>Entomoculture</span>
                            <span
                                class="ml-auto text-xs bg-gray-100 px-2 py-0.5 rounded-full"
                                >{stats.nids_length}</span
                            >
                        </a>
                    {/if} -->
                    {#if canViewAccounting}
                        <a
                            href="/accounting"
                            class="flex items-center px-4 py-3 text-gray-600 rounded-lg hover:bg-gray-50 hover:text-green-600 transition-colors"
                            on:click={closeSidebar}
                        >
                            <span class="text-xl mr-3">💰</span>
                            <span>Comptabilité</span>
                        </a>
                    {/if}
                    <!-- <a
                        href="/compost"
                        class="flex items-center px-4 py-3 text-gray-600 rounded-lg hover:bg-gray-50 hover:text-green-600 transition-colors"
                        on:click={closeSidebar}
                    >
                        <span class="text-xl mr-3">🌱</span>
                        <span>Compostage</span>
                    </a>
                    <a
                        href="/vaccination"
                        class="flex items-center px-4 py-3 text-gray-600 rounded-lg hover:bg-gray-50 hover:text-green-600 transition-colors"
                        on:click={closeSidebar}
                    >
                        <span class="text-xl mr-3">💉</span>
                        <span>Vaccination</span>
                    </a> -->
                </nav>

                <div class="absolute bottom-4 left-0 right-0 px-3">
                    <div class="border-t border-gray-200 pt-4">
                        <!-- Profil utilisateur avec lien -->
                        <button
                            on:click={goToProfile}
                            class="w-full flex items-center px-4 py-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer group"
                        >
                            <div
                                class="w-8 h-8 bg-gradient-to-r from-green-600 to-emerald-600 rounded-full flex items-center justify-center flex-shrink-0"
                            >
                                <span class="text-white text-sm font-bold">
                                    {$authStore.user?.full_name?.charAt(0) ||
                                        $authStore.user?.username?.charAt(0) ||
                                        "U"}
                                </span>
                            </div>
                            <div class="ml-3 flex-1 text-left">
                                <p class="text-sm font-medium text-gray-800 group-hover:text-green-600 transition-colors">
                                    {$authStore.user?.full_name ||
                                        $authStore.user?.username}
                                </p>
                                <p class="text-xs text-gray-500">
                                    {$authStore.user?.roles?.[0] ||
                                        "Utilisateur"}
                                </p>
                            </div>
                            <div class="flex items-center gap-1">
                                <span class="text-xs text-gray-400 group-hover:text-green-600 transition-colors">
                                    ⚙️
                                </span>
                            </div>
                        </button>

                        <!-- Bouton déconnexion -->
                        <button
                            on:click={() => authStore.logout()}
                            class="w-full mt-2 flex items-center justify-center px-4 py-2 text-sm text-red-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                        >
                            <span class="mr-2">🚪</span>
                            Déconnexion
                        </button>
                    </div>
                </div>
            </div>
        </aside>

        <!-- Main Content -->
        <div class="md:ml-64">
            <!-- Top Navbar -->
            <nav
                class="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-30"
            >
                <div class="px-4 sm:px-6 lg:px-8">
                    <div class="flex justify-between items-center h-16">
                        <div class="flex items-center md:hidden">
                            <button
                                on:click={toggleSidebar}
                                class="text-gray-500 hover:text-gray-700 focus:outline-none"
                            >
                                <span class="text-2xl">☰</span>
                            </button>
                        </div>

                        <div class="flex-1 flex justify-end items-center gap-4">
                            <div
                                class="hidden sm:flex items-center text-gray-500 text-sm"
                            >
                                <span class="mr-2">📅</span>
                                <span
                                    >{new Date().toLocaleDateString("fr-FR", {
                                        weekday: "long",
                                        day: "numeric",
                                        month: "long",
                                        year: "numeric",
                                    })}</span
                                >
                            </div>

                            <!-- Lien rapide vers le profil -->
                            <a
                                href="/profile"
                                class="flex items-center gap-2 text-gray-500 hover:text-green-600 transition-colors text-sm"
                            >
                                <span class="text-xl">👤</span>
                                <span class="hidden sm:inline">Profil</span>
                            </a>

                            <a
                                href="/alerts"
                                class="relative text-gray-500 hover:text-gray-700"
                            >
                                <span class="text-xl">🔔</span>
                                {#if alerts.length > 0}
                                    <span
                                        class="absolute -top-1 -right-1 w-4 h-4 bg-red-500 text-white text-xs rounded-full flex items-center justify-center"
                                        >{alerts.length}</span
                                    >
                                {/if}
                            </a>
                        </div>
                    </div>
                </div>
            </nav>

            <!-- Main Content -->
            <main class="p-4 sm:p-6 lg:p-8">
                <!-- Hero Section -->
                <div
                    class="bg-gradient-to-r from-green-600 to-emerald-700 rounded-2xl shadow-lg overflow-hidden mb-8"
                >
                    <div class="px-6 py-8 sm:px-8 sm:py-10">
                        <div
                            class="flex items-center justify-between flex-wrap gap-4"
                        >
                            <div>
                                <div class="flex items-center gap-2 mb-2">
                                    <span class="text-2xl">🇧🇫</span>
                                    <span
                                        class="text-green-200 text-sm font-medium"
                                        >Burkina Faso</span
                                    >
                                </div>
                                <h1
                                    class="text-2xl sm:text-3xl font-bold text-white"
                                >
                                    Bonjour, {$authStore.user?.full_name ||
                                        $authStore.user?.username}
                                </h1>
                                <p class="text-green-100 mt-2">
                                    Bienvenue sur votre plateforme de gestion
                                    agricole connectée
                                </p>
                            </div>
                            <div
                                class="bg-white/10 rounded-xl px-4 py-3 text-center"
                            >
                                <p class="text-green-100 text-sm">Météo</p>
                                <p class="text-white text-2xl font-bold">
                                    --°C
                                </p>
                                <p class="text-green-100 text-xs">
                                    Chargement...
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- KPIs Grid - Dynamique selon les permissions -->
                <div
                    class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 mb-8"
                >
                    {#if canViewBovins}
                        <div
                            class="bg-white rounded-xl shadow-sm p-5 border border-gray-100 hover:shadow-md transition-shadow group"
                        >
                            <div class="flex items-center justify-between">
                                <div>
                                    <p class="text-gray-500 text-sm mb-1">
                                        Bovins
                                    </p>
                                    <p class="text-3xl font-bold text-gray-800">
                                        {formatNumber(stats.bovins_length)}
                                    </p>
                                </div>
                                <div
                                    class="w-14 h-14 bg-blue-100 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform"
                                >
                                    <span class="text-3xl">🐄</span>
                                </div>
                            </div>
                        </div>
                    {/if}

                    {#if canViewOvins}
                        <div
                            class="bg-white rounded-xl shadow-sm p-5 border border-gray-100 hover:shadow-md transition-shadow group"
                        >
                            <div class="flex items-center justify-between">
                                <div>
                                    <p class="text-gray-500 text-sm mb-1">
                                        Ovins
                                    </p>
                                    <p class="text-3xl font-bold text-gray-800">
                                        {formatNumber(stats.ovins_length)}
                                    </p>
                                </div>
                                <div
                                    class="w-14 h-14 bg-green-100 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform"
                                >
                                    <span class="text-3xl">🐑</span>
                                </div>
                            </div>
                        </div>
                    {/if}

                    {#if canViewCaprins}
                        <div
                            class="bg-white rounded-xl shadow-sm p-5 border border-gray-100 hover:shadow-md transition-shadow group"
                        >
                            <div class="flex items-center justify-between">
                                <div>
                                    <p class="text-gray-500 text-sm mb-1">
                                        Caprins
                                    </p>
                                    <p class="text-3xl font-bold text-gray-800">
                                        {formatNumber(stats.caprins_length)}
                                    </p>
                                </div>
                                <div
                                    class="w-14 h-14 bg-yellow-100 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform"
                                >
                                    <span class="text-3xl">🐐</span>
                                </div>
                            </div>
                        </div>
                    {/if}

                    <!-- {#if canViewAvicoles}
                        <div
                            class="bg-white rounded-xl shadow-sm p-5 border border-gray-100 hover:shadow-md transition-shadow group"
                        >
                            <div class="flex items-center justify-between">
                                <div>
                                    <p class="text-gray-500 text-sm mb-1">
                                        Avicoles
                                    </p>
                                    <p class="text-3xl font-bold text-gray-800">
                                        {formatNumber(stats.avicoles_length)}
                                    </p>
                                </div>
                                <div
                                    class="w-14 h-14 bg-purple-100 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform"
                                >
                                    <span class="text-3xl">🐔</span>
                                </div>
                            </div>
                        </div>
                    {/if} -->
                </div>

                <!-- Second Row KPIs -->
                <div
                    class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5 mb-8"
                >
                    <!-- {#if canViewPiscicoles}
                        <div
                            class="bg-gradient-to-r from-cyan-500 to-cyan-600 rounded-xl shadow-sm p-5 text-white"
                        >
                            <div class="flex items-center justify-between">
                                <div>
                                    <p class="text-cyan-100 text-sm mb-1">
                                        Piscicoles
                                    </p>
                                    <p class="text-3xl font-bold">
                                        {formatNumber(stats.piscicoles_length)}
                                    </p>
                                    <p class="text-cyan-100 text-xs mt-2">
                                        Total individus
                                    </p>
                                </div>
                                <div
                                    class="w-14 h-14 bg-white/20 rounded-2xl flex items-center justify-center"
                                >
                                    <span class="text-3xl">🐟</span>
                                </div>
                            </div>
                        </div>
                    {/if}

                    {#if canViewAvicoles}
                        <div
                            class="bg-gradient-to-r from-orange-500 to-orange-600 rounded-xl shadow-sm p-5 text-white"
                        >
                            <div class="flex items-center justify-between">
                                <div>
                                    <p class="text-orange-100 text-sm mb-1">
                                        Production d'œufs
                                    </p>
                                    <p class="text-3xl font-bold">
                                        {formatNumber(
                                            productionStats.oeufs_jour,
                                        )}
                                    </p>
                                    <p class="text-orange-100 text-xs mt-2">
                                        par jour
                                    </p>
                                </div>
                                <div
                                    class="w-14 h-14 bg-white/20 rounded-2xl flex items-center justify-center"
                                >
                                    <span class="text-3xl">🥚</span>
                                </div>
                            </div>
                        </div>
                    {/if} -->

                    {#if canViewAccounting}
                        <div
                            class="bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-xl shadow-sm p-5 text-white"
                        >
                            <div class="flex items-center justify-between">
                                <div>
                                    <p class="text-emerald-100 text-sm mb-1">
                                        Bénéfice du mois
                                    </p>
                                    <p class="text-2xl font-bold">
                                        {formatCurrency(
                                            financialStats.benefice_mois,
                                        )}
                                    </p>
                                    <p class="text-emerald-100 text-xs mt-2">
                                        CA: {formatCurrency(
                                            financialStats.ca_mois,
                                        )}
                                    </p>
                                </div>
                                <div
                                    class="w-14 h-14 bg-white/20 rounded-2xl flex items-center justify-center"
                                >
                                    <span class="text-3xl">💰</span>
                                </div>
                            </div>
                        </div>
                    {/if}
                </div>

                <!-- Activités récentes et Alertes -->
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                    <!-- Activités récentes -->
                    <div
                        class="lg:col-span-2 bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden"
                    >
                        <div
                            class="px-6 py-4 border-b border-gray-100 flex justify-between items-center"
                        >
                            <h2 class="text-lg font-semibold text-gray-800">
                                📋 Activités récentes
                            </h2>
                            <a
                                href="/reports"
                                class="text-sm text-green-600 hover:text-green-700"
                                >Voir tout →</a
                            >
                        </div>
                        <div class="divide-y divide-gray-100">
                            {#if recentActivities.length === 0}
                                <div class="px-6 py-8 text-center">
                                    <span class="text-4xl mb-2">📭</span>
                                    <p class="text-gray-500 text-sm">
                                        Aucune activité récente
                                    </p>
                                </div>
                            {:else}
                                {#each recentActivities as activity}
                                    <div
                                        class="px-6 py-4 hover:bg-gray-50 transition-colors"
                                    >
                                        <div class="flex items-center gap-3">
                                            <div
                                                class="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center text-xl"
                                            >
                                                {activity.icon || "📌"}
                                            </div>
                                            <div class="flex-1">
                                                <p
                                                    class="text-sm font-medium text-gray-800"
                                                >
                                                    {activity.description}
                                                </p>
                                                <p
                                                    class="text-xs text-gray-500"
                                                >
                                                    {activity.entity_name || ""}
                                                </p>
                                            </div>
                                            <div class="text-right">
                                                <p
                                                    class="text-xs text-gray-400"
                                                >
                                                    {new Date(
                                                        activity.created_at,
                                                    ).toLocaleDateString(
                                                        "fr-FR",
                                                    )}
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                {/each}
                            {/if}
                        </div>
                    </div>

                    <!-- Alertes -->
                    <div
                        class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden"
                    >
                        <div class="px-6 py-4 border-b border-gray-100">
                            <h2 class="text-lg font-semibold text-gray-800">
                                ⚠️ Alertes
                            </h2>
                        </div>
                        <div class="divide-y divide-gray-100">
                            {#if alerts.length === 0}
                                <div class="px-6 py-8 text-center">
                                    <span class="text-4xl mb-2">✅</span>
                                    <p class="text-gray-500 text-sm">
                                        Aucune alerte
                                    </p>
                                </div>
                            {:else}
                                {#each alerts as alert}
                                    <div class="px-6 py-4">
                                        <div class="flex items-start gap-3">
                                            <span class="text-lg"
                                                >{getNiveauIcon(
                                                    alert.niveau,
                                                )}</span
                                            >
                                            <div class="flex-1">
                                                <div
                                                    class="flex items-center gap-2 mb-1"
                                                >
                                                    <span
                                                        class={`px-2 py-0.5 rounded-full text-xs font-medium ${getNiveauClass(alert.niveau)}`}
                                                    >
                                                        {alert.niveau}
                                                    </span>
                                                    <span
                                                        class="text-xs text-gray-400"
                                                        >{new Date(
                                                            alert.date_alerte,
                                                        ).toLocaleDateString(
                                                            "fr-FR",
                                                        )}</span
                                                    >
                                                </div>
                                                <p
                                                    class="text-sm text-gray-700"
                                                >
                                                    {alert.message}
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                {/each}
                            {/if}
                        </div>
                        <div
                            class="px-6 py-3 bg-gray-50 border-t border-gray-100"
                        >
                            <a
                                href="/alerts"
                                class="text-sm text-green-600 hover:text-green-700 flex items-center gap-1"
                            >
                                Gérer les alertes →
                            </a>
                        </div>
                    </div>
                </div>

                <!-- Footer -->
                <footer class="mt-8 text-center text-gray-400 text-sm">
                    <p>
                        © 2026 Farm Manager Burkina Faso - Solution de gestion
                        agricole connectée
                    </p>
                </footer>
            </main>
        </div>
    </div>
{/if}