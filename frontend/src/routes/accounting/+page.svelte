<!-- src/routes/accounting/+page.svelte -->
<script lang="ts">
  import { onMount } from "svelte";
  import { accountingApi } from "$lib/api/accounting";
  import AccountingTable from "$lib/components/tables/AccountingTable.svelte";
  import Button from "$lib/components/ui/Button.svelte";
  import Modal from "$lib/components/ui/Modal.svelte";
  import AccountingForm from "$lib/components/forms/AccountingForm.svelte";
  import KpiCard from "$lib/components/dashboard/KpiCard.svelte";

  let transactions: any[] = [];
  let summary: any = null;
  let loading = true;
  let showModal = false;
  let selectedType: "depense" | "recette" = "depense";

  onMount(async () => {
    await loadData();
  });

  async function loadData() {
    loading = true;
    try {
      const [transactionsRes, summaryRes] = await Promise.all([
        accountingApi.getDepenses({ limit: 100 }),
        accountingApi.getSummary(),
      ]);
      transactions = transactionsRes.items;
      summary = summaryRes;
    } catch (error) {
      console.error("Failed to load data:", error);
    } finally {
      loading = false;
    }
  }

  function handleAddDepense() {
    selectedType = "depense";
    showModal = true;
  }

  function handleAddRecette() {
    selectedType = "recette";
    showModal = true;
  }

  async function handleSubmit(formData: any) {
    try {
      if (formData.type === "depense") {
        await accountingApi.createDepense(formData);
      } else {
        await accountingApi.createRecette(formData);
      }
      showModal = false;
      await loadData();
    } catch (error) {
      console.error("Failed to save transaction:", error);
    }
  }
</script>

<div class="space-y-6">
  <h1 class="text-2xl font-bold text-gray-900">Comptabilité</h1>

  <!-- KPIs -->
  <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
    <KpiCard
      title="Chiffre d'affaires"
      value={`${(summary?.total_recettes || 0).toLocaleString("fr-FR")} €`}
      icon="M2.25 18.75a60.07 60.07 0 0115.797 2.101c.727.198 1.453-.342 1.453-1.096V18.75M3.75 4.5v.75A.75.75 0 013 6h-.75m0 0v11.25m0 0H21m-1.5 0h.75m0 0v-7.5a.75.75 0 00-.75-.75h-3.75m-6 0h3.75"
      color="success"
    />
    <KpiCard
      title="Dépenses"
      value={`${(summary?.total_depenses || 0).toLocaleString("fr-FR")} €`}
      icon="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375 7.444 2.25 12 2.25s8.25 1.847 8.25 4.125zm0 5.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125"
      color="danger"
    />
    <KpiCard
      title="Bénéfice"
      value={`${(summary?.benefice || 0).toLocaleString("fr-FR")} €`}
      icon="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
      color="primary"
    />
    <KpiCard
      title="Marge brute"
      value={`${(summary?.marge_brute_pourcent || 0).toFixed(1)}%`}
      icon="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75z"
      color="warning"
    />
  </div>

  <div class="flex justify-end gap-2">
    <Button on:click={handleAddDepense} variant="danger">
      <svg
        class="w-4 h-4 mr-1"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M12 4v16m8-8H4"
        />
      </svg>
      Ajouter une dépense
    </Button>
    <Button on:click={handleAddRecette} variant="success">
      <svg
        class="w-4 h-4 mr-1"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M12 4v16m8-8H4"
        />
      </svg>
      Ajouter une recette
    </Button>
  </div>

  <AccountingTable
    {transactions}
    {loading}
    on:edit={() => {}}
    on:delete={() => {}}
    on:export={() => {}}
    on:add={handleAddDepense}
  />

  <Modal
    open={showModal}
    title={selectedType === "depense"
      ? "Ajouter une dépense"
      : "Ajouter une recette"}
    on:close={() => (showModal = false)}
  >
    <AccountingForm
      type={selectedType}
      formData={{
        categorie: "",
        montant: 0,
        date: new Date().toISOString().split("T")[0],
      }}
      loading={false}
      on:submit={handleSubmit}
      on:cancel={() => (showModal = false)}
    />
  </Modal>
</div>
