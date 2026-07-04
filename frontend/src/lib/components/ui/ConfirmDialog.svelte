<!-- lib/components/ui/ConfirmDialog.svelte -->
<script lang="ts">
    import { createEventDispatcher } from "svelte";
    import Button from "./Button.svelte";
    import Modal from "./Modal.svelte";

    export let open: boolean = false;
    export let title: string = "Confirmation";
    export let message: string = "Êtes-vous sûr de vouloir continuer ?";
    export let confirmText: string = "Confirmer";
    export let cancelText: string = "Annuler";
    export let confirmVariant: "primary" | "danger" = "primary";

    const dispatch = createEventDispatcher();

    function handleConfirm() {
        dispatch("confirm");
        open = false;
    }

    function handleCancel() {
        dispatch("cancel");
        open = false;
    }
</script>

<Modal {open} {title} on:close={handleCancel} size="sm">
    <p class="text-gray-600">{message}</p>
    <div slot="footer" class="flex justify-end gap-3">
        <Button on:click={handleCancel} variant="outline">
            {cancelText}
        </Button>
        <Button on:click={handleConfirm} variant={confirmVariant}>
            {confirmText}
        </Button>
    </div>
</Modal>
