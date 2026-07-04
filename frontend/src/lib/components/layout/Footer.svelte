<!-- lib/components/layout/Footer.svelte -->
<script lang="ts">
    let currentYear = new Date().getFullYear();
</script>

<footer class="bg-white border-t border-gray-200 py-4 px-6">
    <div
        class="flex flex-col sm:flex-row justify-between items-center gap-2 text-sm text-gray-500"
    >
        <div class="flex items-center gap-4">
            <span>© {currentYear} Farm Manager</span>
            <span class="w-1 h-1 bg-gray-300 rounded-full"></span>
            <span>Tous droits réservés</span>
        </div>
        <div class="flex items-center gap-4">
            <a
                href="/mentions-legales"
                class="hover:text-primary-600 transition-colors"
                >Mentions légales</a
            >
            <a
                href="/confidentialite"
                class="hover:text-primary-600 transition-colors"
                >Confidentialité</a
            >
            <a href="/contact" class="hover:text-primary-600 transition-colors"
                >Contact</a
            >
        </div>
    </div>
</footer>
