// src/routes/caprins/[id]/+page.ts
import { error } from '@sveltejs/kit';
import type { PageLoad } from './$types';
import { get } from 'svelte/store';
import { caprinsStore } from '$lib/stores/caprins';

export const load: PageLoad = async ({ params }) => {
    const id = parseInt(params.id);
    
    if (isNaN(id) || id <= 0) {
        throw error(400, 'ID de caprin invalide');
    }
    
    // ✅ Récupérer le caprin depuis le store avec get() (plus simple)
    const caprin = caprinsStore.getCaprin();
    
    // Si le caprin n'est pas dans le store, rediriger vers la liste
    if (!caprin) {
        throw error(404, 'Caprin non trouvé');
    }
    
    return {
        id: params.id,
        caprin: caprin
    };
};