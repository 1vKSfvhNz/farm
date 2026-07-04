// src/routes/ovins/[id]/+page.ts
import { error } from '@sveltejs/kit';
import type { PageLoad } from './$types';
import { get } from 'svelte/store';
import { ovinsStore } from '$lib/stores/ovins';

export const load: PageLoad = async ({ params }) => {
    const id = parseInt(params.id);
    
    if (isNaN(id) || id <= 0) {
        throw error(400, 'ID de ovin invalide');
    }
    
    // ✅ Récupérer le ovin depuis le store avec get() (plus simple)
    const ovin = ovinsStore.getOvin();
    
    // Si le ovin n'est pas dans le store, rediriger vers la liste
    if (!ovin) {
        throw error(404, 'Ovin non trouvé');
    }
    
    return {
        id: params.id,
        ovin: ovin
    };
};