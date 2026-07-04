// src/routes/bovins/[id]/+page.ts
import { error } from '@sveltejs/kit';
import type { PageLoad } from './$types';
import { bovinsStore } from '$lib/stores/bovins';

export const load: PageLoad = async ({ params }) => {
    const id = parseInt(params.id);
    
    if (isNaN(id) || id <= 0) {
        throw error(400, 'ID de bovin invalide');
    }
    
    // ✅ Récupérer le bovin depuis le store avec get() (plus simple)
    const bovin = bovinsStore.getBovin();
    
    // Si le bovin n'est pas dans le store, rediriger vers la liste
    if (!bovin) {
        throw error(404, 'Bovin non trouvé');
    }
    
    return {
        id: params.id,
        bovin: bovin
    };
};