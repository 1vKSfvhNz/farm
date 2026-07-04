import { API_URL } from "$lib/api/client";

export function getFullPhotoUrl(photoUrl: string | null | undefined): string | null {
    if (!photoUrl) return null;
    if (photoUrl.startsWith('http://') || photoUrl.startsWith('https://')) {
        return photoUrl;
    }
    // Supprimer /api/v1 de l'URL de base si présent
    const baseUrl = API_URL.replace('/api/v1', '');
    return `${baseUrl}${photoUrl}`;
}

// Convertir l'image en base64
export function fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

export function handleImageError(event: Event) {
    const target = event.currentTarget as HTMLImageElement;
    target.style.display = 'none';

    const fallback = target.nextElementSibling as HTMLElement;
    if (fallback) {
        fallback.classList.remove('hidden');
    }
}

