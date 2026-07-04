// src/app.d.ts
/// <reference types="@sveltejs/kit" />

declare global {
	namespace App {
		// Interface pour les données du layout
		interface Locals {
			user?: {
				id: number;
				email: string;
				roles: string[];
			};
		}

		interface PageData {
			// Types pour les données de page
		}

		interface Platform {
			// Types spécifiques à la plateforme
		}

		interface PrivateEnv {
			// Variables d'environnement privées
		}

		interface PublicEnv {
			// Variables d'environnement publiques
		}
	}
}

export { };