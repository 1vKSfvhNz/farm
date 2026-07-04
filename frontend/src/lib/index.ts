// lib/index.ts - Point d'entrée unique pour l'API
// Exports des types
export * from './types/auth';
export * from './types/animal';
export * from './types/bovin';
export * from './types/ovin';
export * from './types/caprin';
export * from './types/avicole';
export * from './types/piscicole';
export * from './types/apiary';
export * from './types/entomoculture';
export * from './types/enclos';
export * from './types/accounting';
export * from './types/vaccination';
export * from './types/compost';
export * from './types/pesee';
export * from './types/alimentation';
export * from './types/water_quality';
export * from './types/bea';
export * from './types/predictions';
export * from './types/alerts';
export * from './types/pagination';
export * from './types/dashboard';
export * from './types/experimental';
export * from './types/video';

// Exports des API
export { apiClient } from './api/client';
export { authApi } from './api/auth';
export { usersApi } from './api/users';
export { bovinsApi } from './api/bovins';
export { ovinsApi } from './api/ovins';
export { caprinsApi } from './api/caprins';
export { avicolesApi } from './api/avicoles';
export { piscicolesApi } from './api/piscicoles';
export { apiaryApi } from './api/apiary';
export { entomocultureApi } from './api/entomoculture';
export { enclosApi } from './api/enclos';
export { accountingApi } from './api/accounting';
export { vaccinationApi } from './api/vaccination';
export { compostApi } from './api/compost';
export { peseeApi } from './api/pesee';
export { alimentationApi } from './api/alimentation';
export { waterQualityApi } from './api/water_quality';
export { beaApi } from './api/bea';
export { predictionsApi } from './api/predictions';
export { alertsApi } from './api/alerts';
export { dashboardApi } from './api/dashboard';
export { exportsApi } from './api/exports';
export { weatherApi } from './api/weather';
export { experimentalApi } from './api/experimental';
export { videoApi } from './api/video';

// Exports des stores
export { authStore } from './stores/auth';
export { notificationStore } from './stores/notifications';