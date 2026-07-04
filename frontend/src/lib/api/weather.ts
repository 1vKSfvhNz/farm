// lib/api/weather.ts (corrigé)
import { apiClient } from './client';

export const weatherApi = {
    getCurrentWeather: (latitude?: number, longitude?: number) =>
        apiClient.get('/weather/current', { params: { latitude, longitude } }),

    getForecast: (latitude?: number, longitude?: number, days?: number) =>
        apiClient.get('/weather/forecast', { params: { latitude, longitude, days } })
};