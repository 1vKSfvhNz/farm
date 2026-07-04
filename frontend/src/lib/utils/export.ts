// lib/utils/export.ts - Version corrigée
import { apiClient } from '../api/client';

export const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
};

export const exportToCSV = async (
    exportFn: () => Promise<Blob>,
    filename: string
): Promise<void> => {
    try {
        const blob = await exportFn();
        downloadBlob(blob, filename);
    } catch (error) {
        console.error('Export failed:', error);
        throw error;
    }
};

export const exportAnimals = async (espece?: string, enclos_id?: number): Promise<void> => {
    const blob = await apiClient.get<Blob>('/exports/animals/csv', {
        params: { espece, enclos_id },
        responseType: 'blob'
    });
    downloadBlob(blob, `animaux_${new Date().toISOString().split('T')[0]}.csv`);
};

export const exportFinancial = async (startDate: string, endDate: string): Promise<void> => {
    const blob = await apiClient.get<Blob>('/exports/financial/csv', {
        params: { start_date: startDate, end_date: endDate },
        responseType: 'blob'
    });
    downloadBlob(blob, `finances_${startDate}_${endDate}.csv`);
};

export const exportWeighings = async (espece?: string, startDate?: string, endDate?: string): Promise<void> => {
    const blob = await apiClient.get<Blob>('/exports/pesees/csv', {
        params: { espece, start_date: startDate, end_date: endDate },
        responseType: 'blob'
    });
    downloadBlob(blob, `pesees_${new Date().toISOString().split('T')[0]}.csv`);
};