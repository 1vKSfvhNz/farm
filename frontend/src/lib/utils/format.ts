// lib/utils/format.ts
export const formatNumber = (value: number | null | undefined, decimals: number = 2): string => {
    if (value === null || value === undefined) return '-';
    return value.toLocaleString('fr-FR', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
};

export const formatCurrency = (value: number | null | undefined): string => {
    if (value === null || value === undefined) return '-';
    return new Intl.NumberFormat('fr-FR', {
        style: 'currency',
        currency: 'EUR',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
};

export const formatPercentage = (value: number | null | undefined, decimals: number = 1): string => {
    if (value === null || value === undefined) return '-';
    return `${value.toLocaleString('fr-FR', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    })}%`;
};

export const formatWeight = (kg: number | null | undefined): string => {
    if (kg === null || kg === undefined) return '-';
    if (kg < 1) {
        return `${(kg * 1000).toFixed(0)} g`;
    }
    return `${kg.toLocaleString('fr-FR', { minimumFractionDigits: 1, maximumFractionDigits: 1 })} kg`;
};

export const formatVolume = (m3: number | null | undefined): string => {
    if (m3 === null || m3 === undefined) return '-';
    return `${m3.toLocaleString('fr-FR', { minimumFractionDigits: 1, maximumFractionDigits: 1 })} m³`;
};

export const formatArea = (m2: number | null | undefined): string => {
    if (m2 === null || m2 === undefined) return '-';
    return `${m2.toLocaleString('fr-FR', { minimumFractionDigits: 0, maximumFractionDigits: 0 })} m²`;
};

export const formatTemperature = (celsius: number | null | undefined): string => {
    if (celsius === null || celsius === undefined) return '-';
    return `${celsius.toFixed(1)}°C`;
};

export const formatpH = (ph: number | null | undefined): string => {
    if (ph === null || ph === undefined) return '-';
    return ph.toFixed(1);
};

export const truncateText = (text: string | null | undefined, maxLength: number = 50): string => {
    if (!text) return '-';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
};

export const capitalizeFirstLetter = (str: string | null | undefined): string => {
    if (!str) return '-';
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};