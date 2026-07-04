// lib/utils/date.ts
export const formatDate = (date: string | Date | null | undefined): string => {
    if (!date) return '-';
    const d = typeof date === 'string' ? new Date(date) : date;
    return d.toLocaleDateString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric'
    });
};

export const formatDateTime = (date: string | Date | null | undefined): string => {
    if (!date) return '-';
    const d = typeof date === 'string' ? new Date(date) : date;
    return d.toLocaleDateString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
};

export const formatTime = (date: string | Date | null | undefined): string => {
    if (!date) return '-';
    const d = typeof date === 'string' ? new Date(date) : date;
    return d.toLocaleTimeString('fr-FR', {
        hour: '2-digit',
        minute: '2-digit'
    });
};

export const getAgeInDays = (birthDate: string | Date): number => {
    const birth = typeof birthDate === 'string' ? new Date(birthDate) : birthDate;
    const today = new Date();
    const diffTime = Math.abs(today.getTime() - birth.getTime());
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
};

export const getAgeInMonths = (birthDate: string | Date): number => {
    const birth = typeof birthDate === 'string' ? new Date(birthDate) : birthDate;
    const today = new Date();
    const months = (today.getFullYear() - birth.getFullYear()) * 12;
    return months + (today.getMonth() - birth.getMonth());
};

export const getAgeInYears = (birthDate: string | Date): number => {
    return Math.floor(getAgeInMonths(birthDate) / 12);
};

export const formatAge = (birthDate: string | Date | null | undefined): string => {
    if (!birthDate) return '-';
    const years = getAgeInYears(birthDate);
    const months = getAgeInMonths(birthDate) % 12;
    const days = getAgeInDays(birthDate) % 30;

    if (years > 0) {
        return `${years} an${years > 1 ? 's' : ''}${months > 0 ? ` ${months} mois` : ''}`;
    }
    if (months > 0) {
        return `${months} mois${days > 0 ? ` ${days} j` : ''}`;
    }
    return `${days} jour${days > 1 ? 's' : ''}`;
};

export const isDateInPast = (date: string | Date): boolean => {
    const d = typeof date === 'string' ? new Date(date) : date;
    return d < new Date();
};

export const isDateInFuture = (date: string | Date): boolean => {
    const d = typeof date === 'string' ? new Date(date) : date;
    return d > new Date();
};

export const daysBetween = (start: string | Date, end: string | Date): number => {
    const startDate = typeof start === 'string' ? new Date(start) : start;
    const endDate = typeof end === 'string' ? new Date(end) : end;
    const diffTime = Math.abs(endDate.getTime() - startDate.getTime());
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
};