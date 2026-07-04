// lib/utils/validation.ts
export const isValidEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@([^\s@.,]+\.)+[^\s@.,]{2,}$/;
    return emailRegex.test(email);
};

export const isValidPhone = (phone: string): boolean => {
    const phoneRegex = /^(\+33|0)[1-9](\d{2}){4}$/;
    return phoneRegex.test(phone.replace(/\s/g, ''));
};

export const isValidSiret = (siret: string): boolean => {
    const siretRegex = /^\d{14}$/;
    return siretRegex.test(siret);
};

export const isValidNumber = (value: any, min?: number, max?: number): boolean => {
    const num = Number(value);
    if (isNaN(num)) return false;
    if (min !== undefined && num < min) return false;
    if (max !== undefined && num > max) return false;
    return true;
};

export const isValidDate = (date: string): boolean => {
    const d = new Date(date);
    return d instanceof Date && !isNaN(d.getTime());
};

export const isFutureDate = (date: string): boolean => {
    if (!isValidDate(date)) return false;
    return new Date(date) > new Date();
};

export const isPastDate = (date: string): boolean => {
    if (!isValidDate(date)) return false;
    return new Date(date) < new Date();
};

export const validateRequired = (value: any): boolean => {
    if (value === null || value === undefined) return false;
    if (typeof value === 'string') return value.trim().length > 0;
    if (typeof value === 'number') return true;
    if (Array.isArray(value)) return value.length > 0;
    return !!value;
};

export const validateMinLength = (value: string, min: number): boolean => {
    return value.length >= min;
};

export const validateMaxLength = (value: string, max: number): boolean => {
    return !value || value.length <= max;
};

export const validateRange = (value: number, min: number, max: number): boolean => {
    return value >= min && value <= max;
};