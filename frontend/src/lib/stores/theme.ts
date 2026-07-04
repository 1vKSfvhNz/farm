// frontend/src/lib/stores/theme.ts
import { writable } from 'svelte/store';
import { browser } from '../utils/browser';

function createThemeStore() {
    const initialTheme = browser && localStorage.getItem('theme') === 'dark' ? 'dark' : 'light';
    const { subscribe, set, update } = writable(initialTheme);

    if (browser) {
        // Appliquer le thème initial
        if (initialTheme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }

    return {
        subscribe,
        toggle: () => {
            update(theme => {
                const newTheme = theme === 'dark' ? 'light' : 'dark';
                if (browser) {
                    localStorage.setItem('theme', newTheme);
                    if (newTheme === 'dark') {
                        document.documentElement.classList.add('dark');
                    } else {
                        document.documentElement.classList.remove('dark');
                    }
                }
                return newTheme;
            });
        },
        set: (theme: 'dark' | 'light') => {
            set(theme);
            if (browser) {
                localStorage.setItem('theme', theme);
                if (theme === 'dark') {
                    document.documentElement.classList.add('dark');
                } else {
                    document.documentElement.classList.remove('dark');
                }
            }
        }
    };
}

export const theme = createThemeStore();