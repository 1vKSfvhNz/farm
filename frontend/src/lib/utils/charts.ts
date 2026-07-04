// lib/utils/charts.ts
import type { ChartDataPoint, ChartSeries, GrowthChartData } from '../types/charts';

export const formatChartData = (data: ChartDataPoint[]): { labels: string[]; values: number[] } => {
    return {
        labels: data.map(d => d.label),
        values: data.map(d => d.value)
    };
};

export const formatSeriesData = (series: ChartSeries[]): {
    labels: string[];
    datasets: { name: string; data: number[]; color?: string }[];
} => {
    const allLabels = new Set<string>();
    series.forEach(s => {
        s.data.forEach(d => allLabels.add(d.label));
    });

    const labels = Array.from(allLabels);

    const datasets = series.map(s => ({
        name: s.name,
        data: labels.map(label => {
            const point = s.data.find(d => d.label === label);
            return point?.value ?? 0;
        }),
        color: s.color
    }));

    return { labels, datasets };
};

export const prepareGrowthChart = (data: GrowthChartData): {
    labels: string[];
    actual: number[];
    predictedMin: number[];
    predictedMax: number[];
    predictedAvg: number[];
} => {
    const labels = data.weighings.map(w => `J${w.age_jours}`);
    const actual = data.weighings.map(w => w.poids);

    let predictedMin: number[] = [];
    let predictedMax: number[] = [];
    let predictedAvg: number[] = [];

    if (data.predictions) {
        predictedMin = data.predictions.map(p => p.poids_min);
        predictedMax = data.predictions.map(p => p.poids_max);
        predictedAvg = data.predictions.map(p => p.poids_moyen);
    }

    return { labels, actual, predictedMin, predictedMax, predictedAvg };
};

export const prepareFinancialChart = (data: {
    depenses: ChartDataPoint[];
    recettes: ChartDataPoint[];
}): {
    labels: string[];
    depenses: number[];
    recettes: number[];
    benefice: number[];
} => {
    const allLabels = new Set<string>();
    data.depenses.forEach(d => allLabels.add(d.label));
    data.recettes.forEach(d => allLabels.add(d.label));

    const labels = Array.from(allLabels).sort();

    const depenses = labels.map(label => {
        const point = data.depenses.find(d => d.label === label);
        return point?.value ?? 0;
    });

    const recettes = labels.map(label => {
        const point = data.recettes.find(d => d.label === label);
        return point?.value ?? 0;
    });

    const benefice = recettes.map((r, i) => r - depenses[i]);

    return { labels, depenses, recettes, benefice };
};

export const getChartColors = (count: number): string[] => {
    const colors = [
        '#3b82f6', // blue
        '#ef4444', // red
        '#10b981', // green
        '#f59e0b', // amber
        '#8b5cf6', // violet
        '#ec4899', // pink
        '#06b6d4', // cyan
        '#84cc16', // lime
        '#f97316', // orange
        '#6366f1', // indigo
    ];

    if (count <= colors.length) return colors.slice(0, count);

    // Générer des couleurs supplémentaires
    const additionalColors = [];
    for (let i = colors.length; i < count; i++) {
        const hue = (i * 137) % 360;
        additionalColors.push(`hsl(${hue}, 70%, 50%)`);
    }

    return [...colors, ...additionalColors];
};