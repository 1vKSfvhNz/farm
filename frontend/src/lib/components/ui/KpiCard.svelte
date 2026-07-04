<script lang="ts">
  type ColorType =
    | "primary"
    | "secondary"
    | "accent"
    | "info"
    | "warning"
    | "success"
    | "danger";

  export let stat: {
    label: string;
    value: string;
    change: string;
    icon: string;
    color: ColorType;
  };

  const colorClasses: Record<ColorType, string> = {
    primary: "from-primary-500 to-primary-600",
    secondary: "from-secondary-500 to-secondary-600",
    accent: "from-accent-500 to-accent-600",
    info: "from-blue-500 to-blue-600",
    warning: "from-yellow-500 to-yellow-600",
    success: "from-green-500 to-green-600",
    danger: "from-red-500 to-red-600",
  };

  function getColorClass(color: ColorType): string {
    return colorClasses[color];
  }
</script>

<div class="stat-card group">
  <div class="flex items-start justify-between">
    <div>
      <p class="text-sm font-medium text-gray-500">
        {stat.label}
      </p>

      <p class="text-2xl font-bold text-gray-900 mt-1">
        {stat.value}
      </p>

      <p class="text-xs text-green-600 mt-2 flex items-center gap-1">
        <svg
          class="w-3 h-3"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M5 10l7-7m0 0l7 7m-7-7v18"
          />
        </svg>

        {stat.change} vs mois dernier
      </p>
    </div>

    <div class={`stat-icon bg-gradient-to-br ${getColorClass(stat.color)}`}>
      <svg
        class="w-6 h-6"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d={stat.icon}
        />
      </svg>
    </div>
  </div>
</div>
