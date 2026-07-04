<!-- lib/components/video/ROIOverlay.svelte -->
<script lang="ts">
    import { onMount } from "svelte";

    export let imageUrl: string;
    export let rois: Array<{
        id: number;
        name: string;
        x: number;
        y: number;
        width: number;
        height: number;
        color?: string;
    }> = [];
    export let editable: boolean = false;
    export let onROISelect: (roi: (typeof rois)[0]) => void = () => {};

    let canvas: HTMLCanvasElement;
    let ctx: CanvasRenderingContext2D | null = null;
    let image = new Image();
    let isDrawing = false;
    let currentROI: {
        x: number;
        y: number;
        width: number;
        height: number;
    } | null = null;
    let startX = 0;
    let startY = 0;

    const colors = [
        "#3b82f6",
        "#ef4444",
        "#10b981",
        "#f59e0b",
        "#8b5cf6",
        "#ec4899",
    ];

    function draw() {
        if (!ctx || !canvas) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (image.complete && image.naturalWidth > 0) {
            ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        }

        // Dessiner les ROIs existants
        rois.forEach((roi, index) => {
            const color = roi.color || colors[index % colors.length];
            ctx!.strokeStyle = color;
            ctx!.lineWidth = 2;
            ctx!.strokeRect(roi.x, roi.y, roi.width, roi.height);

            ctx!.fillStyle = color + "40";
            ctx!.fillRect(roi.x, roi.y, roi.width, roi.height);

            ctx!.fillStyle = color;
            ctx!.font = "12px Arial";
            ctx!.fillText(roi.name, roi.x + 5, roi.y + 20);
        });

        // Dessiner le ROI en cours de création
        if (currentROI) {
            ctx.strokeStyle = "#10b981";
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(
                currentROI.x,
                currentROI.y,
                currentROI.width,
                currentROI.height,
            );
            ctx.setLineDash([]);
        }
    }

    function handleMouseDown(e: MouseEvent) {
        if (!editable || !canvas) return;
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        startX = (e.clientX - rect.left) * scaleX;
        startY = (e.clientY - rect.top) * scaleY;
        isDrawing = true;
    }

    function handleMouseMove(e: MouseEvent) {
        if (!editable || !isDrawing || !canvas) return;
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const currentX = (e.clientX - rect.left) * scaleX;
        const currentY = (e.clientY - rect.top) * scaleY;

        currentROI = {
            x: Math.min(startX, currentX),
            y: Math.min(startY, currentY),
            width: Math.abs(currentX - startX),
            height: Math.abs(currentY - startY),
        };
        draw();
    }

    function handleMouseUp() {
        if (!editable || !isDrawing || !currentROI) return;
        if (currentROI.width > 10 && currentROI.height > 10) {
            const name = prompt(
                "Nom de la zone d'intérêt:",
                `ROI ${rois.length + 1}`,
            );
            if (name) {
                onROISelect({
                    id: Date.now(),
                    name,
                    x: currentROI.x,
                    y: currentROI.y,
                    width: currentROI.width,
                    height: currentROI.height,
                });
            }
        }
        isDrawing = false;
        currentROI = null;
        draw();
    }

    function resizeCanvas() {
        if (!canvas || !image.complete) return;
        const container = canvas.parentElement;
        if (container) {
            const maxWidth = container.clientWidth - 32;
            const ratio = image.naturalWidth / image.naturalHeight;
            canvas.width = Math.min(maxWidth, image.naturalWidth);
            canvas.height = canvas.width / ratio;
            draw();
        }
    }

    onMount(() => {
        if (!canvas) return;
        ctx = canvas.getContext("2d");

        image.onload = () => {
            resizeCanvas();
            draw();
        };
        image.src = imageUrl;

        canvas.addEventListener("mousedown", handleMouseDown);
        canvas.addEventListener("mousemove", handleMouseMove);
        canvas.addEventListener("mouseup", handleMouseUp);
        window.addEventListener("resize", resizeCanvas);

        return () => {
            canvas?.removeEventListener("mousedown", handleMouseDown);
            canvas?.removeEventListener("mousemove", handleMouseMove);
            canvas?.removeEventListener("mouseup", handleMouseUp);
            window.removeEventListener("resize", resizeCanvas);
        };
    });

    $: if (image.complete && ctx) {
        draw();
    }
</script>

<div class="bg-gray-900 rounded-xl overflow-hidden shadow-lg">
    <canvas bind:this={canvas} class="w-full" style="height: auto;"></canvas>
    {#if editable}
        <div class="p-3 bg-gray-800 text-center text-sm text-gray-400">
            Cliquez et faites glisser pour dessiner une zone d'intérêt
        </div>
    {/if}
</div>
