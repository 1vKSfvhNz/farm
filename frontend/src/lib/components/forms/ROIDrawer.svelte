<!-- lib/components/forms/ROIDrawer.svelte -->
<script lang="ts">
    import { onMount } from "svelte";
    import { createEventDispatcher } from "svelte";
    import Button from "../ui/Button.svelte";

    export let imageUrl: string;
    export let width: number = 640;
    export const height: number = 480;
    export let existingROIs: Array<{
        id: number;
        name: string;
        x: number;
        y: number;
        width: number;
        height: number;
    }> = [];

    const dispatch = createEventDispatcher();

    let canvas: HTMLCanvasElement;
    let ctx: CanvasRenderingContext2D | null = null;
    let image = new Image();
    let isDrawing = false;
    let startX = 0;
    let startY = 0;
    let currentROI: {
        x: number;
        y: number;
        width: number;
        height: number;
    } | null = null;
    let rois = [...existingROIs];

    function draw() {
        if (!ctx || !canvas) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (image.complete && image.naturalWidth > 0) {
            ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        }

        // Dessiner les ROIs existants
        rois.forEach((roi, _index) => {
            ctx!.strokeStyle = "#3b82f6";
            ctx!.lineWidth = 2;
            ctx!.strokeRect(roi.x, roi.y, roi.width, roi.height);

            ctx!.fillStyle = "#3b82f640";
            ctx!.fillRect(roi.x, roi.y, roi.width, roi.height);

            ctx!.fillStyle = "#3b82f6";
            ctx!.font = "12px Arial";
            ctx!.fillText(roi.name, roi.x + 5, roi.y + 20);
        });

        // Dessiner le ROI en cours
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
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        startX = (e.clientX - rect.left) * scaleX;
        startY = (e.clientY - rect.top) * scaleY;
        isDrawing = true;
    }

    function handleMouseMove(e: MouseEvent) {
        if (!isDrawing) return;
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
        if (!isDrawing || !currentROI) return;
        if (currentROI.width > 10 && currentROI.height > 10) {
            const name = prompt(
                "Nom de la zone d'intérêt:",
                `ROI ${rois.length + 1}`,
            );
            if (name) {
                const newROI = {
                    id: Date.now(),
                    name,
                    x: currentROI.x,
                    y: currentROI.y,
                    width: currentROI.width,
                    height: currentROI.height,
                };
                rois.push(newROI);
                dispatch("add", newROI);
            }
        }
        isDrawing = false;
        currentROI = null;
        draw();
    }

    function removeROI(id: number) {
        rois = rois.filter((r) => r.id !== id);
        dispatch("remove", id);
        draw();
    }

    function saveROIs() {
        dispatch("save", rois);
    }

    function cancel() {
        dispatch("cancel");
    }

    function resizeCanvas() {
        if (!canvas || !image.complete) return;
        const container = canvas.parentElement;
        if (container) {
            const maxWidth = container.clientWidth - 32;
            const ratio = image.naturalWidth / image.naturalHeight;
            canvas.width = Math.min(maxWidth, image.naturalWidth, width);
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
</script>

<div class="space-y-4">
    <div class="bg-gray-900 rounded-xl overflow-hidden shadow-lg">
        <canvas bind:this={canvas} class="w-full" style="height: auto;"
        ></canvas>
    </div>

    <div class="flex flex-wrap gap-2">
        {#each rois as roi}
            <div
                class="inline-flex items-center gap-2 px-3 py-1.5 bg-blue-50 border border-blue-200 rounded-lg"
            >
                <span class="text-sm text-blue-800">{roi.name}</span>
                <button
                    on:click={() => removeROI(roi.id)}
                    class="text-blue-400 hover:text-red-500 transition-colors"
                >
                    <svg
                        class="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            stroke-width="2"
                            d="M6 18L18 6M6 6l12 12"
                        />
                    </svg>
                </button>
            </div>
        {/each}
    </div>

    <div class="flex justify-end gap-3 pt-4 border-t border-gray-200">
        <Button on:click={cancel} variant="outline">Annuler</Button>
        <Button on:click={saveROIs} variant="primary">
            Enregistrer les zones
        </Button>
    </div>
</div>
