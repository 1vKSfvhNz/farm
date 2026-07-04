// backend/cpp_src/cuda/preprocessing.cu
/**
 * CUDA kernels pour le prétraitement des images
 * Optimisation GPU pour YOLO et traitement vidéo
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Constantes
#define BLOCK_SIZE 16
#define TILE_SIZE 32

/**
 * Kernel: Conversion BGR vers RGB
 */
__global__ void bgr_to_rgb_kernel(const uint8_t* input, uint8_t* output, 
                                   int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        
        // BGR -> RGB
        output[idx] = input[idx + 2];     // R
        output[idx + 1] = input[idx + 1]; // G
        output[idx + 2] = input[idx];     // B
    }
}

/**
 * Kernel: Normalisation et conversion float
 */
__global__ void normalize_kernel(const uint8_t* input, float* output,
                                  int width, int height, int channels,
                                  float mean_r, float mean_g, float mean_b,
                                  float std_r, float std_g, float std_b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        
        float r = (input[idx] / 255.0f - mean_r) / std_r;
        float g = (input[idx + 1] / 255.0f - mean_g) / std_g;
        float b = (input[idx + 2] / 255.0f - mean_b) / std_b;
        
        output[idx] = r;
        output[idx + 1] = g;
        output[idx + 2] = b;
    }
}

/**
 * Kernel: Redimensionnement avec interpolation bilinéaire
 */
__global__ void resize_bilinear_kernel(const float* input, float* output,
                                        int src_width, int src_height,
                                        int dst_width, int dst_height,
                                        int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < dst_width && y < dst_height) {
        float src_x = (float)x / dst_width * src_width;
        float src_y = (float)y / dst_height * src_height;
        
        int x1 = (int)src_x;
        int y1 = (int)src_y;
        int x2 = min(x1 + 1, src_width - 1);
        int y2 = min(y1 + 1, src_height - 1);
        
        float dx = src_x - x1;
        float dy = src_y - y1;
        
        for (int c = 0; c < channels; c++) {
            float v11 = input[(y1 * src_width + x1) * channels + c];
            float v12 = input[(y2 * src_width + x1) * channels + c];
            float v21 = input[(y1 * src_width + x2) * channels + c];
            float v22 = input[(y2 * src_width + x2) * channels + c];
            
            float v1 = v11 * (1 - dx) + v21 * dx;
            float v2 = v12 * (1 - dx) + v22 * dx;
            
            output[(y * dst_width + x) * channels + c] = v1 * (1 - dy) + v2 * dy;
        }
    }
}

/**
 * Kernel: Conversion BGR vers format planaire (NCHW)
 */
__global__ void bgr_to_nchw_kernel(const uint8_t* input, float* output,
                                    int width, int height, int channels,
                                    float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x);
        
        for (int c = 0; c < channels; c++) {
            float val = input[(y * width + x) * channels + c] * scale;
            output[c * width * height + idx] = val;
        }
    }
}

/**
 * Kernel: Lettreboxing - ajout de bandes noires pour conserver le ratio
 */
__global__ void letterbox_kernel(const uint8_t* input, uint8_t* output,
                                  int src_width, int src_height,
                                  int dst_width, int dst_height,
                                  int pad_left, int pad_top,
                                  float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < dst_width && y < dst_height) {
        int src_x = (int)((x - pad_left) / scale);
        int src_y = (int)((y - pad_top) / scale);
        
        if (src_x >= 0 && src_x < src_width && src_y >= 0 && src_y < src_height) {
            int dst_idx = (y * dst_width + x) * 3;
            int src_idx = (src_y * src_width + src_x) * 3;
            
            output[dst_idx] = input[src_idx];
            output[dst_idx + 1] = input[src_idx + 1];
            output[dst_idx + 2] = input[src_idx + 2];
        } else {
            int dst_idx = (y * dst_width + x) * 3;
            output[dst_idx] = 0;
            output[dst_idx + 1] = 0;
            output[dst_idx + 2] = 0;
        }
    }
}

/**
 * Kernel: Seuillage pour NMS
 */
__global__ void threshold_kernel(float* boxes, float* scores, int* indices,
                                  int num_boxes, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_boxes) {
        if (scores[idx] < threshold) {
            indices[idx] = -1;
        }
    }
}

/**
 * Kernel: Calcul des scores IOU pour NMS
 */
__global__ void iou_kernel(float* boxes, float* ious, int num_boxes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < num_boxes && j < num_boxes && i != j) {
        float x1_i = boxes[i * 4];
        float y1_i = boxes[i * 4 + 1];
        float x2_i = boxes[i * 4 + 2];
        float y2_i = boxes[i * 4 + 3];
        
        float x1_j = boxes[j * 4];
        float y1_j = boxes[j * 4 + 1];
        float x2_j = boxes[j * 4 + 2];
        float y2_j = boxes[j * 4 + 3];
        
        float inter_x1 = max(x1_i, x1_j);
        float inter_y1 = max(y1_i, y1_j);
        float inter_x2 = min(x2_i, x2_j);
        float inter_y2 = min(y2_i, y2_j);
        
        float inter_area = max(0.0f, inter_x2 - inter_x1) * max(0.0f, inter_y2 - inter_y1);
        float area_i = (x2_i - x1_i) * (y2_i - y1_i);
        float area_j = (x2_j - x1_j) * (y2_j - y1_j);
        
        ious[i * num_boxes + j] = inter_area / (area_i + area_j - inter_area);
    }
}

/**
 * Fonction d'appel pour le prétraitement
 */
extern "C" void launch_preprocessing(const uint8_t* d_input, float* d_output,
                                      int width, int height, int channels,
                                      int target_width, int target_height,
                                      cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((target_width + block.x - 1) / block.x,
              (target_height + block.y - 1) / block.y);
    
    resize_bilinear_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height, target_width, target_height, channels);
}

/**
 * Fonction pour la conversion BGR vers RGB
 */
extern "C" void launch_bgr_to_rgb(const uint8_t* d_input, uint8_t* d_output,
                                   int width, int height,
                                   cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    bgr_to_rgb_kernel<<<grid, block, 0, stream>>>(d_input, d_output, width, height, 3);
}

/**
 * Fonction pour la normalisation
 */
extern "C" void launch_normalization(const uint8_t* d_input, float* d_output,
                                      int width, int height,
                                      float mean_r, float mean_g, float mean_b,
                                      float std_r, float std_g, float std_b,
                                      cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    normalize_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height, 3,
        mean_r, mean_g, mean_b, std_r, std_g, std_b);
}

/**
 * Fonction pour la conversion NCHW
 */
extern "C" void launch_bgr_to_nchw(const uint8_t* d_input, float* d_output,
                                    int width, int height, float scale,
                                    cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    bgr_to_nchw_kernel<<<grid, block, 0, stream>>>(d_input, d_output, width, height, 3, scale);
}