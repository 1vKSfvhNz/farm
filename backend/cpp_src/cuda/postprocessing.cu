// backend/cpp_src/cuda/postprocessing.cu
/**
 * CUDA kernels pour le post-traitement des détections
 * NMS, transformation de boîtes, etc.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 256
#define MAX_DETECTIONS 1000

/**
 * Structure pour une détection
 */
struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

/**
 * Kernel: Décodage des sorties YOLO
 */
__global__ void decode_yolo_output_kernel(const float* input, Detection* detections,
                                           int num_anchors, int num_classes,
                                           int grid_size, float stride,
                                           float conf_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_anchors * grid_size * grid_size;
    
    if (idx < total) {
        int anchor_idx = idx / (grid_size * grid_size);
        int remainder = idx % (grid_size * grid_size);
        int grid_y = remainder / grid_size;
        int grid_x = remainder % grid_size;
        
        int offset = idx * (5 + num_classes);
        
        float tx = input[offset];
        float ty = input[offset + 1];
        float tw = input[offset + 2];
        float th = input[offset + 3];
        float obj_conf = input[offset + 4];
        
        if (obj_conf > conf_threshold) {
            float cx = (grid_x + sigmoid(tx)) * stride;
            float cy = (grid_y + sigmoid(ty)) * stride;
            float w = exp(tw) * stride;
            float h = exp(th) * stride;
            
            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            float x2 = cx + w / 2;
            float y2 = cy + h / 2;
            
            // Trouver la meilleure classe
            float max_class_conf = 0;
            int best_class = 0;
            for (int c = 0; c < num_classes; c++) {
                float class_conf = input[offset + 5 + c];
                if (class_conf > max_class_conf) {
                    max_class_conf = class_conf;
                    best_class = c;
                }
            }
            
            float final_conf = obj_conf * max_class_conf;
            
            if (final_conf > conf_threshold) {
                Detection det;
                det.x1 = x1;
                det.y1 = y1;
                det.x2 = x2;
                det.y2 = y2;
                det.confidence = final_conf;
                det.class_id = best_class;
                detections[idx] = det;
            }
        }
    }
}

/**
 * Kernel: Filtrage des détections par confiance
 */
__global__ void filter_by_confidence_kernel(Detection* detections, int* valid_indices,
                                             int num_detections, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_detections) {
        if (detections[idx].confidence >= threshold) {
            valid_indices[idx] = idx;
        } else {
            valid_indices[idx] = -1;
        }
    }
}

/**
 * Kernel: Calcul des scores IOU entre paires de détections
 */
__global__ void compute_iou_matrix_kernel(const Detection* detections, float* iou_matrix,
                                          int num_detections) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < num_detections && j < num_detections && i < j) {
        const Detection& a = detections[i];
        const Detection& b = detections[j];
        
        float inter_x1 = max(a.x1, b.x1);
        float inter_y1 = max(a.y1, b.y1);
        float inter_x2 = min(a.x2, b.x2);
        float inter_y2 = min(a.y2, b.y2);
        
        float inter_area = max(0.0f, inter_x2 - inter_x1) * max(0.0f, inter_y2 - inter_y1);
        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        
        float iou = inter_area / (area_a + area_b - inter_area);
        
        iou_matrix[i * num_detections + j] = iou;
        iou_matrix[j * num_detections + i] = iou;
    }
}

/**
 * Kernel: NMS parallèle (par classe)
 */
__global__ void nms_kernel(const Detection* detections, bool* keep,
                           const float* iou_matrix, int num_detections,
                           float iou_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_detections) {
        keep[idx] = true;
        
        for (int j = 0; j < num_detections; j++) {
            if (j != idx && detections[j].confidence > detections[idx].confidence) {
                if (iou_matrix[idx * num_detections + j] > iou_threshold) {
                    if (detections[idx].class_id == detections[j].class_id) {
                        keep[idx] = false;
                        break;
                    }
                }
            }
        }
    }
}

/**
 * Kernel: Copie des détections filtrées
 */
__global__ void copy_filtered_detections_kernel(const Detection* src, Detection* dst,
                                                 const bool* keep, int num_detections) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_detections && keep[idx]) {
        int write_idx = atomicAdd(&dst[0].confidence, 1);
        dst[write_idx + 1] = src[idx];
    }
}

/**
 * Fonction: Tri des détections par confiance (CUDA thrust)
 */
struct CompareByConfidence {
    __host__ __device__ bool operator()(const Detection& a, const Detection& b) const {
        return a.confidence > b.confidence;
    }
};

/**
 * Fonction principale de post-traitement
 */
extern "C" int launch_postprocessing(const float* d_output, Detection* d_detections,
                                      int num_detections, float conf_threshold,
                                      float iou_threshold, cudaStream_t stream) {
    // Compter les détections valides
    thrust::device_vector<Detection> detections_vec(d_detections, d_detections + num_detections);
    
    // Trier par confiance décroissante
    thrust::sort(thrust::device, detections_vec.begin(), detections_vec.end(),
                 CompareByConfidence());
    
    // Copier vers le device
    thrust::copy(detections_vec.begin(), detections_vec.end(), d_detections);
    
    // Calculer la matrice IOU
    int grid_size = (num_detections + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(grid_size, grid_size);
    
    float* d_iou_matrix;
    cudaMalloc(&d_iou_matrix, num_detections * num_detections * sizeof(float));
    
    compute_iou_matrix_kernel<<<grid, block, 0, stream>>>(
        d_detections, d_iou_matrix, num_detections);
    
    // NMS
    bool* d_keep;
    cudaMalloc(&d_keep, num_detections * sizeof(bool));
    
    nms_kernel<<<(num_detections + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        d_detections, d_keep, d_iou_matrix, num_detections, iou_threshold);
    
    // Compter les détections conservées
    thrust::device_vector<bool> keep_vec(d_keep, d_keep + num_detections);
    int final_count = thrust::count(thrust::device, keep_vec.begin(), keep_vec.end(), true);
    
    cudaFree(d_iou_matrix);
    cudaFree(d_keep);
    
    return final_count;
}

/**
 * Fonction: Transformation des boîtes de l'espace image à l'espace original
 */
__global__ void transform_boxes_kernel(Detection* detections, int num_detections,
                                        float scale_x, float scale_y,
                                        int pad_left, int pad_top) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_detections) {
        Detection& det = detections[idx];
        
        // Ajuster les coordonnées
        det.x1 = (det.x1 - pad_left) / scale_x;
        det.y1 = (det.y1 - pad_top) / scale_y;
        det.x2 = (det.x2 - pad_left) / scale_x;
        det.y2 = (det.y2 - pad_top) / scale_y;
        
        // Clipper aux limites
        det.x1 = max(0.0f, det.x1);
        det.y1 = max(0.0f, det.y1);
    }
}

/**
 * Fonction: Fusion des détections de multiples échelles (FPN)
 */
__global__ void fuse_detections_kernel(const Detection* detections_scale1,
                                         const Detection* detections_scale2,
                                         Detection* output,
                                         int num1, int num2,
                                         float iou_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num1) {
        Detection fused = detections_scale1[idx];
        
        for (int j = 0; j < num2; j++) {
            const Detection& det2 = detections_scale2[j];
            
            if (det2.class_id == fused.class_id) {
                float inter_x1 = max(fused.x1, det2.x1);
                float inter_y1 = max(fused.y1, det2.y1);
                float inter_x2 = min(fused.x2, det2.x2);
                float inter_y2 = min(fused.y2, det2.y2);
                
                float inter_area = max(0.0f, inter_x2 - inter_x1) * max(0.0f, inter_y2 - inter_y1);
                float area_fused = (fused.x2 - fused.x1) * (fused.y2 - fused.y1);
                float area_det2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1);
                float iou = inter_area / (area_fused + area_det2 - inter_area);
                
                if (iou > iou_threshold) {
                    // Fusion par moyenne pondérée par la confiance
                    float total_conf = fused.confidence + det2.confidence;
                    fused.x1 = (fused.x1 * fused.confidence + det2.x1 * det2.confidence) / total_conf;
                    fused.y1 = (fused.y1 * fused.confidence + det2.y1 * det2.confidence) / total_conf;
                    fused.x2 = (fused.x2 * fused.confidence + det2.x2 * det2.confidence) / total_conf;
                    fused.y2 = (fused.y2 * fused.confidence + det2.y2 * det2.confidence) / total_conf;
                    fused.confidence = total_conf / 2;
                }
            }
        }
        
        output[idx] = fused;
    }
}

/**
 * Fonction sigmoïde rapide pour GPU
 */
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}