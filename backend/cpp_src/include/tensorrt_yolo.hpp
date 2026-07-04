// backend/cpp_src/include/tensorrt_yolo.hpp
#ifndef TENSORRT_YOLO_HPP
#define TENSORRT_YOLO_HPP

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

// Forward declarations for CUDA/TensorRT
namespace nvinfer1 {
    class IExecutionContext;
    class ICudaEngine;
    class IBuilder;
}

namespace farm_manager {

/**
 * Structure de détection YOLO
 */
struct YOLODetection {
    std::string class_name;
    float confidence;
    cv::Rect bbox;
    int class_id;
};

/**
 * Classe d'inférence YOLO avec TensorRT
 * Optimisation GPU pour haute performance
 */
class TensorRTYOLO {
public:
    TensorRTYOLO();
    ~TensorRTYOLO();

    /**
     * Initialiser le modèle à partir d'un fichier ONNX ou engine
     * @param model_path Chemin du modèle (.onnx ou .engine)
     * @param confidence_threshold Seuil de confiance
     * @param nms_threshold Seuil NMS
     * @param max_detections Nombre max de détections
     */
    bool initialize(const std::string& model_path, 
                    float confidence_threshold = 0.5f,
                    float nms_threshold = 0.45f,
                    int max_detections = 100);

    /**
     * Détecter des objets dans une image
     * @param frame Image d'entrée (BGR)
     * @return Liste des détections
     */
    std::vector<YOLODetection> detect(const cv::Mat& frame);

    /**
     * Détecter des objets dans une image avec tracking
     * @param frame Image d'entrée
     * @param frame_id ID de la frame
     * @return Liste des détections avec IDs de suivi
     */
    std::vector<YOLODetection> detectWithTracking(const cv::Mat& frame, int frame_id);

    /**
     * Obtenir les statistiques de performance
     */
    struct PerformanceStats {
        float inference_time_ms;
        float preprocess_time_ms;
        float postprocess_time_ms;
        float fps;
        int gpu_memory_mb;
    };
    PerformanceStats getStats() const;

    /**
     * Libérer les ressources GPU
     */
    void release();

    /**
     * Vérifier si le modèle est chargé
     */
    bool isLoaded() const { return is_loaded_; }

    /**
     * Définir la région d'intérêt (ROI) pour optimiser les calculs
     */
    void setROI(const cv::Rect& roi);
    void clearROI();

private:
    /**
     * Charger un modèle ONNX et construire l'engine TensorRT
     */
    bool buildEngineFromONNX(const std::string& onnx_path);

    /**
     * Charger un engine TensorRT précompilé
     */
    bool loadEngine(const std::string& engine_path);

    /**
     * Sauvegarder l'engine TensorRT
     */
    bool saveEngine(const std::string& engine_path);

    /**
     * Prétraiter l'image pour l'inférence
     */
    bool preprocess(const cv::Mat& frame, void* gpu_buffer);

    /**
     * Post-traiter les résultats de l'inférence
     */
    std::vector<YOLODetection> postprocess(float* output, int output_size, int img_width, int img_height);

    /**
     * Appliquer NMS (Non-Maximum Suppression)
     */
    void applyNMS(std::vector<YOLODetection>& detections, float iou_threshold);

    /**
     * Convertir classe ID en nom de classe
     */
    std::string getClassName(int class_id) const;

    // TensorRT members
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    void* cuda_buffer_[2];  // Input and output buffers
    size_t input_size_;
    size_t output_size_;
    int input_width_;
    int input_height_;
    int input_channels_;
    
    // CUDA streams
    void* cuda_stream_;
    
    // Parameters
    float confidence_threshold_;
    float nms_threshold_;
    int max_detections_;
    bool is_loaded_;
    bool use_gpu_;
    
    // ROI optimization
    cv::Rect roi_;
    bool use_roi_;
    
    // Performance tracking
    mutable PerformanceStats stats_;
    mutable std::chrono::steady_clock::time_point last_frame_time_;
    
    // Class names (COCO dataset)
    std::vector<std::string> class_names_;
    
    // Anchor boxes for YOLO
    std::vector<std::vector<float>> anchor_grids_;
    
    // Stride values
    std::vector<int> strides_;
};

} // namespace farm_manager

#endif // TENSORRT_YOLO_HPP