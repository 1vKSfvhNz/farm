// backend/cpp_src/src/tensorrt_yolo.cpp
#include "../include/tensorrt_yolo.hpp"
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

namespace farm_manager {

// Logger pour TensorRT
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

static TRTLogger gLogger;

TensorRTYOLO::TensorRTYOLO()
    : engine_(nullptr)
    , context_(nullptr)
    , cuda_buffer_{nullptr, nullptr}
    , input_size_(0)
    , output_size_(0)
    , input_width_(640)
    , input_height_(640)
    , input_channels_(3)
    , cuda_stream_(nullptr)
    , confidence_threshold_(0.5f)
    , nms_threshold_(0.45f)
    , max_detections_(100)
    , is_loaded_(false)
    , use_gpu_(true)
    , use_roi_(false)
    , strides_{8, 16, 32} {
    
    // Initialisation des classes COCO (simplifié)
    class_names_ = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    
    cudaStreamCreate(&cuda_stream_);
    stats_ = {0.0f, 0.0f, 0.0f, 0.0f, 0};
}

TensorRTYOLO::~TensorRTYOLO() {
    release();
}

bool TensorRTYOLO::initialize(const std::string& model_path, 
                              float confidence_threshold,
                              float nms_threshold,
                              int max_detections) {
    confidence_threshold_ = confidence_threshold;
    nms_threshold_ = nms_threshold;
    max_detections_ = max_detections;
    
    // Vérifier l'extension du fichier
    std::string ext = model_path.substr(model_path.find_last_of(".") + 1);
    
    bool success = false;
    if (ext == "onnx") {
        success = buildEngineFromONNX(model_path);
    } else if (ext == "engine") {
        success = loadEngine(model_path);
    } else {
        std::cerr << "Format de modèle non supporté: " << ext << std::endl;
        return false;
    }
    
    if (!success) return false;
    
    is_loaded_ = true;
    return true;
}

bool TensorRTYOLO::buildEngineFromONNX(const std::string& onnx_path) {
    // Créer le builder TensorRT
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) return false;
    
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) return false;
    
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) return false;
    
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Erreur lors du parsing du fichier ONNX" << std::endl;
        return false;
    }
    
    // Configurer le builder
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return false;
    
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB
    
    // Optimiser pour la performance
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    
    // Construire l'engine
    engine_.reset(builder->buildSerializedNetwork(*network, *config));
    if (!engine_) return false;
    
    // Créer le contexte d'exécution
    context_.reset(engine_->createExecutionContext());
    if (!context_) return false;
    
    // Obtenir les dimensions d'entrée
    auto input_dims = network->getInput(0)->getDimensions();
    input_channels_ = input_dims.d[1];
    input_height_ = input_dims.d[2];
    input_width_ = input_dims.d[3];
    
    // Calculer les tailles des buffers
    input_size_ = input_channels_ * input_height_ * input_width_ * sizeof(float);
    
    auto output_dims = network->getOutput(0)->getDimensions();
    output_size_ = output_dims.d[1] * output_dims.d[2] * sizeof(float);
    
    // Allouer les buffers GPU
    cudaMalloc(&cuda_buffer_[0], input_size_);
    cudaMalloc(&cuda_buffer_[1], output_size_);
    
    return true;
}

bool TensorRTYOLO::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();
    
    // Désérialiser l'engine
    engine_.reset(nvinfer1::deserializeCudaEngine(engine_data.data(), size, nullptr));
    if (!engine_) return false;
    
    context_.reset(engine_->createExecutionContext());
    if (!context_) return false;
    
    // Allouer les buffers GPU
    cudaMalloc(&cuda_buffer_[0], input_size_);
    cudaMalloc(&cuda_buffer_[1], output_size_);
    
    return true;
}

bool TensorRTYOLO::preprocess(const cv::Mat& frame, void* gpu_buffer) {
    cv::Mat input_img;
    
    if (use_roi_ && roi_.width > 0 && roi_.height > 0) {
        input_img = frame(roi_);
    } else {
        input_img = frame;
    }
    
    // Redimensionner et normaliser
    cv::Mat resized_img;
    cv::resize(input_img, resized_img, cv::Size(input_width_, input_height_));
    
    // Convertir BGR -> RGB et en float
    cv::Mat float_img;
    resized_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);
    
    // Copier vers GPU
    cudaMemcpyAsync(gpu_buffer, float_img.data, input_size_, cudaMemcpyHostToDevice, cuda_stream_);
    
    return true;
}

std::vector<YOLODetection> TensorRTYOLO::postprocess(float* output, int output_size, int img_width, int img_height) {
    std::vector<YOLODetection> detections;
    
    // Format de sortie YOLO: [batch, num_detections, 6]
    // Chaque détection: [x1, y1, x2, y2, confidence, class_id]
    int num_detections = output_size / 6;
    
    for (int i = 0; i < num_detections && detections.size() < max_detections_; i++) {
        float* det = output + i * 6;
        float confidence = det[4];
        
        if (confidence < confidence_threshold_) continue;
        
        int class_id = static_cast<int>(det[5]);
        float x1 = det[0] * img_width;
        float y1 = det[1] * img_height;
        float x2 = det[2] * img_width;
        float y2 = det[3] * img_height;
        
        YOLODetection detection;
        detection.class_id = class_id;
        detection.class_name = getClassName(class_id);
        detection.confidence = confidence;
        detection.bbox = cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                   static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
        
        detections.push_back(detection);
    }
    
    // Appliquer NMS
    applyNMS(detections, nms_threshold_);
    
    return detections;
}

void TensorRTYOLO::applyNMS(std::vector<YOLODetection>& detections, float iou_threshold) {
    if (detections.empty()) return;
    
    // Trier par confiance décroissante
    std::sort(detections.begin(), detections.end(),
              [](const YOLODetection& a, const YOLODetection& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<bool> keep(detections.size(), true);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (!keep[i]) continue;
        
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (!keep[j]) continue;
            
            // Calculer l'intersection
            int x1 = std::max(detections[i].bbox.x, detections[j].bbox.x);
            int y1 = std::max(detections[i].bbox.y, detections[j].bbox.y);
            int x2 = std::min(detections[i].bbox.x + detections[i].bbox.width, 
                             detections[j].bbox.x + detections[j].bbox.width);
            int y2 = std::min(detections[i].bbox.y + detections[i].bbox.height,
                             detections[j].bbox.y + detections[j].bbox.height);
            
            int intersection = std::max(0, x2 - x1) * std::max(0, y2 - y1);
            int area_i = detections[i].bbox.width * detections[i].bbox.height;
            int area_j = detections[j].bbox.width * detections[j].bbox.height;
            float iou = static_cast<float>(intersection) / (area_i + area_j - intersection);
            
            if (iou > iou_threshold) {
                keep[j] = false;
            }
        }
    }
    
    // Filtrer les détections gardées
    std::vector<YOLODetection> filtered;
    for (size_t i = 0; i < detections.size(); i++) {
        if (keep[i]) {
            filtered.push_back(detections[i]);
        }
    }
    detections = filtered;
}

std::vector<YOLODetection> TensorRTYOLO::detect(const cv::Mat& frame) {
    if (!is_loaded_) return {};
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Prétraitement
    auto pre_start = std::chrono::high_resolution_clock::now();
    bool success = preprocess(frame, cuda_buffer_[0]);
    if (!success) return {};
    auto pre_end = std::chrono::high_resolution_clock::now();
    stats_.preprocess_time_ms = std::chrono::duration<float, std::milli>(pre_end - pre_start).count();
    
    // Inférence
    auto infer_start = std::chrono::high_resolution_clock::now();
    context_->enqueueV2(cuda_buffer_, cuda_stream_, nullptr);
    auto infer_end = std::chrono::high_resolution_clock::now();
    stats_.inference_time_ms = std::chrono::duration<float, std::milli>(infer_end - infer_start).count();
    
    // Copier les résultats vers CPU
    std::vector<float> output_data(output_size_ / sizeof(float));
    cudaMemcpyAsync(output_data.data(), cuda_buffer_[1], output_size_, cudaMemcpyDeviceToHost, cuda_stream_);
    cudaStreamSynchronize(cuda_stream_);
    
    // Post-traitement
    auto post_start = std::chrono::high_resolution_clock::now();
    auto detections = postprocess(output_data.data(), output_size_ / sizeof(float), 
                                   frame.cols, frame.rows);
    auto post_end = std::chrono::high_resolution_clock::now();
    stats_.postprocess_time_ms = std::chrono::duration<float, std::milli>(post_end - post_start).count();
    
    auto end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(end - start).count();
    stats_.fps = 1000.0f / total_time;
    
    return detections;
}

std::string TensorRTYOLO::getClassName(int class_id) const {
    if (class_id >= 0 && class_id < static_cast<int>(class_names_.size())) {
        return class_names_[class_id];
    }
    return "unknown";
}

void TensorRTYOLO::setROI(const cv::Rect& roi) {
    roi_ = roi;
    use_roi_ = true;
}

void TensorRTYOLO::clearROI() {
    use_roi_ = false;
}

TensorRTYOLO::PerformanceStats TensorRTYOLO::getStats() const {
    return stats_;
}

void TensorRTYOLO::release() {
    if (cuda_buffer_[0]) {
        cudaFree(cuda_buffer_[0]);
        cuda_buffer_[0] = nullptr;
    }
    if (cuda_buffer_[1]) {
        cudaFree(cuda_buffer_[1]);
        cuda_buffer_[1] = nullptr;
    }
    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }
    context_.reset();
    engine_.reset();
    is_loaded_ = false;
}

} // namespace farm_manager