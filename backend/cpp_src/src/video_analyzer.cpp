// backend/cpp_src/src/video_analyzer.cpp
#include "../include/video_analyzer.hpp"
#include <chrono>

namespace farm_manager {

VideoAnalyzer::VideoAnalyzer(const std::string& model_path, bool use_gpu, float conf_threshold, float iou_threshold)
    : next_track_id_(0), max_track_age_(30), iou_threshold_(iou_threshold),
      confidence_threshold_(conf_threshold), use_roi_(false), total_inference_time_(0.0), inference_count_(0), use_gpu_(use_gpu) {
    initialize_yolo(model_path, use_gpu);
}

VideoAnalyzer::~VideoAnalyzer() {
    release();
}

void VideoAnalyzer::initialize_yolo(const std::string& model_path, bool use_gpu) {
    net_ = cv::dnn::readNetFromONNX(model_path);
    
    if (use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    
    // Noms des classes (simplifié)
    class_names_ = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
}

std::vector<Detection> VideoAnalyzer::detect(const cv::Mat& frame) {
    std::vector<Detection> detections;
    
    cv::Mat input_frame = frame.clone();
    if (use_roi_ && roi_.width > 0 && roi_.height > 0) {
        input_frame = frame(roi_);
    }
    
    // Prétraitement
    cv::Mat blob = cv::dnn::blobFromImage(input_frame, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    net_.setInput(blob);
    
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat outputs = net_.forward();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    total_inference_time_ += duration.count();
    inference_count_++;
    
    // Post-traitement
    float* data = (float*)outputs.data;
    int rows = outputs.size[2];
    int cols = outputs.size[3];
    
    for (int i = 0; i < rows; i++) {
        float* row = data + i * cols;
        float confidence = row[4];
        
        if (confidence >= confidence_threshold_) {
            float* classes_scores = row + 5;
            cv::Mat scores(1, class_names_.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            
            if (max_class_score > confidence_threshold_) {
                float x = row[0];
                float y = row[1];
                float w = row[2];
                float h = row[3];
                
                int left = int((x - w / 2) * input_frame.cols);
                int top = int((y - h / 2) * input_frame.rows);
                int width = int(w * input_frame.cols);
                int height = int(h * input_frame.rows);
                
                if (use_roi_) {
                    left += roi_.x;
                    top += roi_.y;
                }
                
                Detection det;
                det.class_name = class_names_[class_id.x];
                det.confidence = confidence;
                det.bbox = cv::Rect(left, top, width, height);
                det.track_id = -1;
                detections.push_back(det);
            }
        }
    }
    
    // NMS (Non-Maximum Suppression)
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    
    for (const auto& det : detections) {
        boxes.push_back(det.bbox);
        confidences.push_back(det.confidence);
    }
    
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, iou_threshold_, indices);
    
    std::vector<Detection> filtered_detections;
    for (int idx : indices) {
        filtered_detections.push_back(detections[idx]);
    }
    
    return filtered_detections;
}

float VideoAnalyzer::calculate_iou(const cv::Rect& box1, const cv::Rect& box2) const {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    int intersection = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int area1 = box1.width * box1.height;
    int area2 = box2.width * box2.height;
    int union_area = area1 + area2 - intersection;
    
    return (union_area > 0) ? (float)intersection / union_area : 0.0f;
}

void VideoAnalyzer::update_tracks(const std::vector<Detection>& detections, int frame_number) {
    // Mettre à jour les pistes existantes
    for (auto& track : tracks_) {
        track.is_active = false;
    }
    
    // Association par IOU
    for (const auto& det : detections) {
        int best_track_id = -1;
        float best_iou = 0.0f;
        
        for (auto& track : tracks_) {
            if (track.last_seen_frame == frame_number - 1) {
                cv::Rect last_bbox(track.positions.back().x, track.positions.back().y, 50, 50);
                float iou = calculate_iou(det.bbox, last_bbox);
                if (iou > best_iou && iou > iou_threshold_) {
                    best_iou = iou;
                    best_track_id = track.track_id;
                }
            }
        }
        
        if (best_track_id >= 0) {
            for (auto& track : tracks_) {
                if (track.track_id == best_track_id) {
                    track.positions.push_back(cv::Point2f(det.bbox.x + det.bbox.width / 2,
                                                           det.bbox.y + det.bbox.height / 2));
                    track.frames_seen++;
                    track.last_seen_frame = frame_number;
                    track.is_active = true;
                    break;
                }
            }
        } else {
            // Nouvelle piste
            TrackedObject new_track;
            new_track.track_id = next_track_id_++;
            new_track.class_name = det.class_name;
            new_track.positions.push_back(cv::Point2f(det.bbox.x + det.bbox.width / 2,
                                                       det.bbox.y + det.bbox.height / 2));
            new_track.frames_seen = 1;
            new_track.first_seen_frame = frame_number;
            new_track.last_seen_frame = frame_number;
            new_track.is_active = true;
            tracks_.push_back(new_track);
        }
    }
    
    // Supprimer les pistes inactives trop vieilles
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
        [this, frame_number](const TrackedObject& t) {
            return !t.is_active && (frame_number - t.last_seen_frame) > max_track_age_;
        }), tracks_.end());
}

std::vector<TrackedObject> VideoAnalyzer::get_active_tracks() const {
    std::vector<TrackedObject> active;
    for (const auto& track : tracks_) {
        if (track.is_active) {
            active.push_back(track);
        }
    }
    return active;
}

void VideoAnalyzer::detect_anomalies(AnalysisResult& result) {
    // Détection d'anomalies basées sur le comportement
    for (const auto& track : result.tracked_objects) {
        if (track.positions.size() > 10) {
            // Vérifier les mouvements erratiques
            float total_movement = 0.0f;
            for (size_t i = 1; i < track.positions.size(); i++) {
                total_movement += cv::norm(track.positions[i] - track.positions[i-1]);
            }
            float avg_movement = total_movement / (track.positions.size() - 1);
            
            if (avg_movement > 100.0f) {
                result.anomalies.push_back("Mouvement erratique détecté - ID: " + std::to_string(track.track_id));
            }
        }
    }
}

AnalysisResult VideoAnalyzer::analyze_frame(const cv::Mat& frame, int frame_number, double timestamp) {
    AnalysisResult result;
    result.frame_number = frame_number;
    result.timestamp = timestamp;
    result.processing_time_ms = 0.0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Détection
    result.detections = detect(frame);
    
    // Tracking
    update_tracks(result.detections, frame_number);
    result.tracked_objects = get_active_tracks();
    
    // Détection d'anomalies
    detect_anomalies(result);
    
    auto end = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    return result;
}

std::vector<AnalysisResult> VideoAnalyzer::process_stream(const std::string& stream_url, int max_frames) {
    std::vector<AnalysisResult> results;
    
    cv::VideoCapture cap(stream_url);
    if (!cap.isOpened()) {
        return results;
    }
    
    cv::Mat frame;
    int frame_count = 0;
    
    while (cap.read(frame) && (max_frames < 0 || frame_count < max_frames)) {
        double timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
        AnalysisResult result = analyze_frame(frame, frame_count++, timestamp);
        results.push_back(result);
    }
    
    cap.release();
    return results;
}

void VideoAnalyzer::set_roi(int x, int y, int width, int height) {
    roi_ = cv::Rect(x, y, width, height);
    use_roi_ = true;
}

void VideoAnalyzer::clear_roi() {
    use_roi_ = false;
}

std::tuple<double, double, double> VideoAnalyzer::get_performance_stats() const {
    double avg_inference_ms = (inference_count_ > 0) ? total_inference_time_ / inference_count_ : 0.0;
    double fps = (avg_inference_ms > 0) ? 1000.0 / avg_inference_ms : 0.0;
    double gpu_memory_mb = 0.0;
    
#ifdef USE_CUDA
    // Estimation mémoire GPU (simplifiée)
    gpu_memory_mb = 500.0;
#endif
    
    return std::make_tuple(fps, avg_inference_ms, gpu_memory_mb);
}

void VideoAnalyzer::release() {
    net_.~Net();
}

} // namespace farm_manager