// backend/cpp_src/include/video_analyzer.hpp
#ifndef VIDEO_ANALYZER_HPP
#define VIDEO_ANALYZER_HPP

#include <vector>
#include <string>
#include <tuple>
#include <opencv2/opencv.hpp>

namespace farm_manager {

struct Detection {
    std::string class_name;
    float confidence;
    cv::Rect bbox;
    int track_id;
};

struct TrackedObject {
    int track_id;
    std::string class_name;
    std::vector<cv::Point2f> positions;
    int frames_seen;
    int first_seen_frame;
    int last_seen_frame;
    bool is_active;
};

struct AnalysisResult {
    int frame_number;
    double timestamp;
    std::vector<Detection> detections;
    std::vector<TrackedObject> tracked_objects;
    std::vector<std::string> anomalies;
    double processing_time_ms;
};

/**
 * Analyseur vidéo avec détection YOLO et tracking BoT-SORT
 */
class VideoAnalyzer {
public:
    VideoAnalyzer(const std::string& model_path, bool use_gpu, float conf_threshold, float iou_threshold);
    ~VideoAnalyzer();

    /**
     * Analyser une frame
     */
    AnalysisResult analyze_frame(const cv::Mat& frame, int frame_number, double timestamp);

    /**
     * Traiter un flux vidéo
     */
    std::vector<AnalysisResult> process_stream(
        const std::string& stream_url,
        int max_frames = -1
    );

    /**
     * Définir la région d'intérêt (ROI)
     */
    void set_roi(int x, int y, int width, int height);
    
    /**
     * Effacer la ROI
     */
    void clear_roi();

    /**
     * Obtenir les statistiques de performance
     */
    std::tuple<double, double, double> get_performance_stats() const;

    /**
     * Libérer les ressources
     */
    void release();

private:
    void initialize_yolo(const std::string& model_path, bool use_gpu);
    std::vector<Detection> detect(const cv::Mat& frame);
    void update_tracks(const std::vector<Detection>& detections, int frame_number);
    std::vector<TrackedObject> get_active_tracks() const;
    void detect_anomalies(AnalysisResult& result);
    float calculate_iou(const cv::Rect& box1, const cv::Rect& box2) const;
    
    cv::dnn::Net net_;
    cv::Mat blob_;
    std::vector<std::string> class_names_;
    
    std::vector<TrackedObject> tracks_;
    int next_track_id_;
    int max_track_age_;
    float iou_threshold_;
    float confidence_threshold_;
    
    cv::Rect roi_;
    bool use_roi_;
    
    double total_inference_time_;
    int inference_count_;
    bool use_gpu_;
};

} // namespace farm_manager

#endif // VIDEO_ANALYZER_HPP