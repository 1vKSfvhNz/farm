// backend/cpp_src/src/botsort_tracker.cpp
#include "../include/botsort_tracker.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace farm_manager {

// Hungarian algorithm helper
class HungarianAlgorithm {
public:
    static std::vector<int> solve(const cv::Mat& cost_matrix) {
        int n = cost_matrix.rows;
        int m = cost_matrix.cols;
        
        std::vector<int> assignment(n, -1);
        std::vector<double> u(n + 1, 0);
        std::vector<double> v(m + 1, 0);
        std::vector<int> p(m + 1, 0);
        std::vector<int> way(m + 1, 0);
        
        for (int i = 1; i <= n; i++) {
            p[0] = i;
            int j0 = 0;
            std::vector<double> minv(m + 1, std::numeric_limits<double>::max());
            std::vector<bool> used(m + 1, false);
            
            do {
                used[j0] = true;
                int i0 = p[j0];
                double delta = std::numeric_limits<double>::max();
                int j1 = 0;
                
                for (int j = 1; j <= m; j++) {
                    if (!used[j]) {
                        double cur = cost_matrix.at<double>(i0 - 1, j - 1) - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if (minv[j] < delta) {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }
                
                for (int j = 0; j <= m; j++) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }
                j0 = j1;
            } while (p[j0] != 0);
            
            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0);
        }
        
        for (int j = 1; j <= m; j++) {
            if (p[j] != 0) {
                assignment[p[j] - 1] = j - 1;
            }
        }
        
        return assignment;
    }
};

BoTSORTTracker::BoTSORTTracker()
    : max_missed_frames_(30)
    , min_hits_(3)
    , iou_threshold_(0.3f)
    , use_reid_(false)
    , next_track_id_(0)
    , total_tracks_created_(0)
    , total_tracks_lost_(0) {
    
    // Initialiser les matrices Kalman
    kalman_transition_matrix_ = (cv::Mat_<float>(8, 8) <<
        1, 0, 0, 0, 1, 0, 0, 0,
        0, 1, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1);
    
    kalman_measurement_matrix_ = (cv::Mat_<float>(4, 8) <<
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0);
}

BoTSORTTracker::~BoTSORTTracker() {}

void BoTSORTTracker::initialize(int max_missed_frames, int min_hits, float iou_threshold, bool use_reid) {
    max_missed_frames_ = max_missed_frames;
    min_hits_ = min_hits;
    iou_threshold_ = iou_threshold;
    use_reid_ = use_reid;
}

cv::KalmanFilter BoTSORTTracker::createKalmanFilter(const cv::Point2f& center) {
    cv::KalmanFilter kf(8, 4, 0);
    
    kf.transitionMatrix = kalman_transition_matrix_;
    kf.measurementMatrix = kalman_measurement_matrix_;
    
    // Matrice de covariance du processus (mouvement)
    setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    kf.processNoiseCov.at<float>(4, 4) = 1e-1;
    kf.processNoiseCov.at<float>(5, 5) = 1e-1;
    kf.processNoiseCov.at<float>(6, 6) = 1e-2;
    kf.processNoiseCov.at<float>(7, 7) = 1e-2;
    
    // Matrice de covariance de mesure
    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    
    // État initial
    kf.statePost = (cv::Mat_<float>(8, 1) << center.x, center.y, 0, 0, 0, 0, 0, 0);
    
    // Matrice de covariance d'erreur initiale
    setIdentity(kf.errorCovPost, cv::Scalar::all(1));
    
    return kf;
}

cv::Point2f BoTSORTTracker::predictTrack(TrackedObject& track) {
    cv::Mat prediction = track.kf.predict();
    return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
}

void BoTSORTTracker::updateKalmanFilter(TrackedObject& track, const cv::Point2f& measurement) {
    cv::Mat measurement_mat = (cv::Mat_<float>(4, 1) << measurement.x, measurement.y, 0, 0);
    track.kf.correct(measurement_mat);
    track.center = measurement;
}

float BoTSORTTracker::computeDistance(const TrackedObject& track, const YOLODetection& detection,
                                       const Eigen::VectorXf& det_features) {
    // Distance IOU
    cv::Rect predicted_bbox = track.bbox;
    cv::Rect det_bbox = detection.bbox;
    
    int x1 = std::max(predicted_bbox.x, det_bbox.x);
    int y1 = std::max(predicted_bbox.y, det_bbox.y);
    int x2 = std::min(predicted_bbox.x + predicted_bbox.width, det_bbox.x + det_bbox.width);
    int y2 = std::min(predicted_bbox.y + predicted_bbox.height, det_bbox.y + det_bbox.height);
    
    int intersection = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int area_pred = predicted_bbox.width * predicted_bbox.height;
    int area_det = det_bbox.width * det_bbox.height;
    float iou = static_cast<float>(intersection) / (area_pred + area_det - intersection + 1e-6);
    
    float distance = 1.0f - iou;
    
    // Ajouter distance ReID si disponible
    if (use_reid_ && reid_cache_.count(track.track_id)) {
        float reid_dist = 1.0f - cosineSimilarity(reid_cache_[track.track_id], det_features);
        distance = 0.7f * distance + 0.3f * reid_dist;
    }
    
    return distance;
}

cv::Mat BoTSORTTracker::computeCostMatrix(const std::vector<TrackedObject>& tracks,
                                           const std::vector<YOLODetection>& detections) {
    int n_tracks = tracks.size();
    int n_dets = detections.size();
    
    cv::Mat cost_matrix = cv::Mat::ones(n_tracks, n_dets, CV_32F) * 1000.0f;
    
    for (int i = 0; i < n_tracks; i++) {
        // Prédire la position de la piste
        cv::Point2f predicted_pos = predictTrack(const_cast<TrackedObject&>(tracks[i]));
        
        for (int j = 0; j < n_dets; j++) {
            cv::Point2f det_center(detections[j].bbox.x + detections[j].bbox.width / 2.0f,
                                    detections[j].bbox.y + detections[j].bbox.height / 2.0f);
            
            // Distance euclidienne
            float dist = cv::norm(predicted_pos - det_center);
            
            // Si la distance est raisonnable, utiliser l'IOU
            if (dist < 200.0f) {
                float iou = computeDistance(tracks[i], detections[j], Eigen::VectorXf());
                if (iou < 0.7f) {
                    cost_matrix.at<float>(i, j) = iou;
                }
            }
        }
    }
    
    return cost_matrix;
}

std::vector<std::pair<int, int>> BoTSORTTracker::associateTracks(const cv::Mat& cost_matrix) {
    std::vector<std::pair<int, int>> matches;
    
    int n_tracks = cost_matrix.rows;
    int n_dets = cost_matrix.cols;
    
    if (n_tracks == 0 || n_dets == 0) return matches;
    
    // Appliquer l'algorithme hongrois
    std::vector<int> assignment = HungarianAlgorithm::solve(cost_matrix);
    
    for (int i = 0; i < n_tracks; i++) {
        int j = assignment[i];
        if (j >= 0 && j < n_dets && cost_matrix.at<float>(i, j) < iou_threshold_) {
            matches.push_back(std::make_pair(i, j));
        }
    }
    
    return matches;
}

std::vector<TrackedObject> BoTSORTTracker::update(const std::vector<YOLODetection>& detections,
                                                   int frame_id,
                                                   const cv::Mat& frame) {
    std::vector<TrackedObject> active_tracks;
    
    // Prédire les positions des pistes existantes
    for (auto& track : tracks_) {
        cv::Point2f predicted = predictTrack(track);
        
        // Mettre à jour la boîte englobante prédite
        int width = track.bbox.width;
        int height = track.bbox.height;
        track.bbox = cv::Rect(static_cast<int>(predicted.x - width / 2),
                               static_cast<int>(predicted.y - height / 2),
                               width, height);
        
        track.missed_frames++;
        track.last_seen_frame = frame_id;
    }
    
    // Extraire les caractéristiques ReID si nécessaire
    std::vector<Eigen::VectorXf> det_features;
    if (use_reid_ && !frame.empty()) {
        for (const auto& det : detections) {
            det_features.push_back(extractReIDFeatures(frame, det));
        }
    }
    
    // Calculer la matrice de coût
    cv::Mat cost_matrix = computeCostMatrix(tracks_, detections);
    
    // Associer les pistes aux détections
    auto matches = associateTracks(cost_matrix);
    
    // Mettre à jour les pistes associées
    std::vector<bool> det_matched(detections.size(), false);
    std::vector<bool> track_matched(tracks_.size(), false);
    
    for (const auto& match : matches) {
        int track_idx = match.first;
        int det_idx = match.second;
        
        TrackedObject& track = tracks_[track_idx];
        const YOLODetection& det = detections[det_idx];
        
        // Mettre à jour la piste
        cv::Point2f center(det.bbox.x + det.bbox.width / 2.0f,
                           det.bbox.y + det.bbox.height / 2.0f);
        
        updateKalmanFilter(track, center);
        track.bbox = det.bbox;
        track.center = center;
        track.class_id = det.class_id;
        track.class_name = det.class_name;
        track.confidence = det.confidence;
        track.frames_seen++;
        track.missed_frames = 0;
        
        if (track.frames_seen >= min_hits_) {
            track.is_confirmed = true;
        }
        
        // Mettre à jour la trajectoire
        track.trajectory.push_back(center);
        if (track.trajectory.size() > 30) {
            track.trajectory.pop_front();
        }
        
        track_matched[track_idx] = true;
        det_matched[det_idx] = true;
        
        // Mettre à jour le cache ReID
        if (use_reid_ && det_idx < static_cast<int>(det_features.size())) {
            reid_cache_[track.track_id] = det_features[det_idx];
        }
        
        active_tracks.push_back(track);
    }
    
    // Créer de nouvelles pistes pour les détections non associées
    for (size_t i = 0; i < detections.size(); i++) {
        if (!det_matched[i]) {
            TrackedObject new_track;
            new_track.track_id = next_track_id_++;
            new_track.class_id = detections[i].class_id;
            new_track.class_name = detections[i].class_name;
            new_track.bbox = detections[i].bbox;
            new_track.center = cv::Point2f(detections[i].bbox.x + detections[i].bbox.width / 2.0f,
                                            detections[i].bbox.y + detections[i].bbox.height / 2.0f);
            new_track.kf = createKalmanFilter(new_track.center);
            new_track.first_seen_frame = frame_id;
            new_track.last_seen_frame = frame_id;
            new_track.frames_seen = 1;
            new_track.missed_frames = 0;
            new_track.is_confirmed = false;
            new_track.confidence = detections[i].confidence;
            new_track.trajectory.push_back(new_track.center);
            
            tracks_.push_back(new_track);
            total_tracks_created_++;
            active_tracks.push_back(new_track);
            
            if (use_reid_ && i < det_features.size()) {
                reid_cache_[new_track.track_id] = det_features[i];
            }
        }
    }
    
    // Supprimer les pistes perdues
    std::vector<TrackedObject> remaining_tracks;
    for (auto& track : tracks_) {
        if (track.missed_frames < max_missed_frames_) {
            remaining_tracks.push_back(track);
            if (track.is_confirmed) {
                active_tracks.push_back(track);
            }
        } else {
            total_tracks_lost_++;
        }
    }
    tracks_ = remaining_tracks;
    
    return active_tracks;
}

std::vector<TrackedObject> BoTSORTTracker::getActiveTracks() const {
    std::vector<TrackedObject> active;
    for (const auto& track : tracks_) {
        if (track.missed_frames < max_missed_frames_) {
            active.push_back(track);
        }
    }
    return active;
}

std::map<int, std::deque<cv::Point2f>> BoTSORTTracker::getTrajectories() const {
    std::map<int, std::deque<cv::Point2f>> trajectories;
    for (const auto& track : tracks_) {
        if (track.trajectory.size() > 1) {
            trajectories[track.track_id] = track.trajectory;
        }
    }
    return trajectories;
}

void BoTSORTTracker::reset() {
    tracks_.clear();
    reid_cache_.clear();
    next_track_id_ = 0;
    total_tracks_created_ = 0;
    total_tracks_lost_ = 0;
}

Eigen::VectorXf BoTSORTTracker::extractReIDFeatures(const cv::Mat& frame, const YOLODetection& detection) {
    // Extraire la région de la détection
    cv::Mat roi = frame(detection.bbox);
    cv::resize(roi, roi, cv::Size(128, 256));
    
    // Features simplifiées (histogramme de couleur)
    Eigen::VectorXf features(512);
    features.setZero();
    
    // Convertir en HSV et calculer histogramme
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    
    int h_bins = 16, s_bins = 16;
    int histSize[] = {h_bins, s_bins};
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    const float* ranges[] = {h_ranges, s_ranges};
    int channels[] = {0, 1};
    
    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    
    // Remplir le vecteur de features
    int idx = 0;
    for (int i = 0; i < h_bins; i++) {
        for (int j = 0; j < s_bins; j++) {
            if (idx < 512) {
                features(idx) = hist.at<float>(i, j);
                idx++;
            }
        }
    }
    
    return features;
}

float BoTSORTTracker::cosineSimilarity(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    float dot = a.dot(b);
    float norm_a = a.norm();
    float norm_b = b.norm();
    if (norm_a == 0 || norm_b == 0) return 0;
    return dot / (norm_a * norm_b);
}

} // namespace farm_manager