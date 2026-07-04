// backend/cpp_src/include/botsort_tracker.hpp
#ifndef BOTSORT_TRACKER_HPP
#define BOTSORT_TRACKER_HPP

#include <vector>
#include <map>
#include <deque>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace farm_manager {

/**
 * Structure d'un objet suivi
 */
struct TrackedObject {
    int track_id;
    int class_id;
    std::string class_name;
    cv::Rect bbox;
    cv::Point2f center;
    cv::KalmanFilter kf;
    Eigen::VectorXf state;
    int first_seen_frame;
    int last_seen_frame;
    int frames_seen;
    int missed_frames;
    bool is_confirmed;
    float confidence;
    std::deque<cv::Point2f> trajectory;
    std::deque<float> velocities;
};

/**
 * Tracker BoT-SORT pour suivi d'objets
 * Basé sur Kalman Filter + IOU + ReID
 */
class BoTSORTTracker {
public:
    BoTSORTTracker();
    ~BoTSORTTracker();

    /**
     * Initialiser le tracker
     * @param max_missed_frames Nombre max de frames sans détection avant suppression
     * @param min_hits Minimum de détections pour confirmer une piste
     * @param iou_threshold Seuil IOU pour association
     * @param use_reid Utiliser ReID pour ré-association
     */
    void initialize(int max_missed_frames = 30,
                    int min_hits = 3,
                    float iou_threshold = 0.3f,
                    bool use_reid = false);

    /**
     * Mettre à jour les pistes avec les nouvelles détections
     * @param detections Détections de la frame courante
     * @param frame_id ID de la frame
     * @param frame Image originale (pour ReID)
     * @return Liste des pistes actives
     */
    std::vector<TrackedObject> update(const std::vector<YOLODetection>& detections,
                                       int frame_id,
                                       const cv::Mat& frame = cv::Mat());

    /**
     * Obtenir toutes les pistes actives
     */
    std::vector<TrackedObject> getActiveTracks() const;

    /**
     * Obtenir l'historique des trajectoires pour visualisation
     */
    std::map<int, std::deque<cv::Point2f>> getTrajectories() const;

    /**
     * Réinitialiser le tracker
     */
    void reset();

    /**
     * Configurer les paramètres
     */
    void setMaxMissedFrames(int value) { max_missed_frames_ = value; }
    void setIouThreshold(float value) { iou_threshold_ = value; }

private:
    /**
     * Créer un nouveau Kalman Filter pour une piste
     */
    cv::KalmanFilter createKalmanFilter(const cv::Point2f& center);

    /**
     * Prédire la position d'une piste
     */
    cv::Point2f predictTrack(TrackedObject& track);

    /**
     * Mettre à jour le Kalman Filter avec une détection
     */
    void updateKalmanFilter(TrackedObject& track, const cv::Point2f& measurement);

    /**
     * Calculer la matrice de coût entre pistes et détections
     */
    cv::Mat computeCostMatrix(const std::vector<TrackedObject>& tracks,
                               const std::vector<YOLODetection>& detections);

    /**
     * Associer pistes et détections (Hungarian algorithm)
     */
    std::vector<std::pair<int, int>> associateTracks(const cv::Mat& cost_matrix);

    /**
     * Extraire les caractéristiques ReID d'une détection
     */
    Eigen::VectorXf extractReIDFeatures(const cv::Mat& frame, const YOLODetection& detection);

    /**
     * Calculer la similarité cosinus entre deux vecteurs
     */
    float cosineSimilarity(const Eigen::VectorXf& a, const Eigen::VectorXf& b);

    /**
     * Calculer la distance entre une piste et une détection
     */
    float computeDistance(const TrackedObject& track, const YOLODetection& detection,
                          const Eigen::VectorXf& det_features);

    // Paramètres
    int max_missed_frames_;
    int min_hits_;
    float iou_threshold_;
    bool use_reid_;
    int next_track_id_;
    
    // Pistes actives
    std::vector<TrackedObject> tracks_;
    
    // Cache ReID
    std::map<int, Eigen::VectorXf> reid_cache_;
    
    // Statistiques
    int total_tracks_created_;
    int total_tracks_lost_;
    
    // Matrice de covariance pour Kalman
    cv::Mat kalman_transition_matrix_;
    cv::Mat kalman_measurement_matrix_;
    cv::Mat kalman_process_noise_;
    cv::Mat kalman_measurement_noise_;
};

} // namespace farm_manager

#endif // BOTSORT_TRACKER_HPP