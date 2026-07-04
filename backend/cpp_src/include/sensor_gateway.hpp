// backend/cpp_src/include/sensor_gateway.hpp
#ifndef SENSOR_GATEWAY_HPP
#define SENSOR_GATEWAY_HPP

#include <vector>
#include <string>
#include <map>
#include <chrono>

namespace farm_manager {

struct SensorDataPoint {
    double timestamp;
    double value;
    float quality;
};

struct AggregatedData {
    double period_start;
    double period_end;
    int count;
    double min_value;
    double max_value;
    double mean_value;
    double median_value;
    double std_dev;
    double sum_value;
};

/**
 * Processeur de données capteurs avec compression et filtrage
 */
class SensorProcessor {
public:
    SensorProcessor(int window_size_seconds = 60);
    ~SensorProcessor();

    /**
     * Ajouter un point de donnée
     */
    bool add_data_point(const std::string& sensor_id, double timestamp, double value);

    /**
     * Ajouter un lot de données
     */
    int add_batch(const std::string& sensor_id, const std::vector<double>& timestamps, const std::vector<double>& values);

    /**
     * Récupérer les données dans une fenêtre temporelle
     */
    std::vector<SensorDataPoint> get_window_data(const std::string& sensor_id, double start_time, double end_time);

    /**
     * Agréger les données par intervalle
     */
    std::vector<AggregatedData> aggregate(
        const std::string& sensor_id,
        double start_time,
        double end_time,
        int interval_seconds = 3600
    );

    /**
     * Compresser les données avec algorithme Douglas-Peucker
     */
    std::vector<SensorDataPoint> compress_data(
        const std::string& sensor_id,
        double start_time,
        double end_time,
        double tolerance = 0.01
    );

    /**
     * Détecter les anomalies statistiques
     */
    std::vector<std::tuple<double, double, double, double, double>> detect_anomalies(
        const std::string& sensor_id,
        double start_time,
        double end_time,
        double std_dev_threshold = 3.0
    );

    /**
     * Obtenir les statistiques pour un capteur
     */
    std::tuple<int, double, double, double, double, double, double> get_stats(const std::string& sensor_id);

    /**
     * Effacer toutes les données d'un capteur
     */
    bool clear_sensor_data(const std::string& sensor_id);

private:
    double calculate_median(std::vector<double>& values);
    double calculate_std_dev(const std::vector<double>& values, double mean);
    double perpendicular_distance(const SensorDataPoint& p1, const SensorDataPoint& p2, const SensorDataPoint& p3);
    void douglas_peucker(const std::vector<SensorDataPoint>& points, int start, int end, double tolerance, std::vector<bool>& keep);
    
    std::map<std::string, std::vector<SensorDataPoint>> data_;
    int window_size_seconds_;
};

} // namespace farm_manager

#endif // SENSOR_GATEWAY_HPP