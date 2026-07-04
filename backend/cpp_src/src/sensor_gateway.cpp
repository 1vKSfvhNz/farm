// backend/cpp_src/src/sensor_gateway.cpp
#include "../include/sensor_gateway.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <mutex>

namespace farm_manager {

SensorProcessor::SensorProcessor(int window_size_seconds)
    : window_size_seconds_(window_size_seconds) {}

SensorProcessor::~SensorProcessor() {}

bool SensorProcessor::add_data_point(const std::string& sensor_id, double timestamp, double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    SensorDataPoint point;
    point.timestamp = timestamp;
    point.value = value;
    point.quality = 1.0;
    
    data_[sensor_id].push_back(point);
    
    // Nettoyer les données trop anciennes
    double cutoff = std::chrono::system_clock::now().time_since_epoch().count() / 1e9 - window_size_seconds_;
    
    auto& vec = data_[sensor_id];
    vec.erase(std::remove_if(vec.begin(), vec.end(),
        [cutoff](const SensorDataPoint& p) { return p.timestamp < cutoff; }), vec.end());
    
    return true;
}

int SensorProcessor::add_batch(const std::string& sensor_id, const std::vector<double>& timestamps, const std::vector<double>& values) {
    if (timestamps.size() != values.size()) return 0;
    
    int added = 0;
    for (size_t i = 0; i < timestamps.size(); i++) {
        if (add_data_point(sensor_id, timestamps[i], values[i])) {
            added++;
        }
    }
    return added;
}

std::vector<SensorDataPoint> SensorProcessor::get_window_data(const std::string& sensor_id, double start_time, double end_time) {
    std::vector<SensorDataPoint> result;
    
    auto it = data_.find(sensor_id);
    if (it == data_.end()) return result;
    
    for (const auto& point : it->second) {
        if (point.timestamp >= start_time && point.timestamp <= end_time) {
            result.push_back(point);
        }
    }
    
    return result;
}

double SensorProcessor::calculate_median(std::vector<double>& values) {
    if (values.empty()) return 0.0;
    
    size_t n = values.size();
    std::sort(values.begin(), values.end());
    
    if (n % 2 == 0) {
        return (values[n/2 - 1] + values[n/2]) / 2.0;
    } else {
        return values[n/2];
    }
}

double SensorProcessor::calculate_std_dev(const std::vector<double>& values, double mean) {
    if (values.size() < 2) return 0.0;
    
    double sum_sq = 0.0;
    for (double v : values) {
        sum_sq += (v - mean) * (v - mean);
    }
    return std::sqrt(sum_sq / (values.size() - 1));
}

std::vector<AggregatedData> SensorProcessor::aggregate(
    const std::string& sensor_id,
    double start_time,
    double end_time,
    int interval_seconds
) {
    std::vector<AggregatedData> result;
    
    auto data = get_window_data(sensor_id, start_time, end_time);
    if (data.empty()) return result;
    
    int num_intervals = std::ceil((end_time - start_time) / interval_seconds);
    
    for (int i = 0; i < num_intervals; i++) {
        double interval_start = start_time + i * interval_seconds;
        double interval_end = interval_start + interval_seconds;
        
        std::vector<double> values;
        for (const auto& point : data) {
            if (point.timestamp >= interval_start && point.timestamp < interval_end) {
                values.push_back(point.value);
            }
        }
        
        if (!values.empty()) {
            AggregatedData agg;
            agg.period_start = interval_start;
            agg.period_end = interval_end;
            agg.count = values.size();
            agg.min_value = *std::min_element(values.begin(), values.end());
            agg.max_value = *std::max_element(values.begin(), values.end());
            agg.mean_value = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            agg.median_value = calculate_median(values);
            agg.std_dev = calculate_std_dev(values, agg.mean_value);
            agg.sum_value = std::accumulate(values.begin(), values.end(), 0.0);
            result.push_back(agg);
        }
    }
    
    return result;
}

double SensorProcessor::perpendicular_distance(const SensorDataPoint& p1, const SensorDataPoint& p2, const SensorDataPoint& p3) {
    if (p1.timestamp == p2.timestamp) return std::abs(p3.value - p1.value);
    
    double dx = p2.timestamp - p1.timestamp;
    double dy = p2.value - p1.value;
    double area = std::abs(dx * (p1.value - p3.value) - (p1.timestamp - p3.timestamp) * dy);
    double distance = area / std::sqrt(dx * dx + dy * dy);
    
    return distance;
}

void SensorProcessor::douglas_peucker(const std::vector<SensorDataPoint>& points, int start, int end, double tolerance, std::vector<bool>& keep) {
    if (end <= start + 1) return;
    
    double max_dist = 0.0;
    int max_index = start;
    
    for (int i = start + 1; i < end; i++) {
        double dist = perpendicular_distance(points[start], points[end], points[i]);
        if (dist > max_dist) {
            max_dist = dist;
            max_index = i;
        }
    }
    
    if (max_dist > tolerance) {
        keep[max_index] = true;
        douglas_peucker(points, start, max_index, tolerance, keep);
        douglas_peucker(points, max_index, end, tolerance, keep);
    }
}

std::vector<SensorDataPoint> SensorProcessor::compress_data(
    const std::string& sensor_id,
    double start_time,
    double end_time,
    double tolerance
) {
    auto data = get_window_data(sensor_id, start_time, end_time);
    if (data.size() < 3) return data;
    
    std::vector<bool> keep(data.size(), false);
    keep[0] = true;
    keep[data.size() - 1] = true;
    
    douglas_peucker(data, 0, data.size() - 1, tolerance, keep);
    
    std::vector<SensorDataPoint> compressed;
    for (size_t i = 0; i < data.size(); i++) {
        if (keep[i]) {
            compressed.push_back(data[i]);
        }
    }
    
    return compressed;
}

std::vector<std::tuple<double, double, double, double, double>> SensorProcessor::detect_anomalies(
    const std::string& sensor_id,
    double start_time,
    double end_time,
    double std_dev_threshold
) {
    std::vector<std::tuple<double, double, double, double, double>> anomalies;
    
    auto data = get_window_data(sensor_id, start_time, end_time);
    if (data.size() < 10) return anomalies;
    
    std::vector<double> values;
    for (const auto& p : data) {
        values.push_back(p.value);
    }
    
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double std_dev = calculate_std_dev(values, mean);
    
    for (const auto& p : data) {
        double z_score = std::abs(p.value - mean) / (std_dev + 1e-6);
        if (z_score > std_dev_threshold) {
            double anomaly_score = std::min(z_score / std_dev_threshold, 5.0);
            anomalies.push_back(std::make_tuple(p.timestamp, p.value, mean, std_dev, anomaly_score));
        }
    }
    
    return anomalies;
}

std::tuple<int, double, double, double, double, double, double> SensorProcessor::get_stats(const std::string& sensor_id) {
    auto it = data_.find(sensor_id);
    if (it == data_.end() || it->second.empty()) {
        return std::make_tuple(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    
    const auto& vec = it->second;
    std::vector<double> values;
    for (const auto& p : vec) {
        values.push_back(p.value);
    }
    
    double min_val = *std::min_element(values.begin(), values.end());
    double max_val = *std::max_element(values.begin(), values.end());
    double mean_val = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double std_dev = calculate_std_dev(values, mean_val);
    
    double first_ts = vec.front().timestamp;
    double last_ts = vec.back().timestamp;
    
    return std::make_tuple(values.size(), first_ts, last_ts, min_val, max_val, mean_val, std_dev);
}

bool SensorProcessor::clear_sensor_data(const std::string& sensor_id) {
    auto it = data_.find(sensor_id);
    if (it != data_.end()) {
        it->second.clear();
        return true;
    }
    return false;
}

} // namespace farm_manager