// backend/cpp_src/src/main.cpp
/**
 * Programme de test pour les modules C++
 * Compilation: g++ -std=c++17 main.cpp -o test_cpp
 */

#include <iostream>
#include <vector>
#include "../include/growth_predictor.hpp"
#include "../include/sensor_gateway.hpp"

int main() {
    std::cout << "=== Farm Manager C++ Modules Test ===" << std::endl;
    
    // Test du prédicteur de croissance
    std::cout << "\n--- Growth Predictor Test ---" << std::endl;
    farm_manager::GrowthPredictor predictor;
    
    std::vector<int> target_days = {0, 30, 60, 90, 120, 180, 365};
    auto results = predictor.predict_gompertz(50.0, 0, target_days, 800.0, 0.01);
    
    for (const auto& r : results) {
        std::cout << "Jour " << std::get<0>(r) 
                  << ": " << std::get<2>(r) << " kg"
                  << " (min: " << std::get<1>(r) 
                  << ", max: " << std::get<3>(r) << ")" << std::endl;
    }
    
    // Test du processeur de capteurs
    std::cout << "\n--- Sensor Processor Test ---" << std::endl;
    farm_manager::SensorProcessor processor(3600);
    
    // Ajouter des données
    for (int i = 0; i < 100; i++) {
        double timestamp = i * 60.0; // toutes les minutes
        double value = 20.0 + 5.0 * std::sin(i * 0.1) + ((rand() % 100) / 100.0);
        processor.add_data_point("temp_sensor", timestamp, value);
    }
    
    auto stats = processor.get_stats("temp_sensor");
    std::cout << "Points: " << std::get<0>(stats) << std::endl;
    std::cout << "Min: " << std::get<3>(stats) << std::endl;
    std::cout << "Max: " << std::get<4>(stats) << std::endl;
    std::cout << "Moyenne: " << std::get<5>(stats) << std::endl;
    
    // Compression des données
    auto compressed = processor.compress_data("temp_sensor", 0, 6000, 0.5);
    std::cout << "Compression: " << compressed.size() << " points (original: 100)" << std::endl;
    
    std::cout << "\n=== Test completed ===" << std::endl;
    return 0;
}