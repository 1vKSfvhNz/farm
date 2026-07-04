// backend/cpp_src/include/growth_predictor.hpp
#ifndef GROWTH_PREDICTOR_HPP
#define GROWTH_PREDICTOR_HPP

#include <vector>
#include <tuple>
#include <cmath>

namespace farm_manager {

/**
 * Prédicteur de croissance animale avec modèles mathématiques
 * Modèles: Gompertz, Logistique, Von Bertalanffy
 */
class GrowthPredictor {
public:
    GrowthPredictor();
    ~GrowthPredictor();

    /**
     * Prédiction avec modèle de Gompertz
     * @param weight_initial Poids initial (kg)
     * @param age_initial_days Âge initial (jours)
     * @param target_days Jours cibles
     * @param weight_inf Poids asymptotique (0 = auto)
     * @param growth_rate Taux de croissance (0 = auto)
     * @return Liste de tuples (day, weight_min, weight_mean, weight_max)
     */
    std::vector<std::tuple<int, double, double, double>> predict_gompertz(
        double weight_initial,
        int age_initial_days,
        const std::vector<int>& target_days,
        double weight_inf = 0.0,
        double growth_rate = 0.0
    );

    /**
     * Prédiction avec modèle Logistique
     */
    std::vector<std::tuple<int, double, double, double>> predict_logistic(
        double weight_initial,
        int age_initial_days,
        const std::vector<int>& target_days,
        double carrying_capacity = 0.0,
        double growth_rate = 0.0
    );

    /**
     * Prédiction avec modèle de Von Bertalanffy (poissons)
     */
    std::vector<std::tuple<int, double, double, double>> predict_von_bertalanffy(
        double weight_initial,
        int age_initial_days,
        const std::vector<int>& target_days,
        double asymptotic_weight = 0.0,
        double metabolic_rate = 0.0
    );

    /**
     * Estimation des paramètres du modèle
     * @param ages_days Âges en jours
     * @param weights_kg Poids en kg
     * @param model_type Type de modèle ("gompertz", "logistic", "von_bertalanffy")
     * @return Tuple (weight_inf, growth_rate, r_squared)
     */
    std::tuple<double, double, double> estimate_parameters(
        const std::vector<int>& ages_days,
        const std::vector<double>& weights_kg,
        const std::string& model_type = "gompertz"
    );

    /**
     * Calcul du R² (coefficient de détermination)
     */
    double calculate_r_squared(
        const std::vector<double>& observed,
        const std::vector<double>& predicted
    );

private:
    double gompertz_function(double t, double A, double mu, double lambda);
    double logistic_function(double t, double K, double r, double t0);
    double von_bertalanffy_function(double t, double W_inf, double k, double t0);
    
    double solve_gompertz_lambda(double weight_initial, int age_initial, double A, double mu);
    double solve_logistic_t0(double weight_initial, int age_initial, double K, double r);
    double solve_von_bertalanffy_t0(double weight_initial, int age_initial, double W_inf, double k);
};

} // namespace farm_manager

#endif // GROWTH_PREDICTOR_HPP