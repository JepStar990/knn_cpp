#include "knn_regressor.h"
#include <algorithm>

KNNRegressor::KNNRegressor(int k, DistanceType type)
    : k(k), distance_type(type) {}

void KNNRegressor::fit(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y
) {
    X_train = X;
    y_train = y;
}

double KNNRegressor::predict(const std::vector<double>& x) const {
    std::vector<std::pair<double, double>> distances;

    for (size_t i = 0; i < X_train.size(); i++) {
        double d = compute_distance(x, X_train[i], distance_type);
        distances.emplace_back(d, y_train[i]);
    }

    std::sort(distances.begin(), distances.end());

    double sum = 0.0;
    for (int i = 0; i < k; i++) {
        sum += distances[i].second;
    }

    return sum / k;
}

std::vector<double> KNNRegressor::predict_batch(
    const std::vector<std::vector<double>>& X
) const {
    std::vector<double> predictions;
    for (const auto& row : X) {
        predictions.push_back(predict(row));
    }
    return predictions;
}
