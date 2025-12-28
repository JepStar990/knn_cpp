#include "scaler.h"
#include <cmath>
#include <stdexcept>

void StandardScaler::fit(const std::vector<std::vector<double>>& X) {
    if (X.empty()) {
        throw std::runtime_error("Empty dataset passed to scaler");
    }

    size_t n_samples = X.size();
    size_t n_features = X[0].size();

    means.assign(n_features, 0.0);
    stds.assign(n_features, 0.0);

    // Compute means
    for (const auto& row : X) {
        if (row.size() != n_features) {
            throw std::runtime_error("Inconsistent feature size");
        }
        for (size_t j = 0; j < n_features; j++) {
            means[j] += row[j];
        }
    }

    for (size_t j = 0; j < n_features; j++) {
        means[j] /= n_samples;
    }

    // Compute standard deviations
    for (const auto& row : X) {
        for (size_t j = 0; j < n_features; j++) {
            double diff = row[j] - means[j];
            stds[j] += diff * diff;
        }
    }

    for (size_t j = 0; j < n_features; j++) {
        stds[j] = std::sqrt(stds[j] / n_samples);

        // Handle zero variance safely
        if (stds[j] == 0.0) {
            stds[j] = 1.0;
        }
    }
}

std::vector<std::vector<double>> StandardScaler::transform(
    const std::vector<std::vector<double>>& X
) const {
    if (means.empty() || stds.empty()) {
        throw std::runtime_error("Scaler has not been fitted");
    }

    std::vector<std::vector<double>> X_out = X;

    for (auto& row : X_out) {
        for (size_t j = 0; j < row.size(); j++) {
            row[j] = (row[j] - means[j]) / stds[j];
        }
    }

    return X_out;
}

std::vector<std::vector<double>> StandardScaler::fit_transform(
    const std::vector<std::vector<double>>& X
) {
    fit(X);
    return transform(X);
}
