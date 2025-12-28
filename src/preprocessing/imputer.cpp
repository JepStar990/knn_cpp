#include "imputer.h"
#include <cmath>
#include <stdexcept>

void MeanImputer::fit(const std::vector<std::vector<double>>& X) {
    if (X.empty()) {
        throw std::runtime_error("Empty dataset passed to imputer");
    }

    size_t n_features = X[0].size();
    means.assign(n_features, 0.0);
    std::vector<size_t> counts(n_features, 0);

    for (const auto& row : X) {
        if (row.size() != n_features) {
            throw std::runtime_error("Inconsistent feature size");
        }

        for (size_t j = 0; j < n_features; j++) {
            if (!std::isnan(row[j])) {
                means[j] += row[j];
                counts[j]++;
            }
        }
    }

    for (size_t j = 0; j < n_features; j++) {
        if (counts[j] == 0) {
            throw std::runtime_error(
                "All values missing in column " + std::to_string(j)
            );
        }
        means[j] /= counts[j];
    }
}

std::vector<std::vector<double>> MeanImputer::transform(
    const std::vector<std::vector<double>>& X
) const {
    if (means.empty()) {
        throw std::runtime_error("Imputer has not been fitted");
    }

    std::vector<std::vector<double>> X_out = X;

    for (auto& row : X_out) {
        for (size_t j = 0; j < row.size(); j++) {
            if (std::isnan(row[j])) {
                row[j] = means[j];
            }
        }
    }

    return X_out;
}

std::vector<std::vector<double>> MeanImputer::fit_transform(
    const std::vector<std::vector<double>>& X
) {
    fit(X);
    return transform(X);
}
