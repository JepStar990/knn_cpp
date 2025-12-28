#include "distance.h"
#include <cmath>
#include <stdexcept>

double euclidean_distance(
    const std::vector<double>& a,
    const std::vector<double>& b
) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector size mismatch");
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double manhattan_distance(
    const std::vector<double>& a,
    const std::vector<double>& b
) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector size mismatch");
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

double compute_distance(
    const std::vector<double>& a,
    const std::vector<double>& b,
    DistanceType type
) {
    switch (type) {
        case DistanceType::EUCLIDEAN:
            return euclidean_distance(a, b);
        case DistanceType::MANHATTAN:
            return manhattan_distance(a, b);
        default:
            throw std::runtime_error("Unknown distance type");
    }
}
