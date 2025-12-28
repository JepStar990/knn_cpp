#include "metrics.h"
#include <cmath>
#include <stdexcept>

double rmse(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::runtime_error("Size mismatch in RMSE");
    }

    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }

    return std::sqrt(sum / y_true.size());
}

double mae(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::runtime_error("Size mismatch in MAE");
    }

    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        sum += std::abs(y_true[i] - y_pred[i]);
    }

    return sum / y_true.size();
}
