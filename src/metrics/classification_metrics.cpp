#include "metrics.h"
#include <stdexcept>

double accuracy(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred
) {
    if (y_true.size() != y_pred.size()) {
        throw std::runtime_error("Size mismatch in accuracy");
    }

    int correct = 0;
    for (size_t i = 0; i < y_true.size(); i++) {
        if (y_true[i] == y_pred[i]) {
            correct++;
        }
    }

    return static_cast<double>(correct) / y_true.size();
}

std::vector<std::vector<int>> confusion_matrix(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    int num_classes
) {
    std::vector<std::vector<int>> matrix(
        num_classes, std::vector<int>(num_classes, 0)
    );

    for (size_t i = 0; i < y_true.size(); i++) {
        matrix[y_true[i]][y_pred[i]]++;
    }

    return matrix;
}
