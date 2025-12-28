#ifndef METRICS_H
#define METRICS_H

#include <vector>

double accuracy(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred
);

std::vector<std::vector<int>> confusion_matrix(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    int num_classes
);

double rmse(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
);

double mae(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
);

#endif
