#ifndef KNN_REGRESSOR_H
#define KNN_REGRESSOR_H

#include <vector>
#include "../distance/distance.h"

class KNNRegressor {
private:
    int k;
    DistanceType distance_type;
    std::vector<std::vector<double>> X_train;
    std::vector<double> y_train;

public:
    KNNRegressor(int k, DistanceType type);

    void fit(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y
    );

    double predict(const std::vector<double>& x) const;

    std::vector<double> predict_batch(
        const std::vector<std::vector<double>>& X
    ) const;
};

#endif
