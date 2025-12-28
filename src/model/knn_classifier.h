#ifndef KNN_CLASSIFIER_H
#define KNN_CLASSIFIER_H

#include <vector>
#include "../distance/distance.h"

class KNNClassifier {
private:
    int k;
    DistanceType distance_type;
    std::vector<std::vector<double>> X_train;
    std::vector<int> y_train;

public:
    KNNClassifier(int k, DistanceType type);

    void fit(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& y
    );

    int predict(const std::vector<double>& x) const;

    std::vector<int> predict_batch(
        const std::vector<std::vector<double>>& X
    ) const;
};

#endif
