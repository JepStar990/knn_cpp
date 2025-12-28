#include "dataset.h"
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <random>

Dataset::Dataset(
    const std::vector<std::vector<double>>& features,
    const std::vector<double>& targets
) : X(features), y(targets) {
    if (X.size() != y.size()) {
        throw std::runtime_error("Feature/target size mismatch");
    }
}

size_t Dataset::num_samples() const {
    return X.size();
}

size_t Dataset::num_features() const {
    if (X.empty()) return 0;
    return X[0].size();
}

void train_test_split(
    const Dataset& full,
    Dataset& train,
    Dataset& test,
    double test_ratio,
    unsigned int seed
) {
    if (test_ratio <= 0.0 || test_ratio >= 1.0) {
        throw std::runtime_error("test_ratio must be between 0 and 1");
    }

    size_t n = full.num_samples();
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    size_t test_size = static_cast<size_t>(n * test_ratio);

    test.X.clear();
    test.y.clear();
    train.X.clear();
    train.y.clear();

    for (size_t i = 0; i < n; i++) {
        size_t idx = indices[i];
        if (i < test_size) {
            test.X.push_back(full.X[idx]);
            test.y.push_back(full.y[idx]);
        } else {
            train.X.push_back(full.X[idx]);
            train.y.push_back(full.y[idx]);
        }
    }
}
