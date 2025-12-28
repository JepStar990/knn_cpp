#ifndef DATASET_H
#define DATASET_H

#include <vector>

class Dataset {
public:
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    Dataset() = default;
    Dataset(
        const std::vector<std::vector<double>>& features,
        const std::vector<double>& targets
    );

    size_t num_samples() const;
    size_t num_features() const;
};

void train_test_split(
    const Dataset& full,
    Dataset& train,
    Dataset& test,
    double test_ratio,
    unsigned int seed = 42
);

#endif
