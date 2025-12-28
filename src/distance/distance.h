#ifndef DISTANCE_H
#define DISTANCE_H

#include <vector>

enum class DistanceType {
    EUCLIDEAN,
    MANHATTAN
};

double euclidean_distance(
    const std::vector<double>& a,
    const std::vector<double>& b
);

double manhattan_distance(
    const std::vector<double>& a,
    const std::vector<double>& b
);

double compute_distance(
    const std::vector<double>& a,
    const std::vector<double>& b,
    DistanceType type
);

#endif
