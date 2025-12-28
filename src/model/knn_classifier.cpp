#include "knn_classifier.h"
#include <algorithm>
#include <unordered_map>
#include <limits>

KNNClassifier::KNNClassifier(int k, DistanceType type)
    : k(k), distance_type(type) {}

void KNNClassifier::fit(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y
) {
    X_train = X;
    y_train = y;
}

int KNNClassifier::predict(const std::vector<double>& x) const {
    std::vector<std::pair<double, int>> distances;

    for (size_t i = 0; i < X_train.size(); i++) {
        double d = compute_distance(x, X_train[i], distance_type);
        distances.emplace_back(d, y_train[i]);
    }

    std::sort(distances.begin(), distances.end());

    std::unordered_map<int, int> votes;
    std::unordered_map<int, double> dist_sum;

    for (int i = 0; i < k; i++) {
        int label = distances[i].second;
        votes[label]++;
        dist_sum[label] += distances[i].first;
    }

    int best_label = -1;
    int best_votes = -1;
    double best_dist = std::numeric_limits<double>::max();

    for (const auto& pair : votes) {
        int label = pair.first;
        int count = pair.second;
        double avg_dist = dist_sum[label] / count;

        if (count > best_votes ||
            (count == best_votes && avg_dist < best_dist)) {
            best_votes = count;
            best_dist = avg_dist;
            best_label = label;
        }
    }

    return best_label;
}

std::vector<int> KNNClassifier::predict_batch(
    const std::vector<std::vector<double>>& X
) const {
    std::vector<int> predictions;
    for (const auto& row : X) {
        predictions.push_back(predict(row));
    }
    return predictions;
}
