#include <iostream>
#include <string>
#include <cmath>

#include "io/csv_reader.h"
#include "data/dataset.h"
#include "preprocessing/imputer.h"
#include "preprocessing/scaler.h"
#include "model/knn_regressor.h"
#include "model/knn_classifier.h"
#include "metrics/metrics.h"

#include <fstream>

// Utility: remove rows with missing targets
void remove_missing_targets(
    std::vector<std::vector<double>>& X,
    std::vector<double>& y
) {
    std::vector<std::vector<double>> X_clean;
    std::vector<double> y_clean;

    for (size_t i = 0; i < y.size(); i++) {
        if (!std::isnan(y[i])) {
            X_clean.push_back(X[i]);
            y_clean.push_back(y[i]);
        }
    }

    X = std::move(X_clean);
    y = std::move(y_clean);
}

// Save predictions for visualization
void save_predictions(
    const std::string& filename,
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred
) {
    std::ofstream file(filename);
    file << "actual,predicted\n";
    for (size_t i = 0; i < y_true.size(); i++) {
        file << y_true[i] << "," << y_pred[i] << "\n";
    }
}

int main(int argc, char* argv[]) {
    try {
        std::string csv_path = "data/winequality.csv";
        size_t target_col = 11;   // Wine quality column
        int k = 5;

        std::cout << "Loading dataset...\n";

        CSVReader reader;
        CSVData raw = reader.read(csv_path, true, target_col);

        // Remove rows with missing targets
        remove_missing_targets(raw.features, raw.targets);

        Dataset full(raw.features, raw.targets);

        Dataset train, test;
        train_test_split(full, train, test, 0.2);

        std::cout << "Train samples: " << train.num_samples() << "\n";
        std::cout << "Test samples: " << test.num_samples() << "\n";

        // Imputation
        MeanImputer imputer;
        train.X = imputer.fit_transform(train.X);
        test.X = imputer.transform(test.X);

        // Scaling
        StandardScaler scaler;
        train.X = scaler.fit_transform(train.X);
        test.X = scaler.transform(test.X);

        // KNN Regression
        KNNRegressor knn(k, DistanceType::EUCLIDEAN);
        knn.fit(train.X, train.y);

        std::vector<double> predictions =
            knn.predict_batch(test.X);

        double test_rmse = rmse(test.y, predictions);
        double test_mae = mae(test.y, predictions);

        std::cout << "KNN Regression Results\n";
        std::cout << "RMSE: " << test_rmse << "\n";
        std::cout << "MAE : " << test_mae << "\n";

        save_predictions("predictions.csv", test.y, predictions);

        std::cout << "Predictions saved to predictions.csv\n";
    }
    catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
