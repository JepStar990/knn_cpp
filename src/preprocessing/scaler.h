#ifndef SCALER_H
#define SCALER_H

#include <vector>

class StandardScaler {
private:
    std::vector<double> means;
    std::vector<double> stds;

public:
    void fit(const std::vector<std::vector<double>>& X);
    std::vector<std::vector<double>> transform(
        const std::vector<std::vector<double>>& X
    ) const;

    std::vector<std::vector<double>> fit_transform(
        const std::vector<std::vector<double>>& X
    );
};

#endif
