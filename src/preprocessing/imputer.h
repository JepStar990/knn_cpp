#ifndef IMPUTER_H
#define IMPUTER_H

#include <vector>

class MeanImputer {
private:
    std::vector<double> means;

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
