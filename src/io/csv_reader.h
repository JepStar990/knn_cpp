#ifndef CSV_READER_H
#define CSV_READER_H

#include <vector>
#include <string>

struct CSVData {
    std::vector<std::vector<double>> features;
    std::vector<double> targets;   // double for regression OR encoded class
    std::vector<std::string> raw_targets;
};

class CSVReader {
public:
    CSVData read(
        const std::string& filename,
        bool has_header = true,
        size_t target_col = -1
    );

private:
    bool is_missing(const std::string& token);
    double to_double(const std::string& token);
};

#endif
