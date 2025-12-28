#include "csv_reader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <cmath>

bool CSVReader::is_missing(const std::string& token) {
    return token.empty() ||
           token == "NA" ||
           token == "NaN" ||
           token == "?";
}

double CSVReader::to_double(const std::string& token) {
    if (is_missing(token)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::stod(token);
}

CSVData CSVReader::read(
    const std::string& filename,
    bool has_header,
    size_t target_col
) {
    CSVData data;

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file");
    }

    std::string line;

    // Skip header if present
    if (has_header) {
        std::getline(file, line);
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        std::vector<double> row;
        std::vector<std::string> raw_cells;

        while (std::getline(ss, cell, ',')) {
            raw_cells.push_back(cell);
        }

        if (target_col >= raw_cells.size()) {
            throw std::runtime_error("Target column index out of range");
        }

        for (size_t i = 0; i < raw_cells.size(); i++) {
            if (i == target_col) continue;
            row.push_back(to_double(raw_cells[i]));
        }

        data.features.push_back(row);
        data.targets.push_back(to_double(raw_cells[target_col]));
        data.raw_targets.push_back(raw_cells[target_col]);
    }

    return data;
}
