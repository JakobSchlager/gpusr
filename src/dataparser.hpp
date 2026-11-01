#ifndef DATAPARSER_H
#define DATAPARSER_H

#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <vector>

// TODO: a 1-D vector with mdspan (or stride) may already be better
auto read_dataset(std::istream&& in) {
    std::vector<std::vector<double>> result;
    std::string line;

    while (std::getline(in, line)) {
        std::istringstream line_stream(line);
        std::vector<double> values;
        double value;

        while (line_stream >> value) {
            values.push_back(value);
        }

        result.push_back(std::move(values));
    }

    return result;
}

#endif
