#pragma once

#include <sstream>
#include <string>
#include <vector>
#include <cassert>
#include "dataset.h"

dataset read_dataset(std::istream&& in) {
    std::vector<double> result;
    std::string line;

    size_t rows = 0;
    size_t cols = 0;
    while (std::getline(in, line)) {
        std::istringstream line_stream(line);
        double value;

        size_t current_cols = 0;
        while (line_stream >> value) {
            result.push_back(value);
	    ++current_cols;
        }

        if (rows == 0) cols = current_cols;
        assert(current_cols == cols && "The number of columns must stay consistent.");

	++rows;
    }

    assert(rows > 0);
    assert(cols > 0);

    return {
	.data = std::move(result),
	.rows = rows,
	.cols = cols
    };
}

