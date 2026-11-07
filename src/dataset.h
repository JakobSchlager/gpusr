#pragma once

#include <vector>
#include <mdspan>
#include <cstddef>
#include <ranges>

struct dataset {
	std::vector<double> data;
	std::size_t 	    rows;
	std::size_t 	    cols;

	auto view() {
		return std::mdspan(data.data(), rows, cols);
	}

	auto view() const {
		return std::mdspan(data.data(), rows, cols);
	}

        auto rows_view() const {
            return std::views::iota(std::size_t{0}, rows)
                | std::views::transform([this](std::size_t i) {
                    return std::span<const double>(
                        data.data() + i * cols, cols
                    );
                });
        }
};

