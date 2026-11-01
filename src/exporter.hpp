#pragma once

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

struct exporter {
    std::filesystem::path _dir{"../../results"}; 
    std::string _filename;

    exporter(std::string_view filename) 
    : _filename{filename} { }

    void append_result(int iteration, double score) {
        generation_results.emplace_back(iteration, score);
    }

    void export_csv() {
        std::filesystem::create_directories(_dir);
        const auto path = _dir / (_filename + ".csv");

        std::ofstream out(path);
        assert(out); 

        for (auto const & [iteration, score] : generation_results) {
            out << iteration << "," << score << std::endl;
        }
    }

private:
    std::vector<std::pair<int, double>> generation_results{};
};

