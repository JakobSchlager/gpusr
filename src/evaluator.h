#pragma once

#include <vector>

using polish = std::tuple<
    std::vector<int>,    // types  
    std::vector<double>, // instances  
    std::vector<int>     // sizes
>;

using dataset_t = std::vector<std::vector<double>>;

std::vector<double> evaluate(
    std::vector<polish> const & population, 
    dataset_t           const & dataset
);
