#pragma once

#include <vector>
#include "./dataset.h"

using polish = std::tuple<
    std::vector<int>,    // types  
    std::vector<double>, // instances  
    std::vector<int>     // sizes
>;

std::vector<double> evaluate(
    std::vector<polish> const & population, 
    dataset             const & data
);
