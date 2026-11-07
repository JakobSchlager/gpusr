#pragma once

#include <vector>
#include "./dataset.h"

using polish = std::tuple<
    std::vector<int>,    // types  
    std::vector<double>, // instances  
    std::vector<int>     // sizes
>;

struct gpu_evaluator{
    size_t const data_rows, data_cols;

    int    *d_population_types, *d_lengths;
    double *d_population_instances, *d_mse, *d_dataset;

    gpu_evaluator(dataset const & data);
   
    std::vector<double> evaluate(std::vector<polish> const & population);

    ~gpu_evaluator();
};
   
// TODO(Architecture): CPU side
std::vector<double> evaluate(std::vector<polish> const & population, dataset const & data);

