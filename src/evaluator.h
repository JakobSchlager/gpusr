#pragma once

#include <vector>
#include "./dataset.h"

using polish = std::tuple<
    std::vector<int>,    // types  
    std::vector<double>, // instances  
    std::vector<int>     // sizes
>;

struct cpu_evaluator{
    dataset const & m_data;

    // TODO(Architecture): Storing a reference to data. 
    // Could just bind evaluate to data as well.
    cpu_evaluator(dataset const & data);
   
    std::vector<double> evaluate(std::vector<polish> const & population);
};

struct gpu_evaluator{
    size_t const data_rows, data_cols;

    int    *d_population_types, *d_lengths;
    double *d_population_instances, *d_mse, *d_dataset;

    gpu_evaluator(dataset const & data);
   
    std::vector<double> evaluate(std::vector<polish> const & population);

    ~gpu_evaluator();
};

