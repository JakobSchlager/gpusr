#include <vector>

#include <cassert>
#include <iterator>
#include <cmath>
#include <vector>
#include <span>

#include "./operators.hpp"
#include "./parameters.h"
#include "./dataset.h"

#include "./evaluator.h"

using polish = std::tuple<
    std::vector<int>,    // types  
    std::vector<double>, // instances  
    std::vector<int>     // sizes
>;

double interpret(
    std::vector<int>        const & types, 
    std::vector<double>     const & instances,
    std::span<const double> const & variables, 
    auto		          & stack
) {
    size_t sp = 0;

    for(int index = std::size(types) - 1; index >= 0; --index) {
        switch(types[index]) {
            case static_cast<int>(type::VARIABLE_INDEX): 
            {
                auto variable_index = static_cast<std::size_t>(instances[index]);
                stack[sp++] = variables[variable_index];
                break;
            }
            case static_cast<int>(type::CONSTANT):
            {
                stack[sp++] = instances[index];
                break;
            }
            case static_cast<int>(type::OP_1):
            {
                assert(sp >= 1);

                double const v = stack[--sp];

                switch(static_cast<int>(instances[index])) {
                    case static_cast<int>(op_code::SIN): 
                    {
                        stack[sp++] = std::sin(v);
                        break;
                    }
                    case static_cast<int>(op_code::COS): 
                    {
                        stack[sp++] = std::cos(v); 
                        break;
                    }
                    case static_cast<int>(op_code::SQRT):
                    {
                        stack[sp++] = std::sqrt(v); 
                        break;
                    }
                    case static_cast<int>(op_code::EXP): 
                    {
                        stack[sp++] = std::exp(v); 
                        break;
                    }
                    case static_cast<int>(op_code::LOG): 
                    {
                        stack[sp++] = std::log(v); 
                        break;
                    }
                    default: 
                        assert(false && "unknown unary operator");

                }
                break;
            }
            case static_cast<int>(type::OP_2):
            {
                assert(sp >= 2);

                double const v1 = stack[--sp];
                double const v2 = stack[--sp];

                switch (static_cast<int>(instances[index])) {
                    case static_cast<int>(op_code::ADD):                 
                        stack[sp++] = v1 + v2;
                        break;
                    case static_cast<int>(op_code::SUB):                 
                        stack[sp++] = v1 - v2;
                        break;
                    case static_cast<int>(op_code::MUL):                 
                        stack[sp++] = v1 * v2;
                        break;
                    case static_cast<int>(op_code::DIV):                 
                        stack[sp++] = v1 / v2;
                        break;
                    default: 
                        assert(false && "unknown binary operator");
                }

                break;
            }
            default: 
                assert(false && "uknown node type");
        }
    }

    assert(sp == 1 && "valid expression must leave exactly one value on the evaluation stack"); 
    return stack[0];
}

double interpret(
    polish                  const & candidate, 
    std::span<const double> const & variables, 
    auto                          & stack
) {
    auto const & [types, instances, sizes] = candidate;
    return interpret(types, instances, variables, stack);
}

cpu_evaluator::cpu_evaluator(dataset const & data) : m_data{data} {};

std::vector<double> cpu_evaluator::evaluate(std::vector<polish> const & population) {
    std::vector<double> results{};

    std::size_t const y_index = m_data.cols - 1;
    // can reuse the stack (should be unique per thread)
    std::array<double, max_nodes> stack; 
    for(auto const & candidate : population) {
        // TODO: hopefully don't need a running average here (in case of overflow)
        double ssr = 0;
        for(auto const & datapoint : m_data.rows_view()) {
            double const result = interpret(candidate, datapoint, stack);
            if (!std::isfinite(result)) {
              ssr = std::numeric_limits<double>::infinity();
              break;
            }
            double const error  = result - datapoint[y_index];
            ssr += error * error;
        }
        double const mse = ssr / m_data.rows;
        results.push_back(mse);
    }

    return results;
}

