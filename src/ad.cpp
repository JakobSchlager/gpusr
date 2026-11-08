#include "operators.hpp"
#include <vector>
#include <span>
#include <print>
#include <cmath>

// Automatic Differentation playground
// https://arxiv.org/abs/1502.05767

using polish = std::tuple<
    std::vector<int>,    // types  
    std::vector<double>, // instances  
    std::vector<int>     // sizes
>;

auto forward(
    std::vector<int>        const & types, 
    std::vector<double>     const & instances,
    std::vector<double>     const & variables, 
    auto		    & stack
) {
    size_t sp = 0;

    // stores the primal of each intermedia subtree (i.e. their value)
    std::vector<double> primals(std::size(types));
    // stores the derivates
    std::vector<double> adjoints(std::size(types));

    for(int index = std::size(types) - 1; index >= 0; --index) {
        switch(types[index]) {
            case static_cast<int>(type::VARIABLE_INDEX): 
            {
                auto variable_index = static_cast<std::size_t>(instances[index]);
		double const value = variables[variable_index];
                stack[sp++] = value;
		primals[index] = value;

                break;
            }
            case static_cast<int>(type::CONSTANT):
            {
		double const value = instances[index];
                stack[sp++] = value;
		primals[index] = value;
                break;
            }
            case static_cast<int>(type::OP_2):
            {
                assert(sp >= 2);

                double const v1 = stack[--sp];
                double const v2 = stack[--sp];

                switch (static_cast<int>(instances[index])) {
                    case static_cast<int>(op_code::ADD):                 
	            {
			double const value = v1 + v2;
                        stack[sp++] = value;
			primals[index] = value;
                        break;
		    }
                    case static_cast<int>(op_code::MUL):                 
		    {
			double const value = v1 * v2;
                        stack[sp++] = value;
			primals[index] = value;
                        break;
		    }
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
    for (auto const val : primals) {
	std::println("{0}", val);
    }
    return primals;
}

auto backwards(
    std::vector<int>    const & types, 
    std::vector<double> const & instances,
    std::vector<int>    const & sizes, 
    std::vector<double> const & variables, 
    std::vector<double> const & primals
) {
    std::vector<double> adjoints(sizes[0]);
    adjoints[0] = 1.0; // TODO: always? 

    for(size_t index = 0; index < sizes[0]; ++index) {
        switch(types[index]) {
            case static_cast<int>(type::VARIABLE_INDEX): 
            {
                break;
            }
            case static_cast<int>(type::CONSTANT):
            {
                break;
            }
            case static_cast<int>(type::OP_2):
            {
                assert(sp >= 2);

		double const parent    = adjoints[index];
		size_t const left_idx  = index + 1;
		size_t const right_idx = left_idx + sizes[left_idx];

                switch (static_cast<int>(instances[index])) {
                    case static_cast<int>(op_code::ADD):                 
	            {
			adjoints[left_idx]  = parent * 1;
			adjoints[right_idx] = parent * 1;
			break;
		    }
                    case static_cast<int>(op_code::MUL):                 
		    {
			adjoints[left_idx]  = parent * primals[right_idx];
			adjoints[right_idx] = parent * primals[left_idx];
                        break;
		    }
                    default: 
                        assert(false && "unknown binary operator");
                }

                break;
            }
            default: 
                assert(false && "uknown node type");
        }
    }
     

    return adjoints;
}

void run_on(polish const & example, std::vector<double> const & variables) {
	std::array<double, 15> stack;
        auto const & [types, instances, sizes] = example;
	auto const & primals = forward(types, instances, variables, stack); 
	auto const adjoints  = backwards(types, instances, sizes, variables, primals); 
	
	std::println("Adjoints:");
	for (size_t i = 0; auto const d_x : adjoints) std::println("{}: {}", i++, d_x);
}

int main(void) {
    	std::vector<double> variables { -1, -1, -1, -1, -1, -1}; 
	
	// a * b + c with a = 5, b = 10, c = 15
 	polish abc {
		{1, 1, 3,  3,  3}, 
		{5, 7, 5, 10, 15},
                {5, 3, 1,  1,  1}
	};
	run_on(abc, variables);

	return 0;
}




