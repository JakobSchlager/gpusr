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

// Note: here we would probably not need the stack at all, since sizes are known this point.
// But for the later integration this will probably not be the case.
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
            case static_cast<int>(type::OP_1):
            {
                assert(sp >= 1);

                double const v = stack[--sp];

                switch(static_cast<int>(instances[index])) {
                    case static_cast<int>(op_code::SIN): 
                    {
                        double const value = std::sin(v);
                        stack[sp++] = value;
			primals [index] = value;
                        break;
                    }
                    case static_cast<int>(op_code::COS): 
                    {
                        double const value = std::cos(v); 
                        stack[sp++] = value;
			primals [index] = value;
                        break;
                    }
                    case static_cast<int>(op_code::SQRT):
                    {
                        double const value = std::sqrt(v); 
                        stack[sp++] = value;
			primals [index] = value;
                        break;
                    }
                    case static_cast<int>(op_code::EXP): 
                    {
                        double const value = std::exp(v); 
                        stack[sp++] = value;
			primals [index] = value;
                        break;
                    }
                    case static_cast<int>(op_code::LOG): 
                    {
                        double const value = std::log(v); 
                        stack[sp++] = value;
			primals [index] = value;
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
	            {
			double const value = v1 + v2;
                        stack[sp++]        = value;
			primals[index]     = value;
                        break;
		    }
                    case static_cast<int>(op_code::SUB):                 
		    {
			double const value = v1 - v2;
                        stack[sp++]        = value;
			primals[index]     = value;
                        break;
		    }
                    case static_cast<int>(op_code::MUL):                 
		    {
			double const value = v1 * v2;
                        stack[sp++]        = value;
			primals[index]     = value;
                        break;
		    }
                    case static_cast<int>(op_code::DIV):                 
		    {
			double const value = v1 / v2;
                        stack[sp++]        = value;
			primals[index]     = value;
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
    // for (auto const val : primals) {
    //     std::println("{0}", val);
    // }
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
    adjoints[0] = 1.0;

    for(size_t index = 0; index < sizes[0]; ++index) {
        switch(types[index]) {
            case static_cast<int>(type::OP_1):
            {
		double const parent    = adjoints[index];
                size_t const child_idx = index + 1;

                switch(static_cast<int>(instances[index])) {
                    case static_cast<int>(op_code::SIN): 
                    {
			adjoints[child_idx] += parent * std::cos(primals[child_idx]);
                        break;
                    }
                    case static_cast<int>(op_code::COS): 
                    {
			double const d = -std::sin(primals[child_idx]);
			adjoints[child_idx] += parent * d;
                        break;
                    }
                    case static_cast<int>(op_code::SQRT):
                    {
			double const d = 1 / (2 * std::sqrt(primals[child_idx]));
			adjoints[child_idx] += parent * d;
                        break;
                    }
                    case static_cast<int>(op_code::EXP): 
                    {
			// TODO: could reuse primals[index], leaving it explicitely for now.
			adjoints[child_idx] += parent * std::exp(primals[child_idx]);
                        break;
                    }
                    case static_cast<int>(op_code::LOG): 
                    {
			adjoints[child_idx] += parent * (1 / primals[child_idx]);
                        break;
                    }
                    default: 
                        assert(false && "unknown unary operator");

                }
                break;
            }
            case static_cast<int>(type::OP_2):
            {
		double const parent    = adjoints[index];
		size_t const left_idx  = index + 1;
		size_t const right_idx = left_idx + sizes[left_idx];

                switch (static_cast<int>(instances[index])) {
                    case static_cast<int>(op_code::ADD):                 
	            {
			adjoints[left_idx]  += parent * 1;
			adjoints[right_idx] += parent * 1;
			break;
		    }
                    case static_cast<int>(op_code::SUB):                 
	            {
			adjoints[left_idx]  += parent * 1;
			adjoints[right_idx] -= parent * 1;
			break;
		    }
                    case static_cast<int>(op_code::MUL):                 
		    {
			adjoints[left_idx]  += parent * primals[right_idx];
			adjoints[right_idx] += parent * primals[left_idx];
                        break;
		    }
                    case static_cast<int>(op_code::DIV): {
                        double const l = primals[left_idx];
                        double const r = primals[right_idx];
                    
                        adjoints[left_idx]  += parent / r;
                        adjoints[right_idx] -= parent * l / (r * r);
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
	for (size_t i = 0; i < sizes[0]; ++i) {
		bool const is_num = types[i] > 1;
		if (is_num) std::println("{}: {}", i, adjoints[i]);
	}
}

// for testing, i.e. compare the results to a finite difference approximation
void finite_difference(polish const & example, std::vector<double> const & variables) {
	std::array<double, 15> stack;
        auto const & [types, instances, sizes] = example;

	double const h = 1e-5;

	int const type_var   = static_cast<int>(type::VARIABLE_INDEX);
	int const type_const = static_cast<int>(type::CONSTANT);

	for (int i = 0; i < sizes[0]; ++i) {
		// since we do not filter out variables, we want to compare variables or constants
		// TODO: should really factor this out as well, otherwise the enum checks become extremely brittle
		bool const is_num    = (types[i] > 1);
	        if (!is_num) continue;

		auto copy_inst = instances;
		auto copy_var  = variables;
		
		// const path
		double * value  = &copy_inst[i];
		if (types[i] == type_var) {
			size_t const var_index = instances[i];
			value                  = &copy_var[var_index];
		}
		double const original = *value;

		*value           = original + h;
		auto const left  = forward(types, copy_inst, copy_var, stack)[0]; 

		*value           = original - h;
		auto const right = forward(types, copy_inst, copy_var, stack)[0]; 

		double const fd = (left - right) / (2*h);
		std::println("{}", fd);
	}
}

void test(polish const & tree, std::vector<double> const & variables, std::string_view title) {
	std::println("================== {} ==================", title);
	run_on(tree, variables);
	std::println("finite difference");
	finite_difference(tree, variables);
	std::println();
}

// TODO: Can probably just filter out actual constants after getting the full adjoints?
int main(void) {
    	std::vector<double> variables { -1, -1, -1, -1, -1, -1}; 
	
	// a * b + c with a = 5, b = 10, c = 15, should be (b, a, 1)
 	polish abc {
		{1, 1, 3,  3,  3}, 
		{5, 7, 5, 10, 15},
                {5, 3, 1,  1,  1}
	};
	test(abc, variables, "abc");

	// Nested expression (on the right)
        // a=2, b=3, c=4
        // Expected gradient: (b+c, a, a) = (7, 2, 2)
        polish nested {
            {1, 3, 1, 3, 3},
            {7, 2, 5, 3, 4},
            {5, 1, 3, 1, 1}
        };
	test(nested, variables, "nested");

	// Multiple Subtrees
        // a=2, b=3, c=4, d=5
        // Expected gradient: (9, 9, 5, 5)
        polish both_subtrees {
            {1, 1, 3, 3, 1, 3, 3},
            {7, 5, 2, 3, 5, 4, 5},
            {7, 3, 1, 1, 3, 1, 1}
        };
        test(both_subtrees, variables, "both_subtrees");
  
	// Division and Subtraction
        // a=2, b=3, c=10, d=4
        // Expected gradient: (0.5, 1/3, -1/6, 1/6)
        polish quotient {
            {1, 1, 3, 3, 1, 3, 3},
            {8, 7, 2, 3, 6, 10, 4},
            {7, 3, 1, 1, 3, 1, 1}
        };
	test(quotient, variables, "quotient");
        
	// Nested unary functions
        // a=2, b=3
        polish nested_unary {
            {0, 1, 3, 0, 3},
            {0, 7, 2, 1, 3},
            {5, 4, 1, 2, 1}
        };
	test(nested_unary, variables, "nested_unary");

        // a=2.5
        // Expected gradient: (1)
        polish cancellation {
            {0, 0, 3},
            {3, 4, 2.5},
            {3, 2, 1}
        };
        test(cancellation, variables, "cancellation");

        // a=2, x=variables[0], b=5
        // With variables[0] = 3:
        // Expected constant gradient: (3, 1)
        polish mixed {
            {1, 1, 3, 2, 3},
            {5, 7, 2, 0, 5},
            {5, 3, 1, 1, 1}
        };
      
        variables[0] = 3;
        test(mixed, variables, "mixed");
  
	return 0;
}

