#include "./dataparser.h"
#include <cassert>
#include <random>
#include <stack>
#include <cmath>
#include <ranges>
#include <vector>

// Tree representation:
// Expressions are stored as linearized trees in prefix (Polish) notation,
// corresponding to a preorder traversal (root, left, right).
// This representation supports stack-based evaluation by traversing
// the encoded nodes in reverse order.
// See: https://arxiv.org/abs/2501.17168
//
// Each individual uses a fixed storage capacity of max_nodes, allowing
// constant-stride addressing within a population (offset = idx * max_nodes).
// The actual number of nodes may differ between individuals.

std::random_device r;
// std::default_random_engine e1(r());
std::default_random_engine e1(1);

enum class op_codes : int {
    ADD,
    SUB,
    MUL,
    DIV,
    
    LENGTH 
};

enum class encoding : int {
    OP_2,           // binary operation
    VARIABLE_INDEX,  
    CONSTANT 
};

// returns a tuple (types, instances)
// where (types_i, instances_i) means instance_i is of type_i
// f.e.: [OP_2, OP_2, VARIABLE_INDEX, CONSTANT, CONSTANT]
//       [ADD,  MUL,  1,              25.5,     13.3]
// 
//                         ADD
//                        /   \
//                      MUL   13.3
//                     /   \   
//                VAR[1]   25.5 
std::tuple<std::vector<int>, std::vector<double>> create(
    size_t const max_nodes,
    double const min_const,
    double const max_const,
    size_t const no_variables
) {
    std::bernoulli_distribution            choose_operator(0.75);
    std::bernoulli_distribution            choose_variable(0.5);
    // chooses an index in the variable array
    std::uniform_int_distribution<size_t>  variable_dist(0, no_variables - 1);
    std::uniform_int_distribution<int>     operator_dist(0, static_cast<int>(op_codes::LENGTH) -1);
    std::uniform_real_distribution<double> constant_dist(min_const, max_const);

    std::vector<int>    encoding(max_nodes, -1); 
    auto encoding_itr { std::begin(encoding) }; 
    std::vector<double> tree;
    tree.reserve(max_nodes);

    // for a simple first implementation, we can avoid a stack if we do not care about tree depth. 
    // we can simply keep track of how many more elements we need to "fill" the tree
    int remaining = 1;

    while (remaining > 0) {
        assert(remaining >= 0);
        assert(tree.size() + remaining <= max_nodes);

        bool can_insert_operator = tree.size() + remaining + 2 <= max_nodes;

        if (can_insert_operator && choose_operator(e1)) {
            tree.emplace_back(operator_dist(e1));
            // TODO(arity): remaining = remaining - 1 + arity(operator)
            *encoding_itr++ = static_cast<int>(encoding::OP_2);
            ++remaining;
            continue;
        } 
        if (choose_variable(e1)) {
            tree.emplace_back(variable_dist(e1)); 
            *encoding_itr++ = static_cast<int>(encoding::VARIABLE_INDEX);
            --remaining;
            continue;
        }

        tree.emplace_back(constant_dist(e1)); 
        *encoding_itr++ = static_cast<int>(encoding::CONSTANT);
        --remaining;
    }

    return { encoding, tree };
}

// returns a subtree size array for a given tree's types
std::vector<int> get_sizes(const std::vector<int>& types) {
    std::vector<int> sizes(types.size());

    int index = 0;
    auto visit = [&](this auto&& self) -> int {
        int current = index++;
        int type = types[current];

        if (type > static_cast<int>(encoding::OP_2)) return sizes[current] = 1;

        int left_size = self();
        int right_size = self();

        return sizes[current] = 1 + left_size + right_size;
    };

    visit();
    return sizes;
}

double apply(
    std::vector<int>    const & types, 
    std::vector<double> const & instances,
    std::vector<double> const & variables, 
    int const   length
) {
    std::vector<double> stack(length);
    size_t sp = 0;

    for(int index = length - 1; index >= 0; --index) {
        switch(types[index]) {
            case static_cast<int>(encoding::VARIABLE_INDEX): 
            {
                stack[sp++] = variables[instances[index]];
                break;
            }
            case static_cast<int>(encoding::CONSTANT):
            {
                stack[sp++] = instances[index];
                break;
            }
            case static_cast<int>(encoding::OP_2):
            {
                double v1 = stack[--sp];
                double v2 = stack[--sp];

                switch (static_cast<int>(instances[index])) {
                    case static_cast<int>(op_codes::ADD):                 
                        stack[sp++] = v1 + v2;
                        break;
                    case static_cast<int>(op_codes::SUB):                 
                        stack[sp++] = v1 - v2;
                        break;
                    case static_cast<int>(op_codes::MUL):                 
                        stack[sp++] = v1 * v2;
                        break;
                    case static_cast<int>(op_codes::DIV):                 
                        stack[sp++] = v1 / v2;
                        break;
                    default: 
                        assert(false && "unknown operator");
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


// fitness proportional selection (roulette-wheel)
// TODO(interface)
auto get_fitness_proportional_selection(std::vector<double> const& errors) {
    std::vector<double> fitness;
    fitness.reserve(errors.size());

    for (double error : errors) {
        fitness.push_back(std::isfinite(error) && error >= 0.0 
            ? 1.0 / (1.0 + error) 
            : 0.0); 
    }

    std::discrete_distribution<std::size_t> selection(fitness.begin(),
                                                        fitness.end());

    return selection;
}

int main(void) {
    // read_dataset(std::ifstream {"../data/test.txt"}); 
    // auto [encoding, tree] = create(
    //     static_cast<size_t>(pow(2, 3)), 5, 10, 4); 
    
    auto [types, instances] = create(
        8, 5, 10, 4); 
    
    auto sizes = get_sizes(types); 

    auto print_tree = [&](this auto && self, int & index, int layer) -> void {
        std::println("{:>{}}{}: {}, {}", "", layer * 2, types[index], instances[index], sizes[index]);
        
        bool const is_operator = types[index] != 0; 
        ++index; 
        if (is_operator) return;

        self(index, layer + 1);
        self(index, layer + 1);
    }; 

    int index = 0; 
    print_tree(index, 0); 

    return 0; 
}
