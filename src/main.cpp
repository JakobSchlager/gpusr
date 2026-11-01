#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <random>
#include <cmath>
#include <ranges>
#include <type_traits>
#include <vector>

#include "./operators.hpp"
#include "./evaluator.hpp"
#include "./dataparser.hpp"
#include "./exporter.hpp"

// Tree representation:
// Expressions are stored as linearized trees in prefix (Polish) notation,
// corresponding to a preorder traversal (root, left, right).
// This representation supports stack-based evaluation by traversing
// the encoded nodes in reverse order.
// See: https://arxiv.org/abs/2501.17168

std::random_device r;
// std::default_random_engine e1(r());
std::default_random_engine e1(1);

using polish = std::tuple<
    std::vector<int>,    // types  
    std::vector<double>, // instances  
    std::vector<int>     // sizes
>;

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
    size_t const no_variables,
    std::vector<op_code> const & active_operators
) {
    assert(max_nodes >= 1);
    assert(no_variables >= 1);

    std::bernoulli_distribution            choose_operator(0.75);
    std::bernoulli_distribution            choose_variable(0.5);
    // chooses an index in the variable array
    std::uniform_int_distribution<size_t>  variable_dist(0, no_variables - 1);
    std::uniform_int_distribution<int>     operator_dist(0, std::size(active_operators) -1);
    std::uniform_real_distribution<double> constant_dist(min_const, max_const);

    std::vector<int>    types;
    std::vector<double> instances;
    types.reserve(max_nodes);
    instances.reserve(max_nodes);

    // for a simple first implementation, we can avoid a stack if we do not care about tree depth. 
    // we can simply keep track of how many more elements we need to "fill" the tree
    int remaining = 1;

    while (remaining > 0) {
        assert(remaining >= 0);
        assert(std::size(instances) + remaining <= max_nodes);

        auto op = active_operators[operator_dist(e1)];
        auto const [arity, type] = opcode_arity(op);
        bool can_insert_operator = std::size(instances) + remaining + arity <= max_nodes;
        if (can_insert_operator && choose_operator(e1)) {
            instances.emplace_back(static_cast<int>(op));
            types.push_back(static_cast<int>(type));
            remaining = remaining - 1 + arity;
            continue;
        } 
        if (choose_variable(e1)) {
            instances.emplace_back(variable_dist(e1)); 
            types.push_back(static_cast<int>(type::VARIABLE_INDEX)); 
            --remaining;
            continue;
        }

        instances.emplace_back(constant_dist(e1)); 
        types.push_back(static_cast<int>(type::CONSTANT));
        --remaining;
    }

    return { types, instances };
}

// returns a subtree size array for a given tree's types
std::vector<int> get_sizes(const std::vector<int>& types) {
    std::vector<int> sizes(types.size());

    int index = 0;
    auto visit = [&](this auto&& self) -> int {
        int current = index++;
        int type = types[current];

        if (type > static_cast<int>(type::OP_2)) return sizes[current] = 1;

        int const arity = static_cast<int>(type) + 1; 
        int size = 1;
        for (int i = 0; i < arity; ++i) size += self();
        
        return sizes[current] = size;
    };

    visit();
    assert(index == static_cast<int>(types.size()));
    return sizes;
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

void replace_subtree(
    auto& dst,
    const auto& src_begin,
    const auto& src_end,
    std::size_t const index,
    std::size_t const count
) {
    dst.erase(dst.begin() + index, dst.begin() + index + count);
    dst.insert(dst.begin() + index, src_begin, src_end);
};

void mutate(
    std::vector<int>&    types, 
    std::vector<double>& instances, 
    std::vector<int>&    sizes, 
    int const            max_nodes, 
    double const         min_const,
    double const         max_const,
    int const            no_variables,
    std::vector<op_code> const & active_operators
) {
    // prefix "m_" for "mutation"
    int const m_index = std::uniform_int_distribution<>(0, sizes[0] - 1)(e1);
    int const original_size  = sizes[m_index];
    
    int const replacement_max_nodes = max_nodes - sizes[0] + original_size;
    auto const [m_types, m_instances] = create(replacement_max_nodes, min_const, max_const, no_variables, active_operators);

    replace_subtree(types, 
        std::begin(m_types), std::end(m_types),m_index, original_size); 
    replace_subtree(instances, 
        std::begin(m_instances), std::end(m_instances), m_index, original_size);
    sizes = get_sizes(types);
}

void mutate(
    polish&      candidate,
    int const    max_nodes, 
    double const min_const,
    double const max_const,
    int const    no_variables,
    std::vector<op_code> const & active_operators
) {
   auto & [types, instances, sizes] = candidate;
   mutate(types, instances, sizes, max_nodes, min_const, max_const, no_variables, active_operators);
}

// chooses one subtree from each parent and exchanges them
// returns two new candidates (c1, c2), where 
//  c1 = a[0:i] b[j:j+size(b, j)] a[i + size(a, i):]
//  c2 = b[0:j] a[i:i+size(a, i)] b[j + size(b, j):]
// for two crossover points i, j.
std::pair<polish, polish> subtree_crossover(
    polish const& a,
    polish const& b,
    size_t const  max_nodes
) {
    auto [a_types, a_instances, a_sizes] = a;
    auto [b_types, b_instances, b_sizes] = b;

    auto crossover_point_a_dist = std::uniform_int_distribution<>(0, a_sizes[0] - 1);
    auto crossover_point_b_dist = std::uniform_int_distribution<>(0, b_sizes[0] - 1);
    int i = crossover_point_a_dist(e1);
    int j = crossover_point_b_dist(e1);
    // continuously sample the distributions until we find a pair (i, j) which do not violate the max_nodes constraint.
    // may find a better solution for this, but it is guaranteed to have a solution (exchanging the roots).
    while (a_sizes[0] - a_sizes[i] + b_sizes[j] > max_nodes || b_sizes[0] - b_sizes[j] + a_sizes[i] > max_nodes) {
        i = crossover_point_a_dist(e1);
        j = crossover_point_b_dist(e1);
    }

    int const size_a = a_sizes[i];
    int const size_b = b_sizes[j];

    // replace subtree of A with subtree of B
    replace_subtree(
        a_types,
        b_types.begin() + j,
        b_types.begin() + j + size_b,
        i,
        size_a
    );
    replace_subtree(
        a_instances,
        b_instances.begin() + j,
        b_instances.begin() + j + size_b,
        i,
        size_a
    );

    // replace subtree of B with subtree of A
    // since we copied a and b already, we can use the original a as our temporary
    auto const & [tmp_types, tmp_instances, _] = a;
    replace_subtree(
        b_types,
        tmp_types.begin() + i,
        tmp_types.begin() + i + size_a,
        j,
        size_b
    );
    replace_subtree(
        b_instances,
        tmp_instances.begin() + i,
        tmp_instances.begin() + i + size_a,
        j,
        size_b
    );

    a_sizes = get_sizes(a_types);
    b_sizes = get_sizes(b_types);

    return {
        {std::move(a_types), std::move(a_instances), std::move(a_sizes)},
        {std::move(b_types), std::move(b_instances), std::move(b_sizes)}
    };
}

void print_tree(polish const & candidate) {
    auto const & [types, instances, sizes] = candidate;

    auto print_tree_rec = [&](this auto && self, int & index, int layer) -> void {
        assert(index >= 0 && index < static_cast<int>(std::size(types)));

        std::println("{:>{}}{}: {}, {}", "", layer * 2, types[index], instances[index], sizes[index]);
        
        bool const is_op_code = types[index] <= static_cast<int>(type::OP_2); 
        int const  arity = is_op_code ? opcode_arity(instances[index]).first : 0;
        ++index;
        
        for (int i = 0; i < arity; ++i) self(index, layer + 1);
    }; 

    int index = 0; 
    print_tree_rec(index, 0); 

    assert(index == static_cast<int>(std::size(types)));
}

void DEBUG_check(polish const & candidate, size_t const max_nodes) {
    auto const & [types, instances, sizes] = candidate;

    assert(types.size() == instances.size());
    assert(types.size() == sizes.size());
    assert(sizes[0] == static_cast<int>(types.size()));
    assert(types.size() <= max_nodes);
}

template<typename F, typename... Args>
requires std::is_invocable_r_v<void, F, Args...>
auto measure_time(F&& f, Args&&... args) {
    auto const start = std::chrono::steady_clock::now();

    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);

    auto const end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start);
}

void run_on(std::string_view filename) {
    exporter exporter{filename}; 

    int   constexpr population_size = 100;
    int   constexpr iterations      = 1000;
    float constexpr p_mutation      = 0.25;
    int   constexpr max_nodes       = 10;
    int   constexpr min_const       = -100;
    int   constexpr max_const       = 100;
    int   constexpr no_elites       = 1;

    auto  const      ds              = read_dataset(std::ifstream {std::filesystem::path("../../data") / filename += ".txt"}); 
    int   const     no_variables    = std::size(ds[0]) - 1;

    // leaving it as vector for "weighted" sampling and random access
    std::vector<op_code> active_operators { 
        op_code::SIN,
        op_code::EXP,

        op_code::ADD, 
        op_code::MUL, 
        op_code::DIV 
    }; 

    std::bernoulli_distribution do_mutate(p_mutation);

    std::vector<polish> population;      population.reserve(population_size);
    std::vector<polish> next_population; next_population.reserve(population_size);

    for (int i = 0; i < population_size; ++i) {
        auto [types, instances] = create(max_nodes, min_const, max_const, no_variables, active_operators); 
        auto sizes = get_sizes(types); 

        population.emplace_back(std::move(types), std::move(instances), std::move(sizes));
    }

    for (int i = 0; i < iterations; ++i) {
        auto const errors = evaluate(population, ds, max_nodes);
        next_population.clear();

        auto select = get_fitness_proportional_selection(errors);

        // elitism
        std::vector<std::size_t> indices(errors.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::vector<std::size_t> elite_indices(no_elites);
        std::partial_sort_copy(
            indices.begin(), indices.end(),
            elite_indices.begin(), elite_indices.end(),
            [&](auto a, auto b) {
                return errors[a] < errors[b];
            }
        );
        for (auto index : elite_indices) {
            next_population.push_back(population[index]);
        }

        while (next_population.size() < population_size) {
            auto const parent1 = select(e1);
            auto const parent2 = select(e1);

            auto [child1, child2] = subtree_crossover(
                population[parent1],
                population[parent2],
                max_nodes
            );

            if (do_mutate(e1)) mutate(child1, max_nodes, min_const, max_const, no_variables, active_operators); 
            DEBUG_check(child1, max_nodes); 
            next_population.push_back(std::move(child1));

            if (next_population.size() >= population_size) continue;

            if (do_mutate(e1)) mutate(child2, max_nodes, min_const, max_const, no_variables, active_operators);
            DEBUG_check(child2, max_nodes); 
            next_population.push_back(std::move(child2));
        }
        
        auto best = std::min_element(errors.begin(), errors.end());
        exporter.append_result(i, *best); 
        if (i == iterations - 1) {
            std::println("Generation {}: Best MSE = {}", i, *best);
            auto best_index = std::distance(errors.begin(), best);
            print_tree(population[best_index]);
        }

        population.swap(next_population);
    }

    exporter.export_csv();
}

int main(void) {
    std::string_view files[] {
        "linear",
        "poly", 
        "product",
        "rational",
        "trig",
        "exp",
        "sinxyzq",
        "poly4"
    }; 

    for (auto const & file : files) {
        auto const time = measure_time(run_on, file); 
        print("File {0} took {1}ms", file, time.count()); 
    }

    return 0;
}
