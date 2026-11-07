#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <random>
#include <ranges>
#include <type_traits>
#include <vector>
#include <print>

#include "./operators.hpp"
#include "./evaluator.h"
// #include "./evaluator_cuda.cuh"
#include "./dataparser.hpp"
#include "./exporter.hpp"
#include "./evolution.hpp"
#include "./parameters.h"

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

    auto  const     data         = read_dataset(std::ifstream {std::filesystem::path("../../data") / filename += ".txt"}); 
    int   const     no_variables = data.cols - 1;

    gpu_evaluator evaluator(data);
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
        auto const errors = evaluator.evaluate(population);
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
