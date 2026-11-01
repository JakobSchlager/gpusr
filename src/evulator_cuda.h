#ifndef EVALUTOR_CUDA_CU
#define EVALUTOR_CUDA_CU

#include <vector>
#include <cub/cub.h>

#include "./operators.hpp"

#define CUDA_CHECK(call) do {                         \
    cudaError_t err = (call);                         \
    if (err != cudaSuccess) {                         \
        std::fprintf(stderr,                          \
            "CUDA error at %s:%d: %s\n",              \
            __FILE__, __LINE__,                       \
            cudaGetErrorString(err));                 \
        std::abort();                                 \
    }                                                 \
} while (0)

using polish = std::tuple<
    std::vector<int>,    // types  
    std::vector<double>, // instances  
    std::vector<int>     // sizes
>;

template <int MAX_NODES>
__device__ double interpret_gpu(
    int*    const types, 
    double* const instances,
    double* const variables, 
    int     const length
) {
    double stack[MAX_NODES]; 
    size_t sp = 0;

    for(int index = length - 1; index >= 0; --index) {
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

template<int MAX_NODES, int BLOCK_SIZE>
__global__ void evaluate_kernel(
    const int* types,
    const double* instances,
    const int* lengths,
    const double* dataset,
    double* mse,
    int max_nodes,
    int dataset_size,
    int row_length
) {
    int const individual = blockIdx.x;
    int tid = threadIdx.x;

    const int* tree_types = types + individual * max_nodes;
    const double* tree_instances = instances + individual * max_nodes;
    int length = lengths[individual];

    double sse = 0.0;

    for(int row = tid; row < dataset_size; row += BLOCK_SIZE) {
	const double *datapoint = dataset + row * row_length;

	double prediction = interpret_gpu<MAX_NODES>(
	    tree_types,
	    tree_instances,
	    datapoint,
	    length
	); 

	double const error = prediction - datapoint[row_length - 1];
	sse += error * error;
    }

    using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    double total = BlockReduce(temp_storage).Sum(sum);

    if (tid == 0) {
        mse[individual] = total / dataset_size;
    }
}

template<int MAX_NODES>
std::vector<double> evaluate(
    std::vector<polish> const & population, 
    auto                const & dataset
) {
    int    *d_population_types, *d_lengths;
    double *d_population_instances, *d_mse, *d_dataset;

    size_t const population_size = std::size(population);
    cudaMalloc(&d_population_types,     MAX_NODES * population_size * sizeof(int));
    cudaMalloc(&d_population_instances, MAX_NODES * population_size * sizeof(double));
    cudaMalloc(&d_lengths,     population_size * sizeof(int));

    cudaMalloc(&d_mse,                  population_size * sizeof(double)); 

    std::size_t const row_length = std::size(dataset[0]); // n_variables + y
    cudaMalloc(&d_dataset, std::size(dataset) * row_length * sizeof(double));

    for(int i = 0; auto const & [types, instances, sizes] : population) {
    	cudaMemcpy(d_population_types     + i * MAX_NODES, types.data,     std::size(types) * sizeof(int),        cudaMemcpyHostToDevice);
    	cudaMemcpy(d_population_instances + i * MAX_NODES, instances.data, std::size(instances) * sizeof(double), cudaMemcpyHostToDevice);
    	cudaMemcpy(d_lengths + i, sizes.data[0], sizeof(int), cudaMemcpyHostToDevice);
    	++i;
    }

    for(int i = 0; auto const & row : dataset) {
	cudaMemcpy(d_dataset + i++ * row_length, row.data(), row_length * sizeof(double), cudaMemcpyHostToDevice);
    }

    // launch one block per candidate, each thread then evaluates |dataset| / BLOCK_SIZE data points
    // should allow to reuse the stacks a bit, while also making reduction easier
    constexpr int BLOCK_SIZE = 256;

    evaluate_kernel<MAX_NODES, BLOCK_SIZE><<<population_size, BLOCK_SIZE>>>(
	d_population_types,
	d_population_instances,
	d_lengths,
	d_dataset,
	d_mse,
	dataset.size(),
	row_length
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> results(population_size);
    CUDA_CHECK(cudaMemcpy(
        results.data(),
        d_mse,
        population_size * sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(d_population_types));
    CUDA_CHECK(cudaFree(d_population_instances));
    CUDA_CHECK(cudaFree(d_lengths));
    CUDA_CHECK(cudaFree(d_mse));
    CUDA_CHECK(cudaFree(d_dataset));

    return results;
}

#endif
