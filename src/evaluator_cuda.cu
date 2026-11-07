#include <vector>
#include <cub/block/block_reduce.cuh>

#include "./operators.hpp"
#include "./parameters.h"
#include "./dataset.h"

#include "./evaluator.h"

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
    int    const * types, 
    double const * instances,
    double const * variables, 
    int    const   length
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
    int    const * types,
    double const * instances,
    int    const * lengths,
    double const * dataset,
    double       * mse,
    int    const dataset_size,
    int    const row_length
) {
    int const individual = blockIdx.x;
    int const tid        = threadIdx.x;

    int    const * tree_types     = types + individual * max_nodes;
    double const * tree_instances = instances + individual * max_nodes;
    int    const   length         = lengths[individual];

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

    double total = BlockReduce(temp_storage).Sum(sse);

    if (tid == 0) {
        mse[individual] = total / dataset_size;
    }
}


// TODO(Architecture): population size, parameters, ...
gpu_evaluator::gpu_evaluator(dataset const & data) 
      : data_rows{data.rows}, data_cols{data.cols}
    {
      // contiguous encoding buffers
      h_types.resize(population_size * max_nodes);
      h_instances.resize(population_size * max_nodes);
      h_lengths.resize(population_size);

      // dataset
      std::size_t const dataset_bytes = data.rows * data.cols * sizeof(double);
      CUDA_CHECK(cudaMalloc(&d_dataset, dataset_bytes));
      cudaMemcpy(d_dataset, data.data.data(), dataset_bytes, cudaMemcpyHostToDevice);

      // population and result
      // size_t const population_size = population_size;
      CUDA_CHECK(cudaMalloc(&d_population_types,     max_nodes * population_size * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&d_population_instances, max_nodes * population_size * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&d_lengths,              population_size * sizeof(int)));

      CUDA_CHECK(cudaMalloc(&d_mse,                  population_size * sizeof(double)));
    }

std::vector<double> gpu_evaluator::evaluate(std::vector<polish> const & population) {
   for(int i = 0; auto const & [types, instances, sizes] : population) {
        assert(types.size() <= max_nodes);
        assert(types.size() == instances.size());
        assert(!sizes.empty());

        std::copy(
            types.begin(), types.end(),
            h_types.begin() + i * max_nodes
        );

        std::copy(
            instances.begin(), instances.end(),
            h_instances.begin() + i * max_nodes
        );

        h_lengths[i] = sizes[0];
	++i;
   }

   CUDA_CHECK(cudaMemcpy(
       d_population_types,
       h_types.data(),
       h_types.size() * sizeof(int),
       cudaMemcpyHostToDevice
   ));
   CUDA_CHECK(cudaMemcpy(
       d_population_instances,
       h_instances.data(),
       h_instances.size() * sizeof(double),
       cudaMemcpyHostToDevice
   ));
   CUDA_CHECK(cudaMemcpy(
       d_lengths,
       h_lengths.data(),
       h_lengths.size() * sizeof(int),
       cudaMemcpyHostToDevice
   ));

   // launch one block per candidate, each thread then evaluates |dataset| / BLOCK_SIZE data points
   // should allow to reuse the stacks a bit, while also making reduction easier
   constexpr int BLOCK_SIZE = 256;
   evaluate_kernel<max_nodes, BLOCK_SIZE><<<population_size, BLOCK_SIZE>>>(
     d_population_types,
     d_population_instances,
     d_lengths,
     d_dataset,
     d_mse,
     data_rows,
     data_cols
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

   return results;
}
    
gpu_evaluator::~gpu_evaluator() {
   CUDA_CHECK(cudaFree(d_population_types));
   CUDA_CHECK(cudaFree(d_population_instances));
   CUDA_CHECK(cudaFree(d_lengths));
   CUDA_CHECK(cudaFree(d_mse));
   CUDA_CHECK(cudaFree(d_dataset));
}

