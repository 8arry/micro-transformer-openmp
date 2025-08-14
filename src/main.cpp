#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <omp.h>
#include "transformer.h"

using namespace MicroTransformer;

void print_header()
{
    std::cout << "================================================================" << std::endl;
    std::cout << "           Micro Transformer with OpenMP Parallelization       " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "C++ Standard: " << __cplusplus << std::endl;
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
    std::cout << "Max threads available: " << omp_get_max_threads() << std::endl;
    std::cout << "================================================================" << std::endl
              << std::endl;
}

void demonstrate_basic_functionality()
{
    std::cout << "=== Basic Functionality Demonstration ===" << std::endl;

    // Create a small configuration for demonstration
    TransformerConfig config;
    config.seq_length = 32;
    config.embed_dim = 128;
    config.num_heads = 4;
    config.ff_dim = 512;
    config.num_layers = 2;

    std::cout << "Creating Transformer with configuration:" << std::endl;
    std::cout << "  Sequence Length: " << config.seq_length << std::endl;
    std::cout << "  Embedding Dim: " << config.embed_dim << std::endl;
    std::cout << "  Number of Heads: " << config.num_heads << std::endl;
    std::cout << "  Feed-Forward Dim: " << config.ff_dim << std::endl;
    std::cout << "  Number of Layers: " << config.num_layers << std::endl
              << std::endl;

    // Create transformer
    TransformerEncoder encoder(config);

    // Generate random input
    std::cout << "Generating random input..." << std::endl;
    Matrix input = Utils::generate_random_input(config.seq_length, config.embed_dim, -0.5f, 0.5f);
    Utils::print_matrix_stats(input, "Input");
    std::cout << std::endl;

    // Test serial implementation
    std::cout << "Running serial forward pass..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    Matrix serial_output = encoder.forward(input, false);
    auto end = std::chrono::high_resolution_clock::now();
    auto serial_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    Utils::print_matrix_stats(serial_output, "Serial Output");
    std::cout << "Serial execution time: " << std::fixed << std::setprecision(3) << serial_time << " ms" << std::endl
              << std::endl;

    // Test parallel implementation
    std::cout << "Running parallel forward pass..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    Matrix parallel_output = encoder.forward(input, true);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    Utils::print_matrix_stats(parallel_output, "Parallel Output");
    std::cout << "Parallel execution time: " << std::fixed << std::setprecision(3) << parallel_time << " ms" << std::endl;

    // Calculate speedup and verify correctness
    double speedup = serial_time / parallel_time;
    bool correct = PerformanceBenchmark::verify_numerical_correctness(serial_output, parallel_output, 1e-4f);

    std::cout << "Speedup: " << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "Numerical correctness: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;
}

void run_comprehensive_benchmark()
{
    std::cout << "=== Comprehensive Performance Benchmark ===" << std::endl;

    // Base configuration
    TransformerConfig base_config;
    base_config.embed_dim = 256;
    base_config.num_heads = 8;
    base_config.ff_dim = 1024;
    base_config.num_layers = 3;

    // Test different thread counts
    std::vector<size_t> thread_counts = {1, 2, 4, 8};

    // Test different sequence lengths
    std::vector<size_t> sequence_lengths = {64, 128, 256};

    std::cout << "Testing configurations:" << std::endl;
    std::cout << "  Thread counts: ";
    for (size_t tc : thread_counts)
        std::cout << tc << " ";
    std::cout << std::endl;
    std::cout << "  Sequence lengths: ";
    for (size_t sl : sequence_lengths)
        std::cout << sl << " ";
    std::cout << std::endl
              << std::endl;

    // Run scalability test
    std::vector<BenchmarkResult> results = PerformanceBenchmark::scalability_test(
        base_config, thread_counts, sequence_lengths, 5);

    // Save results to CSV
    std::string filename = "benchmark_results_" +
                           std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                                              std::chrono::system_clock::now().time_since_epoch())
                                              .count()) +
                           ".csv";
    PerformanceBenchmark::save_results_to_csv(results, filename);

    std::cout << std::endl
              << "Comprehensive benchmark completed!" << std::endl;
}

void run_detailed_component_test()
{
    std::cout << "=== Detailed Component Testing ===" << std::endl;

    TransformerConfig config;
    config.seq_length = 64;
    config.embed_dim = 256;
    config.num_heads = 8;
    config.ff_dim = 1024;
    config.num_layers = 1; // Test single layer for detailed analysis

    Matrix input = Utils::generate_random_input(config.seq_length, config.embed_dim);

    // Test individual components
    std::cout << "Testing Multi-Head Attention..." << std::endl;
    MultiHeadAttention attention(config);

    auto start = std::chrono::high_resolution_clock::now();
    Matrix attn_serial = attention.forward_serial(input);
    auto end = std::chrono::high_resolution_clock::now();
    auto attn_serial_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    start = std::chrono::high_resolution_clock::now();
    Matrix attn_parallel = attention.forward_parallel(input);
    end = std::chrono::high_resolution_clock::now();
    auto attn_parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    std::cout << "  Serial: " << std::fixed << std::setprecision(3) << attn_serial_time << " ms" << std::endl;
    std::cout << "  Parallel: " << attn_parallel_time << " ms" << std::endl;
    std::cout << "  Speedup: " << std::setprecision(2) << attn_serial_time / attn_parallel_time << "x" << std::endl;
    std::cout << "  Correctness: " << (PerformanceBenchmark::verify_numerical_correctness(attn_serial, attn_parallel) ? "PASS" : "FAIL") << std::endl
              << std::endl;

    std::cout << "Testing Feed-Forward Network..." << std::endl;
    FeedForwardNetwork ffn(config);

    start = std::chrono::high_resolution_clock::now();
    Matrix ffn_serial = ffn.forward_serial(input);
    end = std::chrono::high_resolution_clock::now();
    auto ffn_serial_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    start = std::chrono::high_resolution_clock::now();
    Matrix ffn_parallel = ffn.forward_parallel(input);
    end = std::chrono::high_resolution_clock::now();
    auto ffn_parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    std::cout << "  Serial: " << std::fixed << std::setprecision(3) << ffn_serial_time << " ms" << std::endl;
    std::cout << "  Parallel: " << ffn_parallel_time << " ms" << std::endl;
    std::cout << "  Speedup: " << std::setprecision(2) << ffn_serial_time / ffn_parallel_time << "x" << std::endl;
    std::cout << "  Correctness: " << (PerformanceBenchmark::verify_numerical_correctness(ffn_serial, ffn_parallel) ? "PASS" : "FAIL") << std::endl
              << std::endl;
}

int main()
{
    print_header();

    try
    {
        // Demonstrate basic functionality
        demonstrate_basic_functionality();

        // Run detailed component tests
        run_detailed_component_test();

        // Run comprehensive benchmark
        run_comprehensive_benchmark();

        std::cout << "================================================================" << std::endl;
        std::cout << "           All tests completed successfully!                    " << std::endl;
        std::cout << "================================================================" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
