#include "transformer.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <omp.h>

namespace MicroTransformer
{

    // Performance Benchmark Implementation
    BenchmarkResult PerformanceBenchmark::measure_execution(
        TransformerEncoder &encoder,
        const Matrix &input,
        bool use_parallel,
        size_t num_runs)
    {

        BenchmarkResult result;
        result.config = encoder.get_config();
        result.thread_count = omp_get_max_threads();
        result.implementation_type = use_parallel ? "Parallel" : "Serial";

        // Warm-up run
        Matrix warmup_output = encoder.forward(input, use_parallel);

        // Measure execution time over multiple runs
        auto start_time = std::chrono::high_resolution_clock::now();

        Matrix final_output(input.rows(), input.cols());
        for (size_t run = 0; run < num_runs; ++run)
        {
            final_output = encoder.forward(input, use_parallel);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        result.execution_time_ms = static_cast<double>(duration.count()) / (1000.0 * num_runs);

        // Verify numerical correctness if we have a reference (serial) implementation
        if (use_parallel)
        {
            Matrix serial_output = encoder.forward(input, false);
            result.numerical_correctness = verify_numerical_correctness(serial_output, final_output);

            // Calculate maximum deviation
            result.max_deviation = 0.0;
            for (size_t i = 0; i < serial_output.rows(); ++i)
            {
                for (size_t j = 0; j < serial_output.cols(); ++j)
                {
                    double deviation = std::abs(serial_output(i, j) - final_output(i, j));
                    result.max_deviation = std::max(result.max_deviation, deviation);
                }
            }
        }
        else
        {
            result.numerical_correctness = true;
            result.max_deviation = 0.0;
        }

        return result;
    }

    std::vector<BenchmarkResult> PerformanceBenchmark::scalability_test(
        const TransformerConfig &base_config,
        const std::vector<size_t> &thread_counts,
        const std::vector<size_t> &sequence_lengths,
        size_t num_runs)
    {

        std::vector<BenchmarkResult> results;

        for (size_t seq_len : sequence_lengths)
        {
            TransformerConfig config = base_config;
            config.seq_length = seq_len;

            std::cout << "\nTesting sequence length: " << seq_len << std::endl;

            // Generate random input for this sequence length
            Matrix input = Utils::generate_random_input(seq_len, config.embed_dim);

            // Test serial implementation first
            BenchmarkResult serial_result;
            {
                omp_set_num_threads(1);
                TransformerEncoder encoder(config);
                serial_result = measure_execution(encoder, input, false, num_runs);
                results.push_back(serial_result);

                std::cout << "  Serial: " << std::fixed << std::setprecision(3)
                          << serial_result.execution_time_ms << " ms" << std::endl;
            }

            // Test parallel implementations with different thread counts
            for (size_t thread_count : thread_counts)
            {
                if (thread_count == 1)
                    continue; // Already tested serial

                omp_set_num_threads(static_cast<int>(thread_count));
                TransformerEncoder encoder(config);
                BenchmarkResult parallel_result = measure_execution(encoder, input, true, num_runs);
                results.push_back(parallel_result);

                // Calculate speedup compared to serial result for this sequence length
                double speedup = serial_result.execution_time_ms / parallel_result.execution_time_ms;

                std::cout << "  " << thread_count << " threads: "
                          << std::fixed << std::setprecision(3) << parallel_result.execution_time_ms
                          << " ms (speedup: " << std::setprecision(2) << speedup << "x, "
                          << "correctness: " << (parallel_result.numerical_correctness ? "PASS" : "FAIL")
                          << ", max_dev: " << std::scientific << parallel_result.max_deviation << ")"
                          << std::endl;
            }
        }

        return results;
    }

    bool PerformanceBenchmark::verify_numerical_correctness(
        const Matrix &serial_result,
        const Matrix &parallel_result,
        float tolerance)
    {

        if (serial_result.rows() != parallel_result.rows() ||
            serial_result.cols() != parallel_result.cols())
        {
            return false;
        }

        for (size_t i = 0; i < serial_result.rows(); ++i)
        {
            for (size_t j = 0; j < serial_result.cols(); ++j)
            {
                float diff = std::abs(serial_result(i, j) - parallel_result(i, j));
                if (diff > tolerance)
                {
                    return false;
                }
            }
        }

        return true;
    }

    void PerformanceBenchmark::save_results_to_csv(
        const std::vector<BenchmarkResult> &results,
        const std::string &filename)
    {

        std::ofstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        // Write CSV header
        file << "seq_length,embed_dim,num_heads,ff_dim,num_layers,thread_count,"
             << "implementation_type,execution_time_ms,numerical_correctness,max_deviation\n";

        // Write data
        for (const auto &result : results)
        {
            file << result.config.seq_length << ","
                 << result.config.embed_dim << ","
                 << result.config.num_heads << ","
                 << result.config.ff_dim << ","
                 << result.config.num_layers << ","
                 << result.thread_count << ","
                 << result.implementation_type << ","
                 << std::fixed << std::setprecision(6) << result.execution_time_ms << ","
                 << (result.numerical_correctness ? "true" : "false") << ","
                 << std::scientific << result.max_deviation << "\n";
        }

        file.close();
        std::cout << "Results saved to: " << filename << std::endl;
    }

    // Utility Functions Implementation
    namespace Utils
    {

        void set_thread_count(int num_threads)
        {
            omp_set_num_threads(num_threads);
        }

        int get_thread_count()
        {
            return omp_get_max_threads();
        }

        Matrix generate_random_input(size_t seq_length, size_t embed_dim, float min, float max)
        {
            Matrix input(seq_length, embed_dim);
            input.randomize(min, max);
            return input;
        }

        void print_matrix_stats(const Matrix &matrix, const std::string &name)
        {
            float sum = 0.0f;
            float min_val = matrix(0, 0);
            float max_val = matrix(0, 0);

            for (size_t i = 0; i < matrix.rows(); ++i)
            {
                for (size_t j = 0; j < matrix.cols(); ++j)
                {
                    float val = matrix(i, j);
                    sum += val;
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
            }

            float mean = sum / (matrix.rows() * matrix.cols());

            std::cout << name << " stats:" << std::endl;
            std::cout << "  Shape: " << matrix.rows() << "x" << matrix.cols() << std::endl;
            std::cout << "  Mean: " << std::fixed << std::setprecision(6) << mean << std::endl;
            std::cout << "  Min: " << min_val << std::endl;
            std::cout << "  Max: " << max_val << std::endl;
        }

        double get_wall_time()
        {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = now.time_since_epoch();
            return std::chrono::duration<double>(duration).count();
        }

    } // namespace Utils

} // namespace MicroTransformer
