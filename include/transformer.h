#pragma once

#include <vector>
#include <memory>
#include <string>
#include <chrono>

namespace MicroTransformer
{

    // Matrix class for efficient 2D array operations
    class Matrix
    {
    public:
        Matrix(size_t rows, size_t cols);
        Matrix(size_t rows, size_t cols, float value);
        Matrix(const Matrix &other);
        Matrix &operator=(const Matrix &other);
        Matrix(Matrix &&other) noexcept;
        Matrix &operator=(Matrix &&other) noexcept;
        ~Matrix() = default;

        // Accessors
        float &operator()(size_t row, size_t col);
        const float &operator()(size_t row, size_t col) const;
        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        float *data() { return data_.data(); }
        const float *data() const { return data_.data(); }

        // Matrix operations
        Matrix operator*(const Matrix &other) const;
        Matrix multiply_blocked(const Matrix &other) const; // Blocked matrix multiplication for optimization
        Matrix operator+(const Matrix &other) const;
        Matrix transpose() const;
        void randomize(float min = -1.0f, float max = 1.0f);
        void zero();

    private:
        size_t rows_, cols_;
        std::vector<float> data_;
    };

    // Configuration for Transformer model
    struct TransformerConfig
    {
        size_t seq_length = 128;   // Sequence length
        size_t embed_dim = 512;    // Embedding dimension
        size_t num_heads = 8;      // Number of attention heads
        size_t ff_dim = 2048;      // Feed-forward dimension
        size_t num_layers = 6;     // Number of encoder layers
        float dropout_rate = 0.1f; // Dropout rate (not implemented)
        float epsilon = 1e-6f;     // Layer norm epsilon
    };

    // Multi-Head Self-Attention Layer
    class MultiHeadAttention
    {
    public:
        explicit MultiHeadAttention(const TransformerConfig &config);

        Matrix forward(const Matrix &input, bool use_parallel = true);
        Matrix forward_serial(const Matrix &input);
        Matrix forward_parallel(const Matrix &input);

    private:
        TransformerConfig config_;
        size_t head_dim_;

        // Weight matrices
        Matrix W_q_, W_k_, W_v_, W_o_;

        // Helper functions
        Matrix scaled_dot_product_attention(const Matrix &Q, const Matrix &K, const Matrix &V, bool use_parallel = true);
        Matrix softmax(const Matrix &input, bool use_parallel = true) const;
        void split_heads(const Matrix &input, std::vector<Matrix> &heads) const;
        Matrix concat_heads(const std::vector<Matrix> &heads) const;
    };

    // Feed-Forward Network
    class FeedForwardNetwork
    {
    public:
        explicit FeedForwardNetwork(const TransformerConfig &config);

        Matrix forward(const Matrix &input, bool use_parallel = true);
        Matrix forward_serial(const Matrix &input);
        Matrix forward_parallel(const Matrix &input);

    private:
        TransformerConfig config_;
        Matrix W1_, b1_, W2_, b2_;

        Matrix relu(const Matrix &input, bool use_parallel = true) const;
    };

    // Layer Normalization
    class LayerNorm
    {
    public:
        explicit LayerNorm(const TransformerConfig &config);

        Matrix forward(const Matrix &input, bool use_parallel = true);
        Matrix forward_serial(const Matrix &input);
        Matrix forward_parallel(const Matrix &input);

    private:
        TransformerConfig config_;
        Matrix gamma_, beta_;
    };

    // Single Transformer Encoder Layer
    class TransformerEncoderLayer
    {
    public:
        explicit TransformerEncoderLayer(const TransformerConfig &config);

        Matrix forward(const Matrix &input, bool use_parallel = true);
        Matrix forward_serial(const Matrix &input);
        Matrix forward_parallel(const Matrix &input);

    private:
        TransformerConfig config_;
        std::unique_ptr<MultiHeadAttention> attention_;
        std::unique_ptr<FeedForwardNetwork> ffn_;
        std::unique_ptr<LayerNorm> norm1_, norm2_;
    };

    // Complete Transformer Encoder
    class TransformerEncoder
    {
    public:
        explicit TransformerEncoder(const TransformerConfig &config);

        Matrix forward(const Matrix &input, bool use_parallel = true);
        Matrix forward_serial(const Matrix &input);
        Matrix forward_parallel(const Matrix &input);

        const TransformerConfig &get_config() const { return config_; }

    private:
        TransformerConfig config_;
        std::vector<std::unique_ptr<TransformerEncoderLayer>> layers_;
    };

    // Performance measurement utilities
    struct BenchmarkResult
    {
        double execution_time_ms;
        size_t thread_count;
        std::string implementation_type;
        TransformerConfig config;
        bool numerical_correctness;
        double max_deviation;
    };

    class PerformanceBenchmark
    {
    public:
        static BenchmarkResult measure_execution(
            TransformerEncoder &encoder,
            const Matrix &input,
            bool use_parallel = true,
            size_t num_runs = 10);

        static std::vector<BenchmarkResult> scalability_test(
            const TransformerConfig &base_config,
            const std::vector<size_t> &thread_counts,
            const std::vector<size_t> &sequence_lengths,
            size_t num_runs = 5);

        static bool verify_numerical_correctness(
            const Matrix &serial_result,
            const Matrix &parallel_result,
            float tolerance = 1e-4f);

        static void save_results_to_csv(
            const std::vector<BenchmarkResult> &results,
            const std::string &filename);
    };

    // Utility functions
    namespace Utils
    {
        void set_thread_count(int num_threads);
        int get_thread_count();
        Matrix generate_random_input(size_t seq_length, size_t embed_dim, float min = -1.0f, float max = 1.0f);
        void print_matrix_stats(const Matrix &matrix, const std::string &name);
        double get_wall_time();
    }

} // namespace MicroTransformer
