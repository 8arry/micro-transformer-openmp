#include "transformer.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace MicroTransformer
{

    MultiHeadAttention::MultiHeadAttention(const TransformerConfig &config)
        : config_(config), head_dim_(config.embed_dim / config.num_heads),
          W_q_(config.embed_dim, config.embed_dim),
          W_k_(config.embed_dim, config.embed_dim),
          W_v_(config.embed_dim, config.embed_dim),
          W_o_(config.embed_dim, config.embed_dim)
    {

        if (config.embed_dim % config.num_heads != 0)
        {
            throw std::invalid_argument("embed_dim must be divisible by num_heads");
        }

        // Initialize weights with Xavier/Glorot initialization
        float limit = std::sqrt(6.0f / (config.embed_dim + config.embed_dim));
        W_q_.randomize(-limit, limit);
        W_k_.randomize(-limit, limit);
        W_v_.randomize(-limit, limit);
        W_o_.randomize(-limit, limit);
    }

    Matrix MultiHeadAttention::forward(const Matrix &input, bool use_parallel)
    {
        if (use_parallel)
        {
            return forward_parallel(input);
        }
        else
        {
            return forward_serial(input);
        }
    }

    Matrix MultiHeadAttention::forward_serial(const Matrix &input)
    {
        // Linear transformations to get Q, K, V
        Matrix Q = input * W_q_;
        Matrix K = input * W_k_;
        Matrix V = input * W_v_;

        // Split into multiple heads
        std::vector<Matrix> Q_heads(config_.num_heads, Matrix(config_.seq_length, head_dim_));
        std::vector<Matrix> K_heads(config_.num_heads, Matrix(config_.seq_length, head_dim_));
        std::vector<Matrix> V_heads(config_.num_heads, Matrix(config_.seq_length, head_dim_));

        split_heads(Q, Q_heads);
        split_heads(K, K_heads);
        split_heads(V, V_heads);

        // Apply attention for each head
        std::vector<Matrix> attention_outputs(config_.num_heads, Matrix(config_.seq_length, head_dim_));
        for (size_t h = 0; h < config_.num_heads; ++h)
        {
            attention_outputs[h] = scaled_dot_product_attention(Q_heads[h], K_heads[h], V_heads[h], false);
        }

        // Concatenate heads
        Matrix concat_output = concat_heads(attention_outputs);

        // Final linear transformation
        return concat_output * W_o_;
    }

    Matrix MultiHeadAttention::forward_parallel(const Matrix &input)
    {
        // Linear transformations to get Q, K, V in parallel using sections
        Matrix Q(input.rows(), W_q_.cols());
        Matrix K(input.rows(), W_k_.cols());
        Matrix V(input.rows(), W_v_.cols());

#pragma omp parallel sections
        {
#pragma omp section
            {
                Q = input.multiply_blocked(W_q_);
            }
#pragma omp section
            {
                K = input.multiply_blocked(W_k_);
            }
#pragma omp section
            {
                V = input.multiply_blocked(W_v_);
            }
        }

        // Split into multiple heads
        std::vector<Matrix> Q_heads(config_.num_heads, Matrix(config_.seq_length, head_dim_));
        std::vector<Matrix> K_heads(config_.num_heads, Matrix(config_.seq_length, head_dim_));
        std::vector<Matrix> V_heads(config_.num_heads, Matrix(config_.seq_length, head_dim_));

        split_heads(Q, Q_heads);
        split_heads(K, K_heads);
        split_heads(V, V_heads);

        // Apply attention for each head in parallel
        std::vector<Matrix> attention_outputs(config_.num_heads, Matrix(config_.seq_length, head_dim_));

#pragma omp parallel for if (config_.num_heads > 1)
        for (size_t h = 0; h < config_.num_heads; ++h)
        {
            attention_outputs[h] = scaled_dot_product_attention(Q_heads[h], K_heads[h], V_heads[h], true);
        }

        // Concatenate heads
        Matrix concat_output = concat_heads(attention_outputs);

        // Final linear transformation with blocked multiplication
        return concat_output.multiply_blocked(W_o_);
    }

    Matrix MultiHeadAttention::scaled_dot_product_attention(const Matrix &Q, const Matrix &K, const Matrix &V, bool use_parallel)
    {
        // Compute attention scores: Q * K^T
        Matrix K_T = K.transpose();
        Matrix scores(Q.rows(), K.rows());

        // Manual matrix multiplication for scores with scaling
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

        if (use_parallel)
        {
#pragma omp parallel for collapse(2)
            for (size_t i = 0; i < Q.rows(); ++i)
            {
                for (size_t j = 0; j < K.rows(); ++j)
                {
                    float sum = 0.0f;
                    for (size_t k = 0; k < Q.cols(); ++k)
                    {
                        sum += Q(i, k) * K_T(k, j);
                    }
                    scores(i, j) = sum * scale;
                }
            }
        }
        else
        {
            for (size_t i = 0; i < Q.rows(); ++i)
            {
                for (size_t j = 0; j < K.rows(); ++j)
                {
                    float sum = 0.0f;
                    for (size_t k = 0; k < Q.cols(); ++k)
                    {
                        sum += Q(i, k) * K_T(k, j);
                    }
                    scores(i, j) = sum * scale;
                }
            }
        }

        // Apply softmax to get attention weights
        Matrix attention_weights = softmax(scores, use_parallel);

        // Apply attention weights to values: attention_weights * V
        return attention_weights * V;
    }

    Matrix MultiHeadAttention::softmax(const Matrix &input, bool use_parallel) const
    {
        Matrix result(input.rows(), input.cols());

        if (use_parallel)
        {
#pragma omp parallel for
            for (size_t i = 0; i < input.rows(); ++i)
            {
                // Find max for numerical stability using parallel reduction
                float max_val = input(i, 0);
#pragma omp simd reduction(max : max_val)
                for (size_t j = 1; j < input.cols(); ++j)
                {
                    max_val = std::max(max_val, input(i, j));
                }

                // Compute exponentials and sum using parallel reduction
                float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
                for (size_t j = 0; j < input.cols(); ++j)
                {
                    result(i, j) = std::exp(input(i, j) - max_val);
                    sum += result(i, j);
                }

// Normalize with SIMD vectorization
#pragma omp simd
                for (size_t j = 0; j < input.cols(); ++j)
                {
                    result(i, j) /= sum;
                }
            }
        }
        else
        {
            for (size_t i = 0; i < input.rows(); ++i)
            {
                // Find max for numerical stability
                float max_val = input(i, 0);
                for (size_t j = 1; j < input.cols(); ++j)
                {
                    max_val = std::max(max_val, input(i, j));
                }

                // Compute exponentials and sum
                float sum = 0.0f;
                for (size_t j = 0; j < input.cols(); ++j)
                {
                    result(i, j) = std::exp(input(i, j) - max_val);
                    sum += result(i, j);
                }

                // Normalize
                for (size_t j = 0; j < input.cols(); ++j)
                {
                    result(i, j) /= sum;
                }
            }
        }

        return result;
    }

    void MultiHeadAttention::split_heads(const Matrix &input, std::vector<Matrix> &heads) const
    {
// Parallelize over both attention heads and sequence positions using collapse(2)
// This provides better parallel efficiency for large dimensions
#pragma omp parallel for collapse(2) if (!omp_in_parallel())
        for (size_t h = 0; h < config_.num_heads; ++h)
        {
            for (size_t i = 0; i < config_.seq_length; ++i)
            {
                for (size_t j = 0; j < head_dim_; ++j)
                {
                    heads[h](i, j) = input(i, h * head_dim_ + j);
                }
            }
        }
    }

    Matrix MultiHeadAttention::concat_heads(const std::vector<Matrix> &heads) const
    {
        Matrix result(config_.seq_length, config_.embed_dim);

// Parallelize over both attention heads and sequence positions using collapse(2)
// This provides better parallel efficiency for large dimensions
#pragma omp parallel for collapse(2) if (!omp_in_parallel())
        for (size_t h = 0; h < config_.num_heads; ++h)
        {
            for (size_t i = 0; i < config_.seq_length; ++i)
            {
                for (size_t j = 0; j < head_dim_; ++j)
                {
                    result(i, h * head_dim_ + j) = heads[h](i, j);
                }
            }
        }

        return result;
    }

} // namespace MicroTransformer
