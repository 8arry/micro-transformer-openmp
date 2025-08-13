#include "transformer.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace MicroTransformer
{

    // Feed-Forward Network Implementation
    FeedForwardNetwork::FeedForwardNetwork(const TransformerConfig &config)
        : config_(config),
          W1_(config.embed_dim, config.ff_dim),
          b1_(1, config.ff_dim, 0.0f),
          W2_(config.ff_dim, config.embed_dim),
          b2_(1, config.embed_dim, 0.0f)
    {

        // Initialize weights with Xavier/Glorot initialization
        float limit1 = std::sqrt(6.0f / (config.embed_dim + config.ff_dim));
        float limit2 = std::sqrt(6.0f / (config.ff_dim + config.embed_dim));

        W1_.randomize(-limit1, limit1);
        W2_.randomize(-limit2, limit2);

        // Initialize biases to small random values
        b1_.randomize(-0.01f, 0.01f);
        b2_.randomize(-0.01f, 0.01f);
    }

    Matrix FeedForwardNetwork::forward(const Matrix &input, bool use_parallel)
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

    Matrix FeedForwardNetwork::forward_serial(const Matrix &input)
    {
        // First linear transformation: input * W1 + b1
        Matrix hidden = input * W1_;

        // Add bias
        for (size_t i = 0; i < hidden.rows(); ++i)
        {
            for (size_t j = 0; j < hidden.cols(); ++j)
            {
                hidden(i, j) += b1_(0, j);
            }
        }

        // Apply ReLU activation
        Matrix activated = relu(hidden, false);

        // Second linear transformation: activated * W2 + b2
        Matrix output = activated * W2_;

        // Add bias
        for (size_t i = 0; i < output.rows(); ++i)
        {
            for (size_t j = 0; j < output.cols(); ++j)
            {
                output(i, j) += b2_(0, j);
            }
        }

        return output;
    }

    Matrix FeedForwardNetwork::forward_parallel(const Matrix &input)
    {
        // First linear transformation: input * W1 + b1 (Matrix multiplication is already parallelized)
        Matrix hidden = input * W1_;

// Add bias in parallel
#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < hidden.rows(); ++i)
        {
            for (size_t j = 0; j < hidden.cols(); ++j)
            {
                hidden(i, j) += b1_(0, j);
            }
        }

        // Apply ReLU activation
        Matrix activated = relu(hidden, true);

        // Second linear transformation: activated * W2 + b2
        Matrix output = activated * W2_;

// Add bias in parallel
#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < output.rows(); ++i)
        {
            for (size_t j = 0; j < output.cols(); ++j)
            {
                output(i, j) += b2_(0, j);
            }
        }

        return output;
    }

    Matrix FeedForwardNetwork::relu(const Matrix &input, bool use_parallel) const
    {
        Matrix result(input.rows(), input.cols());

        if (use_parallel)
        {
#pragma omp parallel for
            for (size_t i = 0; i < input.rows() * input.cols(); ++i)
            {
                result.data()[i] = std::max(0.0f, input.data()[i]);
            }
        }
        else
        {
            for (size_t i = 0; i < input.rows() * input.cols(); ++i)
            {
                result.data()[i] = std::max(0.0f, input.data()[i]);
            }
        }

        return result;
    }

    // Layer Normalization Implementation
    LayerNorm::LayerNorm(const TransformerConfig &config)
        : config_(config),
          gamma_(1, config.embed_dim, 1.0f),
          beta_(1, config.embed_dim, 0.0f)
    {
    }

    Matrix LayerNorm::forward(const Matrix &input, bool use_parallel)
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

    Matrix LayerNorm::forward_serial(const Matrix &input)
    {
        Matrix result(input.rows(), input.cols());

        for (size_t i = 0; i < input.rows(); ++i)
        {
            // Compute mean
            float mean = 0.0f;
            for (size_t j = 0; j < input.cols(); ++j)
            {
                mean += input(i, j);
            }
            mean /= static_cast<float>(input.cols());

            // Compute variance
            float variance = 0.0f;
            for (size_t j = 0; j < input.cols(); ++j)
            {
                float diff = input(i, j) - mean;
                variance += diff * diff;
            }
            variance /= static_cast<float>(input.cols());

            // Normalize and apply learnable parameters
            float std_dev = std::sqrt(variance + config_.epsilon);
            for (size_t j = 0; j < input.cols(); ++j)
            {
                float normalized = (input(i, j) - mean) / std_dev;
                result(i, j) = gamma_(0, j) * normalized + beta_(0, j);
            }
        }

        return result;
    }

    Matrix LayerNorm::forward_parallel(const Matrix &input)
    {
        Matrix result(input.rows(), input.cols());

#pragma omp parallel for
        for (size_t i = 0; i < input.rows(); ++i)
        {
            // Compute mean
            float mean = 0.0f;
            for (size_t j = 0; j < input.cols(); ++j)
            {
                mean += input(i, j);
            }
            mean /= static_cast<float>(input.cols());

            // Compute variance
            float variance = 0.0f;
            for (size_t j = 0; j < input.cols(); ++j)
            {
                float diff = input(i, j) - mean;
                variance += diff * diff;
            }
            variance /= static_cast<float>(input.cols());

            // Normalize and apply learnable parameters
            float std_dev = std::sqrt(variance + config_.epsilon);
            for (size_t j = 0; j < input.cols(); ++j)
            {
                float normalized = (input(i, j) - mean) / std_dev;
                result(i, j) = gamma_(0, j) * normalized + beta_(0, j);
            }
        }

        return result;
    }

} // namespace MicroTransformer
