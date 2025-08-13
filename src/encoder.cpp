#include "transformer.h"
#include <iostream>

namespace MicroTransformer
{

    // Transformer Encoder Layer Implementation
    TransformerEncoderLayer::TransformerEncoderLayer(const TransformerConfig &config)
        : config_(config),
          attention_(std::make_unique<MultiHeadAttention>(config)),
          ffn_(std::make_unique<FeedForwardNetwork>(config)),
          norm1_(std::make_unique<LayerNorm>(config)),
          norm2_(std::make_unique<LayerNorm>(config))
    {
    }

    Matrix TransformerEncoderLayer::forward(const Matrix &input, bool use_parallel)
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

    Matrix TransformerEncoderLayer::forward_serial(const Matrix &input)
    {
        // Multi-Head Self-Attention with residual connection
        Matrix attention_output = attention_->forward_serial(input);
        Matrix residual1 = input + attention_output;
        Matrix norm1_output = norm1_->forward_serial(residual1);

        // Feed-Forward Network with residual connection
        Matrix ffn_output = ffn_->forward_serial(norm1_output);
        Matrix residual2 = norm1_output + ffn_output;
        Matrix norm2_output = norm2_->forward_serial(residual2);

        return norm2_output;
    }

    Matrix TransformerEncoderLayer::forward_parallel(const Matrix &input)
    {
        // Multi-Head Self-Attention with residual connection
        Matrix attention_output = attention_->forward_parallel(input);
        Matrix residual1 = input + attention_output; // Matrix addition is already parallelized
        Matrix norm1_output = norm1_->forward_parallel(residual1);

        // Feed-Forward Network with residual connection
        Matrix ffn_output = ffn_->forward_parallel(norm1_output);
        Matrix residual2 = norm1_output + ffn_output; // Matrix addition is already parallelized
        Matrix norm2_output = norm2_->forward_parallel(residual2);

        return norm2_output;
    }

    // Complete Transformer Encoder Implementation
    TransformerEncoder::TransformerEncoder(const TransformerConfig &config)
        : config_(config)
    {

        // Create all encoder layers
        layers_.reserve(config.num_layers);
        for (size_t i = 0; i < config.num_layers; ++i)
        {
            layers_.push_back(std::make_unique<TransformerEncoderLayer>(config));
        }

        std::cout << "Initialized Transformer Encoder with:" << std::endl;
        std::cout << "  - " << config.num_layers << " layers" << std::endl;
        std::cout << "  - " << config.num_heads << " attention heads" << std::endl;
        std::cout << "  - " << config.embed_dim << " embedding dimensions" << std::endl;
        std::cout << "  - " << config.seq_length << " sequence length" << std::endl;
        std::cout << "  - " << config.ff_dim << " feed-forward dimensions" << std::endl;
    }

    Matrix TransformerEncoder::forward(const Matrix &input, bool use_parallel)
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

    Matrix TransformerEncoder::forward_serial(const Matrix &input)
    {
        if (input.rows() != config_.seq_length || input.cols() != config_.embed_dim)
        {
            throw std::invalid_argument("Input dimensions don't match configuration");
        }

        Matrix current_output = input;

        // Pass through all encoder layers sequentially
        for (size_t i = 0; i < layers_.size(); ++i)
        {
            current_output = layers_[i]->forward_serial(current_output);
        }

        return current_output;
    }

    Matrix TransformerEncoder::forward_parallel(const Matrix &input)
    {
        if (input.rows() != config_.seq_length || input.cols() != config_.embed_dim)
        {
            throw std::invalid_argument("Input dimensions don't match configuration");
        }

        Matrix current_output = input;

        // Pass through all encoder layers sequentially (layers can't be parallelized as they depend on each other)
        // But each layer's internal operations are parallelized
        for (size_t i = 0; i < layers_.size(); ++i)
        {
            current_output = layers_[i]->forward_parallel(current_output);
        }

        return current_output;
    }

} // namespace MicroTransformer
