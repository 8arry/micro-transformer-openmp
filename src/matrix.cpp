#include "transformer.h"
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <omp.h>

namespace MicroTransformer
{

    // Matrix Implementation
    Matrix::Matrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0f)
    {
    }

    Matrix::Matrix(size_t rows, size_t cols, float value)
        : rows_(rows), cols_(cols), data_(rows * cols, value)
    {
    }

    Matrix::Matrix(const Matrix &other)
        : rows_(other.rows_), cols_(other.cols_), data_(other.data_)
    {
    }

    Matrix &Matrix::operator=(const Matrix &other)
    {
        if (this != &other)
        {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = other.data_;
        }
        return *this;
    }

    Matrix::Matrix(Matrix &&other) noexcept
        : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_))
    {
        other.rows_ = 0;
        other.cols_ = 0;
    }

    Matrix &Matrix::operator=(Matrix &&other) noexcept
    {
        if (this != &other)
        {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = std::move(other.data_);
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }

    float &Matrix::operator()(size_t row, size_t col)
    {
        return data_[row * cols_ + col];
    }

    const float &Matrix::operator()(size_t row, size_t col) const
    {
        return data_[row * cols_ + col];
    }

    Matrix Matrix::operator*(const Matrix &other) const
    {
        if (cols_ != other.rows_)
        {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }

        Matrix result(rows_, other.cols_);

// Optimized matrix multiplication with OpenMP
#pragma omp parallel for collapse(2) if (rows_ * other.cols_ * cols_ > 1000)
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < other.cols_; ++j)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < cols_; ++k)
                {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }

        return result;
    }

    Matrix Matrix::operator+(const Matrix &other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
        {
            throw std::invalid_argument("Matrix dimensions don't match for addition");
        }

        Matrix result(rows_, cols_);

#pragma omp parallel for if (rows_ * cols_ > 1000)
        for (size_t i = 0; i < data_.size(); ++i)
        {
            result.data_[i] = data_[i] + other.data_[i];
        }

        return result;
    }

    Matrix Matrix::transpose() const
    {
        Matrix result(cols_, rows_);

#pragma omp parallel for collapse(2) if (rows_ * cols_ > 1000)
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                result(j, i) = (*this)(i, j);
            }
        }

        return result;
    }

    void Matrix::randomize(float min, float max)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min, max);

#pragma omp parallel
        {
            // Each thread has its own random generator to avoid race conditions
            std::mt19937 local_gen(rd() + omp_get_thread_num());
            std::uniform_real_distribution<float> local_dis(min, max);

#pragma omp for
            for (size_t i = 0; i < data_.size(); ++i)
            {
                data_[i] = local_dis(local_gen);
            }
        }
    }

    void Matrix::zero()
    {
#pragma omp parallel for if (data_.size() > 1000)
        for (size_t i = 0; i < data_.size(); ++i)
        {
            data_[i] = 0.0f;
        }
    }

} // namespace MicroTransformer
