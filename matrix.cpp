#include "matrix.h"

#include <stdexcept>

Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
    : rows_(rows), cols_(cols), values_(rows * cols, value) {}

double& Matrix::operator()(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return values_[row * cols_ + col];
}

double Matrix::operator()(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return values_[row * cols_ + col];
}

void Matrix::fill(double value) {
    std::fill(values_.begin(), values_.end(), value);
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t col = 0; col < cols_; ++col) {
            result(col, row) = values_[row * cols_ + col];
        }
    }
    return result;
}

Matrix multiply_naive(const Matrix& left, const Matrix& right) {
    if (left.cols() != right.rows()) {
        throw std::invalid_argument("Matrix dimensions are incompatible");
    }

    Matrix result(left.rows(), right.cols());
    for (std::size_t i = 0; i < left.rows(); ++i) {
        for (std::size_t j = 0; j < right.cols(); ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < left.cols(); ++k) {
                sum += left(i, k) * right(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix multiply_cache_optimized(const Matrix& left, const Matrix& right) {
    if (left.cols() != right.rows()) {
        throw std::invalid_argument("Matrix dimensions are incompatible");
    }

    Matrix result(left.rows(), right.cols());
    for (std::size_t i = 0; i < left.rows(); ++i) {
        for (std::size_t k = 0; k < left.cols(); ++k) {
            const double left_value = left(i, k);
            for (std::size_t j = 0; j < right.cols(); ++j) {
                // In row-major storage, j walks through contiguous elements of
                // both the right-hand row and result row. This improves cache
                // line utilization and avoids the strided right(k, j) access
                // pattern of i-j-k, reducing L1/L2 cache misses.
                result(i, j) += left_value * right(k, j);
            }
        }
    }
    return result;
}
