#ifndef CPP_LR_MATRIX_H
#define CPP_LR_MATRIX_H

#include <cstddef>
#include <vector>

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double value = 0.0);

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }

    double& operator()(std::size_t row, std::size_t col);
    double operator()(std::size_t row, std::size_t col) const;

    void fill(double value);
    std::vector<double>& data() noexcept { return values_; }
    const std::vector<double>& data() const noexcept { return values_; }

    Matrix transpose() const;

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> values_;
};

Matrix multiply_naive(const Matrix& left, const Matrix& right);
Matrix multiply_cache_optimized(const Matrix& left, const Matrix& right);

#endif
