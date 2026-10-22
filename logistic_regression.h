#ifndef CPP_LR_LOGISTIC_REGRESSION_H
#define CPP_LR_LOGISTIC_REGRESSION_H

#include "matrix.h"

#include <cstddef>
#include <vector>

enum class MultiplicationMethod {
    Naive,
    CacheOptimized
};

class LogisticRegression {
public:
    LogisticRegression(double learning_rate = 0.1, std::size_t iterations = 100);

    void train(const Matrix& features,
               const std::vector<double>& labels,
               MultiplicationMethod method = MultiplicationMethod::CacheOptimized);

    std::vector<double> predict_proba(const Matrix& features) const;
    std::vector<int> predict(const Matrix& features, double threshold = 0.5) const;
    double compute_loss(const Matrix& features, const std::vector<double>& labels) const;

    static double sigmoid(double value) noexcept;
    const Matrix& weights() const noexcept { return weights_; }
    double bias() const noexcept { return bias_; }

private:
    double learning_rate_;
    std::size_t iterations_;
    Matrix weights_;
    double bias_ = 0.0;
};

#endif
