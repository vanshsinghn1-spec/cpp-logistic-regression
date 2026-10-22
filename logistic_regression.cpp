#include "logistic_regression.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
constexpr double kProbabilityEpsilon = 1e-15;
}

LogisticRegression::LogisticRegression(double learning_rate, std::size_t iterations)
    : learning_rate_(learning_rate), iterations_(iterations) {
    if (learning_rate <= 0.0 || iterations == 0) {
        throw std::invalid_argument("Learning rate must be positive and iterations must be non-zero");
    }
}

double LogisticRegression::sigmoid(double value) noexcept {
    if (value >= 0.0) {
        const double exponent = std::exp(-value);
        return 1.0 / (1.0 + exponent);
    }
    const double exponent = std::exp(value);
    return exponent / (1.0 + exponent);
}

void LogisticRegression::train(const Matrix& features,
                               const std::vector<double>& labels,
                               MultiplicationMethod method) {
    if (features.rows() != labels.size() || features.rows() == 0 || features.cols() == 0) {
        throw std::invalid_argument("Features and labels have incompatible or empty dimensions");
    }
    for (double label : labels) {
        if (label != 0.0 && label != 1.0) {
            throw std::invalid_argument("Labels must be binary values (0 or 1)");
        }
    }

    weights_ = Matrix(features.cols(), 1, 0.0);
    bias_ = 0.0;
    const Matrix transposed_features = features.transpose();

    for (std::size_t iteration = 0; iteration < iterations_; ++iteration) {
        Matrix logits = method == MultiplicationMethod::Naive
            ? multiply_naive(features, weights_)
            : multiply_cache_optimized(features, weights_);

        for (std::size_t row = 0; row < logits.rows(); ++row) {
            logits(row, 0) += bias_;
            logits(row, 0) = sigmoid(logits(row, 0));
        }

        Matrix errors(logits.rows(), 1);
        for (std::size_t row = 0; row < logits.rows(); ++row) {
            errors(row, 0) = logits(row, 0) - labels[row];
        }

        Matrix gradients = method == MultiplicationMethod::Naive
            ? multiply_naive(transposed_features, errors)
            : multiply_cache_optimized(transposed_features, errors);

        const double sample_count = static_cast<double>(features.rows());
        for (std::size_t feature = 0; feature < weights_.rows(); ++feature) {
            weights_(feature, 0) -= learning_rate_ * gradients(feature, 0) / sample_count;
        }

        double bias_gradient = 0.0;
        for (std::size_t row = 0; row < errors.rows(); ++row) {
            bias_gradient += errors(row, 0);
        }
        bias_ -= learning_rate_ * bias_gradient / sample_count;
    }
}

std::vector<double> LogisticRegression::predict_proba(const Matrix& features) const {
    if (weights_.rows() == 0 || features.cols() != weights_.rows()) {
        throw std::invalid_argument("Model is untrained or feature dimensions do not match");
    }

    Matrix logits = multiply_cache_optimized(features, weights_);
    std::vector<double> probabilities(features.rows());
    for (std::size_t row = 0; row < features.rows(); ++row) {
        probabilities[row] = sigmoid(logits(row, 0) + bias_);
    }
    return probabilities;
}

std::vector<int> LogisticRegression::predict(const Matrix& features, double threshold) const {
    if (threshold < 0.0 || threshold > 1.0) {
        throw std::invalid_argument("Prediction threshold must be in [0, 1]");
    }

    const std::vector<double> probabilities = predict_proba(features);
    std::vector<int> predictions(probabilities.size());
    for (std::size_t row = 0; row < probabilities.size(); ++row) {
        predictions[row] = probabilities[row] >= threshold ? 1 : 0;
    }
    return predictions;
}

double LogisticRegression::compute_loss(const Matrix& features,
                                        const std::vector<double>& labels) const {
    if (features.rows() != labels.size() || features.rows() == 0) {
        throw std::invalid_argument("Features and labels have incompatible or empty dimensions");
    }

    const std::vector<double> probabilities = predict_proba(features);
    double loss = 0.0;
    for (std::size_t row = 0; row < probabilities.size(); ++row) {
        const double probability = std::max(
            kProbabilityEpsilon,
            std::min(probabilities[row], 1.0 - kProbabilityEpsilon));
        loss += labels[row] * std::log(probability)
            + (1.0 - labels[row]) * std::log(1.0 - probability);
    }
    return -loss / static_cast<double>(probabilities.size());
}
