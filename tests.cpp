#include "logistic_regression.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, double tolerance,
                   const std::string& message) {
    require(std::abs(actual - expected) <= tolerance, message);
}

void test_matrix_multiplication() {
    Matrix left(2, 3);
    Matrix right(3, 2);
    left.data() = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    right.data() = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    const Matrix naive = multiply_naive(left, right);
    const Matrix optimized = multiply_cache_optimized(left, right);
    const double expected[] = {58.0, 64.0, 139.0, 154.0};
    for (std::size_t index = 0; index < 4; ++index) {
        require_close(naive.data()[index], expected[index], 1e-12,
                      "Naive multiplication returned an incorrect value");
        require_close(optimized.data()[index], expected[index], 1e-12,
                      "Optimized multiplication returned an incorrect value");
    }
}

void test_sigmoid() {
    require_close(LogisticRegression::sigmoid(0.0), 0.5, 1e-12,
                  "sigmoid(0) must be 0.5");
    require(LogisticRegression::sigmoid(1000.0) > 0.999,
            "sigmoid must saturate safely for large positive values");
    require(LogisticRegression::sigmoid(-1000.0) < 0.001,
            "sigmoid must saturate safely for large negative values");
}

}

int main() {
    try {
        test_matrix_multiplication();
        test_sigmoid();

        Matrix features(6, 1);
        features.data() = {-3.0, -2.0, -1.0, 1.0, 2.0, 3.0};
        const std::vector<double> labels = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
        LogisticRegression model(0.5, 500);
        const double initial_loss = 0.6931471805599453;
        model.train(features, labels);
        require(model.compute_loss(features, labels) < initial_loss,
                "Training must reduce binary cross-entropy loss");
        const std::vector<int> predictions = model.predict(features);
        require(predictions == std::vector<int>({0, 0, 0, 1, 1, 1}),
                "Training must classify the deterministic dataset");
        std::cout << "All tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
