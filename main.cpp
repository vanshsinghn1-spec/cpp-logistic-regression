#include "logistic_regression.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
struct Dataset {
    Matrix features;
    std::vector<double> labels;
};

Dataset make_dataset(std::size_t samples, std::size_t features) {
    Dataset dataset{Matrix(samples, features), std::vector<double>(samples)};
    std::mt19937 generator(42);
    std::uniform_real_distribution<double> feature_distribution(-1.0, 1.0);
    for (std::size_t row = 0; row < samples; ++row) {
        double score = 0.0;
        for (std::size_t col = 0; col < features; ++col) {
            const double value = feature_distribution(generator);
            dataset.features(row, col) = value;
            if (col < 5) {
                score += value * (static_cast<double>(col) + 1.0);
            }
        }
        dataset.labels[row] = LogisticRegression::sigmoid(score) >= 0.5 ? 1.0 : 0.0;
    }
    return dataset;
}

long long benchmark(const Dataset& dataset,
                    MultiplicationMethod method,
                    const std::string& name) {
    constexpr std::size_t iterations = 30;
    LogisticRegression model(0.1, iterations);
    const auto start = std::chrono::steady_clock::now();
    model.train(dataset.features, dataset.labels, method);
    const auto finish = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);

    std::cout << std::left << std::setw(20) << name
              << elapsed.count() << " ms, final loss "
              << std::fixed << std::setprecision(6)
              << model.compute_loss(dataset.features, dataset.labels) << '\n';
    return elapsed.count();
}

void benchmark_square_matrix_multiplication() {
    constexpr std::size_t dimension = 256;
    Matrix left(dimension, dimension);
    Matrix right(dimension, dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t col = 0; col < dimension; ++col) {
            left(row, col) = static_cast<double>((row + col) % 17) / 17.0;
            right(row, col) = static_cast<double>((row * 3 + col) % 19) / 19.0;
        }
    }

    const auto time_kernel = [&](Matrix (*kernel)(const Matrix&, const Matrix&),
                                 const std::string& name) {
        const auto start = std::chrono::steady_clock::now();
        const Matrix result = kernel(left, right);
        const auto finish = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            finish - start);
        double checksum = 0.0;
        for (double value : result.data()) {
            checksum += value;
        }
        std::cout << std::left << std::setw(20) << name
                  << elapsed.count() << " ms, checksum "
                  << std::fixed << std::setprecision(3) << checksum << '\n';
        return elapsed.count();
    };

    std::cout << "\nSquare matrix kernel benchmark (256 x 256):\n";
    const long long naive_time = time_kernel(multiply_naive, "Naive i-j-k:");
    const long long optimized_time = time_kernel(
        multiply_cache_optimized, "Optimized i-k-j:");
    std::cout << "Square-kernel timing difference (optimized - naive): "
              << optimized_time - naive_time << " ms\n";
}
}

int main() {
    constexpr std::size_t sample_count = 10000;
    constexpr std::size_t feature_count = 100;
    const Dataset dataset = make_dataset(sample_count, feature_count);

    std::cout << "Logistic Regression benchmark (" << sample_count << " samples, "
              << feature_count << " features)\n";
    const long long naive_time = benchmark(
        dataset, MultiplicationMethod::Naive, "Naive i-j-k:");
    const long long optimized_time = benchmark(
        dataset, MultiplicationMethod::CacheOptimized, "Optimized i-k-j:");
    std::cout << "Timing difference (optimized - naive): "
              << optimized_time - naive_time << " ms\n";
    benchmark_square_matrix_multiplication();
    std::cout << "Use 'perf stat -e cache-misses,cache-references ./log_reg' for hardware counters.\n";
    return 0;
}
