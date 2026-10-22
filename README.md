# From-Scratch Logistic Regression in C++17

This project implements binary logistic regression with gradient descent using
only the C++ standard library. It is designed to make the data layout and
matrix access patterns visible when profiling CPU cache behavior.

## Design

- `Matrix` stores all values in one row-major `std::vector<double>`.
- `multiply_naive` uses the conventional `i-j-k` loop order. Accessing a
  row of the left matrix and a column of the right matrix makes the right-hand
  access strided in row-major storage.
- `multiply_cache_optimized` uses `i-k-j` loop order. The inner `j` loop walks
  contiguous rows of both the right matrix and the result, improving spatial
  locality and cache-line reuse.
- `LogisticRegression::train` accepts either multiplication method, so the
  benchmark compares the same algorithm and data with only the matrix kernel
  changed.
- A deterministic test executable validates matrix results, sigmoid stability,
  loss reduction, and classification behavior.
- The executable also includes a 256 x 256 square-matrix benchmark, where
  loop interchange has a larger opportunity to improve row-major locality than
  the one-column matrix products used by logistic regression.

## Build and run

Requirements: a C++17 compiler and GNU Make.

```sh
make
./log_reg
```

The demo creates 10,000 samples with 100 features, trains for 30 iterations
with each matrix multiplication implementation, prints elapsed time and final
binary cross-entropy loss, and then benchmarks both kernels on square matrices.

Run the tests with:

```sh
make test
```

Use a clean rebuild when changing compiler flags:

```sh
make clean
make CXXFLAGS="-std=c++17 -O3 -march=native -ffast-math -Wall -Wextra -Wpedantic"
```

## Profiling cache misses with perf

On Linux systems with `perf` permissions enabled:

```sh
perf stat -e cache-misses,cache-references ./log_reg
```

For a broader view of the pipeline and cache hierarchy:

```sh
perf stat -d ./log_reg
```

Run several times or use `perf stat -r 5 ...` when comparing changes because
frequency scaling and system load affect wall-clock measurements.

## Portability

The default `-march=native` flag tunes the executable for the CPU used to
compile it. This is useful for architecture experiments, but remove that flag
when building a portable binary for another machine.

## License

This project is released under the [MIT License](LICENSE).
