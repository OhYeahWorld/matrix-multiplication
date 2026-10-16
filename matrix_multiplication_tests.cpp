#include "matrix_multiplication.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

namespace {

using namespace matrix_multiplication;

Matrix4 make_matrix(const std::array<float, 16>& values) {
    Matrix4 result;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            result.at(row, column) = values[row * 4 + column];
        }
    }
    return result;
}

bool matches(const Matrix4& actual, const Matrix4& expected) {
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            const float a = actual.at(row, column);
            const float e = expected.at(row, column);
            const float tolerance = 2.0e-5f * (1.0f + std::fabs(e));
            if (!std::isfinite(a) || std::fabs(a - e) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

void print_matrix(const char* label, const Matrix4& matrix) {
    std::cout << label << '\n';
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            std::cout << matrix.at(row, column) << (column == 3 ? '\n' : '\t');
        }
    }
}

struct TestCase {
    const char* name;
    Multiply multiply;
    bool available;
};

bool run_case(const TestCase& test, const Matrix4& left, const Matrix4& right,
              const Matrix4& expected) {
    if (!test.available) {
        std::cout << "SKIP " << test.name << " (CPU instruction set unavailable)\n";
        return true;
    }
    if (!matches(test.multiply(left, right), expected)) {
        std::cerr << "FAIL " << test.name << '\n';
        return false;
    }
    std::cout << "PASS " << test.name << '\n';
    return true;
}

}  // namespace

int main() {
    const std::array<TestCase, 11> tests{{
        {"row loop", multiply_row_loop, true},
        {"column loop", multiply_column_loop, true},
        {"SIMD layout loop", multiply_simd_layout_loop, true},
        {"unrolled reference", multiply_unrolled, true},
        {"SSE v1", multiply_sse_v1, sse_available()},
        {"SSE v2", multiply_sse_v2, sse_available()},
        {"SSE v3", multiply_sse_v3, sse_available()},
        {"SSE v4 streaming", multiply_sse_v4, sse_available()},
        {"AVX v1", multiply_avx_v1, avx_available()},
        {"AVX v2", multiply_avx_v2, avx_available()},
        {"AVX + FMA", multiply_avx_fma, avx_fma_available()},
    }};

    const Matrix4 left = make_matrix({{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,
    }});
    const Matrix4 right = make_matrix({{
        2.0f, 0.0f, 1.0f, 3.0f,
        1.0f, 2.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 2.0f, 0.0f,
        3.0f, 0.0f, 1.0f, 2.0f,
    }});
    const Matrix4 expected = make_matrix({{
        16.0f, 7.0f, 11.0f, 13.0f,
        40.0f, 19.0f, 27.0f, 37.0f,
        64.0f, 31.0f, 43.0f, 61.0f,
        88.0f, 43.0f, 59.0f, 85.0f,
    }});
    std::cout << "Matrix multiplication example (A x B)\n";
    print_matrix("A:", left);
    print_matrix("B:", right);
    print_matrix("Result:", multiply_unrolled(left, right));

    bool passed = true;
    for (const TestCase& test : tests) {
        passed = run_case(test, left, right, expected) && passed;
    }

    Matrix4 zero;
    Matrix4 identity;
    for (std::size_t index = 0; index < 4; ++index) {
        identity.at(index, index) = 1.0f;
    }
    const Matrix4 zero_expected{};
    const Matrix4 identity_expected = left;
    for (const TestCase& test : tests) {
        if (test.available &&
            (!matches(test.multiply(left, zero), zero_expected) ||
             !matches(test.multiply(left, identity), identity_expected))) {
            std::cerr << "FAIL " << test.name << " on zero/identity cases\n";
            passed = false;
        }
    }

    std::mt19937 generator(0x4d415452u);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    for (int sample = 0; sample < 500 && passed; ++sample) {
        Matrix4 random_left;
        Matrix4 random_right;
        for (std::size_t index = 0; index < 16; ++index) {
            random_left.elements[index] = distribution(generator);
            random_right.elements[index] = distribution(generator);
        }
        const Matrix4 random_expected = multiply_unrolled(random_left, random_right);
        for (const TestCase& test : tests) {
            if (test.available && !matches(test.multiply(random_left, random_right), random_expected)) {
                std::cerr << "FAIL " << test.name << " on randomized sample " << sample << '\n';
                passed = false;
                break;
            }
        }
    }
    if (passed) {
        std::cout << "PASS edge cases and randomized comparisons (500 samples)\n";
        return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
}
