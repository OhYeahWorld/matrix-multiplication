#pragma once

#include <cstddef>

namespace matrix_multiplication {

struct alignas(32) Matrix4 {
    float elements[16]{};

    float& at(std::size_t row, std::size_t column) {
        return elements[column * 4 + (3 - row)];
    }

    float at(std::size_t row, std::size_t column) const {
        return elements[column * 4 + (3 - row)];
    }
};

using Multiply = Matrix4 (*)(const Matrix4&, const Matrix4&);

Matrix4 multiply_row_loop(const Matrix4& left, const Matrix4& right);
Matrix4 multiply_column_loop(const Matrix4& left, const Matrix4& right);
Matrix4 multiply_simd_layout_loop(const Matrix4& left, const Matrix4& right);
Matrix4 multiply_unrolled(const Matrix4& left, const Matrix4& right);

Matrix4 multiply_sse_v1(const Matrix4& left, const Matrix4& right);
Matrix4 multiply_sse_v2(const Matrix4& left, const Matrix4& right);
Matrix4 multiply_sse_v3(const Matrix4& left, const Matrix4& right);
Matrix4 multiply_sse_v4(const Matrix4& left, const Matrix4& right);

Matrix4 multiply_avx_v1(const Matrix4& left, const Matrix4& right);
Matrix4 multiply_avx_v2(const Matrix4& left, const Matrix4& right);
Matrix4 multiply_avx_fma(const Matrix4& left, const Matrix4& right);

bool sse_available();
bool avx_available();
bool avx_fma_available();

}  // namespace matrix_multiplication
