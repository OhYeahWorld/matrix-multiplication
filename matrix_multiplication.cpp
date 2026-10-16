#include "matrix_multiplication.hpp"

#include <cmath>

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#define MATRIX_MULTIPLICATION_X86 1
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#define MATRIX_TARGET_SSE __attribute__((target("sse2")))
#define MATRIX_TARGET_AVX __attribute__((target("avx")))
#define MATRIX_TARGET_FMA __attribute__((target("avx,fma")))
#else
#define MATRIX_TARGET_SSE
#define MATRIX_TARGET_AVX
#define MATRIX_TARGET_FMA
#endif
#else
#define MATRIX_MULTIPLICATION_X86 0
#define MATRIX_TARGET_SSE
#define MATRIX_TARGET_AVX
#define MATRIX_TARGET_FMA
#endif

namespace matrix_multiplication {

Matrix4 multiply_row_loop(const Matrix4& left, const Matrix4& right) {
    Matrix4 result;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            for (std::size_t k = 0; k < 4; ++k) {
                result.at(row, column) += left.at(row, k) * right.at(k, column);
            }
        }
    }
    return result;
}

Matrix4 multiply_column_loop(const Matrix4& left, const Matrix4& right) {
    Matrix4 result;
    for (std::size_t column = 0; column < 4; ++column) {
        for (std::size_t row = 0; row < 4; ++row) {
            float value = 0.0f;
            for (std::size_t k = 0; k < 4; ++k) {
                value += left.at(row, k) * right.at(k, column);
            }
            result.at(row, column) = value;
        }
    }
    return result;
}

Matrix4 multiply_simd_layout_loop(const Matrix4& left, const Matrix4& right) {
    Matrix4 result;
    for (std::size_t column = 0; column < 4; ++column) {
        for (std::size_t lane = 0; lane < 4; ++lane) {
            float value = 0.0f;
            for (std::size_t k = 0; k < 4; ++k) {
                value += left.elements[k * 4 + lane] *
                         right.elements[column * 4 + (3 - k)];
            }
            result.elements[column * 4 + lane] = value;
        }
    }
    return result;
}

Matrix4 multiply_unrolled(const Matrix4& m, const Matrix4& n) {
    Matrix4 r;
    r.at(0, 0) = m.at(0, 0)*n.at(0, 0) + m.at(0, 1)*n.at(1, 0) + m.at(0, 2)*n.at(2, 0) + m.at(0, 3)*n.at(3, 0);
    r.at(0, 1) = m.at(0, 0)*n.at(0, 1) + m.at(0, 1)*n.at(1, 1) + m.at(0, 2)*n.at(2, 1) + m.at(0, 3)*n.at(3, 1);
    r.at(0, 2) = m.at(0, 0)*n.at(0, 2) + m.at(0, 1)*n.at(1, 2) + m.at(0, 2)*n.at(2, 2) + m.at(0, 3)*n.at(3, 2);
    r.at(0, 3) = m.at(0, 0)*n.at(0, 3) + m.at(0, 1)*n.at(1, 3) + m.at(0, 2)*n.at(2, 3) + m.at(0, 3)*n.at(3, 3);
    r.at(1, 0) = m.at(1, 0)*n.at(0, 0) + m.at(1, 1)*n.at(1, 0) + m.at(1, 2)*n.at(2, 0) + m.at(1, 3)*n.at(3, 0);
    r.at(1, 1) = m.at(1, 0)*n.at(0, 1) + m.at(1, 1)*n.at(1, 1) + m.at(1, 2)*n.at(2, 1) + m.at(1, 3)*n.at(3, 1);
    r.at(1, 2) = m.at(1, 0)*n.at(0, 2) + m.at(1, 1)*n.at(1, 2) + m.at(1, 2)*n.at(2, 2) + m.at(1, 3)*n.at(3, 2);
    r.at(1, 3) = m.at(1, 0)*n.at(0, 3) + m.at(1, 1)*n.at(1, 3) + m.at(1, 2)*n.at(2, 3) + m.at(1, 3)*n.at(3, 3);
    r.at(2, 0) = m.at(2, 0)*n.at(0, 0) + m.at(2, 1)*n.at(1, 0) + m.at(2, 2)*n.at(2, 0) + m.at(2, 3)*n.at(3, 0);
    r.at(2, 1) = m.at(2, 0)*n.at(0, 1) + m.at(2, 1)*n.at(1, 1) + m.at(2, 2)*n.at(2, 1) + m.at(2, 3)*n.at(3, 1);
    r.at(2, 2) = m.at(2, 0)*n.at(0, 2) + m.at(2, 1)*n.at(1, 2) + m.at(2, 2)*n.at(2, 2) + m.at(2, 3)*n.at(3, 2);
    r.at(2, 3) = m.at(2, 0)*n.at(0, 3) + m.at(2, 1)*n.at(1, 3) + m.at(2, 2)*n.at(2, 3) + m.at(2, 3)*n.at(3, 3);
    r.at(3, 0) = m.at(3, 0)*n.at(0, 0) + m.at(3, 1)*n.at(1, 0) + m.at(3, 2)*n.at(2, 0) + m.at(3, 3)*n.at(3, 0);
    r.at(3, 1) = m.at(3, 0)*n.at(0, 1) + m.at(3, 1)*n.at(1, 1) + m.at(3, 2)*n.at(2, 1) + m.at(3, 3)*n.at(3, 1);
    r.at(3, 2) = m.at(3, 0)*n.at(0, 2) + m.at(3, 1)*n.at(1, 2) + m.at(3, 2)*n.at(2, 2) + m.at(3, 3)*n.at(3, 2);
    r.at(3, 3) = m.at(3, 0)*n.at(0, 3) + m.at(3, 1)*n.at(1, 3) + m.at(3, 2)*n.at(2, 3) + m.at(3, 3)*n.at(3, 3);
    return r;
}

#if MATRIX_MULTIPLICATION_X86

MATRIX_TARGET_SSE
Matrix4 multiply_sse_v1(const Matrix4& left, const Matrix4& right) {
    Matrix4 result;
    const auto* m = reinterpret_cast<const __m128*>(left.elements);
    const auto* n = reinterpret_cast<const __m128*>(right.elements);
    auto* r = reinterpret_cast<__m128*>(result.elements);
    for (int column = 0; column < 4; ++column) {
        const __m128 products0 = _mm_add_ps(
            _mm_mul_ps(m[0], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(3, 3, 3, 3))),
            _mm_mul_ps(m[1], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(2, 2, 2, 2))));
        const __m128 products1 = _mm_add_ps(
            _mm_mul_ps(m[2], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(1, 1, 1, 1))),
            _mm_mul_ps(m[3], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(0, 0, 0, 0))));
        r[column] = _mm_add_ps(products0, products1);
    }
    return result;
}

MATRIX_TARGET_SSE
Matrix4 multiply_sse_v2(const Matrix4& left, const Matrix4& right) {
    Matrix4 result;
    const auto* m = reinterpret_cast<const __m128*>(left.elements);
    const auto* n = reinterpret_cast<const __m128*>(right.elements);
    auto* r = reinterpret_cast<__m128*>(result.elements);
    for (int column = 0; column < 4; ++column) {
        const __m128 a = _mm_mul_ps(m[0], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(3, 3, 3, 3)));
        const __m128 b = _mm_mul_ps(m[1], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(2, 2, 2, 2)));
        const __m128 c = _mm_mul_ps(m[2], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(1, 1, 1, 1)));
        const __m128 d = _mm_mul_ps(m[3], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(0, 0, 0, 0)));
        r[column] = _mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(c, d));
    }
    return result;
}

MATRIX_TARGET_SSE
Matrix4 multiply_sse_v3(const Matrix4& left, const Matrix4& right) {
    Matrix4 result;
    const auto* m = reinterpret_cast<const __m128*>(left.elements);
    const auto* n = reinterpret_cast<const __m128*>(right.elements);
    auto* r = reinterpret_cast<__m128*>(result.elements);
    for (int column = 0; column < 4; ++column) {
        const __m128 a = _mm_mul_ps(m[1], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(2, 2, 2, 2)));
        const __m128 b = _mm_mul_ps(m[3], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(0, 0, 0, 0)));
        const __m128 c = _mm_mul_ps(m[0], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(3, 3, 3, 3)));
        const __m128 d = _mm_mul_ps(m[2], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(1, 1, 1, 1)));
        r[column] = _mm_add_ps(_mm_add_ps(c, a), _mm_add_ps(d, b));
    }
    return result;
}

MATRIX_TARGET_SSE
Matrix4 multiply_sse_v4(const Matrix4& left, const Matrix4& right) {
    Matrix4 result;
    const auto* m = reinterpret_cast<const __m128*>(left.elements);
    const auto* n = reinterpret_cast<const __m128*>(right.elements);
    for (int column = 0; column < 4; ++column) {
        const __m128 a = _mm_add_ps(
            _mm_mul_ps(m[0], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(3, 3, 3, 3))),
            _mm_mul_ps(m[1], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(2, 2, 2, 2))));
        const __m128 b = _mm_add_ps(
            _mm_mul_ps(m[2], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(1, 1, 1, 1))),
            _mm_mul_ps(m[3], _mm_shuffle_ps(n[column], n[column], _MM_SHUFFLE(0, 0, 0, 0))));
        _mm_stream_ps(result.elements + column * 4, _mm_add_ps(a, b));
    }
    _mm_sfence();
    return result;
}

MATRIX_TARGET_AVX
Matrix4 multiply_avx_v1(const Matrix4& left, const Matrix4& right) {
    Matrix4 result;
    const auto* m = reinterpret_cast<const __m128*>(left.elements);
    const float* n = right.elements;
    __m256 mm[4];
    for (int k = 0; k < 4; ++k) {
        mm[k] = _mm256_set_m128(m[k], m[k]);
    }
    for (int column = 0; column < 4; column += 2) {
        const __m256 pair = _mm256_load_ps(n + column * 4);
        const __m256 p0 = _mm256_mul_ps(_mm256_permute_ps(pair, _MM_SHUFFLE(3, 3, 3, 3)), mm[0]);
        const __m256 p1 = _mm256_mul_ps(_mm256_permute_ps(pair, _MM_SHUFFLE(2, 2, 2, 2)), mm[1]);
        const __m256 p2 = _mm256_mul_ps(_mm256_permute_ps(pair, _MM_SHUFFLE(1, 1, 1, 1)), mm[2]);
        const __m256 p3 = _mm256_mul_ps(_mm256_permute_ps(pair, _MM_SHUFFLE(0, 0, 0, 0)), mm[3]);
        const __m256 sum = _mm256_add_ps(_mm256_add_ps(p0, p1), _mm256_add_ps(p2, p3));
        _mm256_stream_ps(result.elements + column * 4, sum);
    }
    _mm_sfence();
    return result;
}

MATRIX_TARGET_AVX
Matrix4 multiply_avx_v2(const Matrix4& left, const Matrix4& right) {
    Matrix4 result;
    const auto* m = reinterpret_cast<const __m128*>(left.elements);
    const float* n = right.elements;
    const __m256 mm0 = _mm256_broadcast_ps(m + 0);
    const __m256 mm1 = _mm256_broadcast_ps(m + 1);
    const __m256 mm2 = _mm256_broadcast_ps(m + 2);
    const __m256 mm3 = _mm256_broadcast_ps(m + 3);
    for (int column = 0; column < 4; column += 2) {
        const __m256 pair = _mm256_load_ps(n + column * 4);
        const __m256 p0 = _mm256_mul_ps(_mm256_permute_ps(pair, _MM_SHUFFLE(3, 3, 3, 3)), mm0);
        const __m256 p1 = _mm256_mul_ps(_mm256_permute_ps(pair, _MM_SHUFFLE(2, 2, 2, 2)), mm1);
        const __m256 p2 = _mm256_mul_ps(_mm256_permute_ps(pair, _MM_SHUFFLE(1, 1, 1, 1)), mm2);
        const __m256 p3 = _mm256_mul_ps(_mm256_permute_ps(pair, _MM_SHUFFLE(0, 0, 0, 0)), mm3);
        const __m256 sum = _mm256_add_ps(_mm256_add_ps(p0, p1), _mm256_add_ps(p2, p3));
        _mm256_stream_ps(result.elements + column * 4, sum);
    }
    _mm_sfence();
    return result;
}

MATRIX_TARGET_FMA
Matrix4 multiply_avx_fma(const Matrix4& left, const Matrix4& right) {
    Matrix4 result;
    const auto* m = reinterpret_cast<const __m128*>(left.elements);
    const float* n = right.elements;
    const __m256 mm0 = _mm256_broadcast_ps(m + 0);
    const __m256 mm1 = _mm256_broadcast_ps(m + 1);
    const __m256 mm2 = _mm256_broadcast_ps(m + 2);
    const __m256 mm3 = _mm256_broadcast_ps(m + 3);
    for (int column = 0; column < 4; column += 2) {
        const __m256 pair = _mm256_load_ps(n + column * 4);
        const __m256 p1 = _mm256_mul_ps(_mm256_permute_ps(pair, _MM_SHUFFLE(2, 2, 2, 2)), mm1);
        const __m256 p3 = _mm256_mul_ps(_mm256_permute_ps(pair, _MM_SHUFFLE(0, 0, 0, 0)), mm3);
        const __m256 sum = _mm256_add_ps(
            _mm256_fmadd_ps(_mm256_permute_ps(pair, _MM_SHUFFLE(3, 3, 3, 3)), mm0, p1),
            _mm256_fmadd_ps(_mm256_permute_ps(pair, _MM_SHUFFLE(1, 1, 1, 1)), mm2, p3));
        _mm256_stream_ps(result.elements + column * 4, sum);
    }
    _mm_sfence();
    return result;
}

bool sse_available() {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse2");
#elif defined(_M_X64)
    return true;
#else
    return false;
#endif
}

bool avx_available() {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx");
#else
    return false;
#endif
}

bool avx_fma_available() {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

#else

Matrix4 multiply_sse_v1(const Matrix4& left, const Matrix4& right) { return multiply_simd_layout_loop(left, right); }
Matrix4 multiply_sse_v2(const Matrix4& left, const Matrix4& right) { return multiply_simd_layout_loop(left, right); }
Matrix4 multiply_sse_v3(const Matrix4& left, const Matrix4& right) { return multiply_simd_layout_loop(left, right); }
Matrix4 multiply_sse_v4(const Matrix4& left, const Matrix4& right) { return multiply_simd_layout_loop(left, right); }
Matrix4 multiply_avx_v1(const Matrix4& left, const Matrix4& right) { return multiply_simd_layout_loop(left, right); }
Matrix4 multiply_avx_v2(const Matrix4& left, const Matrix4& right) { return multiply_simd_layout_loop(left, right); }
Matrix4 multiply_avx_fma(const Matrix4& left, const Matrix4& right) { return multiply_simd_layout_loop(left, right); }

bool sse_available() { return false; }
bool avx_available() { return false; }
bool avx_fma_available() { return false; }

#endif

}  // namespace matrix_multiplication
