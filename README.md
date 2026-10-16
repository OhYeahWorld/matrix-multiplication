# Matrix multiplication examples

This standalone C++17 project implements the multiplication examples from the
adjacent article. `Matrix4` exposes logical `(row, column)` indexing while
storing each column in reverse lane order, matching the article's SIMD layout.

Implemented variants:

- Row-ordered loop, column-ordered loop, SIMD-layout loop, and an unrolled
  reference implementation.
- SSE v1-v4, AVX v1-v2, and AVX+FMA implementations on x86.

The test harness calls hardware SIMD implementations only when the running CPU
and operating system support the required instruction set. Callers should
check the corresponding `*_available()` function before invoking an x86 SIMD
entry point. On non-x86 platforms the SIMD entry points use the scalar
SIMD-layout loop and report the instruction set as unavailable; hardware-
specific test cases are skipped.

Build and run:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The tests compare each available implementation with the unrolled reference
using fixed examples, identity/zero cases, and 500 deterministic random matrix
pairs. Floating-point comparisons use a relative tolerance to account for
different addition and FMA rounding.
