# hrp-bench-c

C implementation of the **HRP (Hierarchical Risk Parity)** core, part of a
cross-language benchmark suite. The same algorithm is implemented in
[C](https://github.com/suenot/hrp-bench-c),
[C++](https://github.com/suenot/hrp-bench-cpp),
[Zig](https://github.com/suenot/hrp-bench-zig),
[Rust](https://github.com/suenot/hrp-bench-rust),
[Python](https://github.com/suenot/hrp-bench-python),
[Node.js](https://github.com/suenot/hrp-bench-node) and
[Bun](https://github.com/suenot/hrp-bench-bun), all driven by an identical
64-bit-LCG synthetic dataset so the resulting weights are bit-identical.

## Build & run

```sh
make run          # gcc -O3 -o hrp_bench bench.c -lm && ./hrp_bench
```

## What it measures

Five timed stages of the HRP pipeline on `N` assets × 365 daily observations:

| Stage | Complexity |
|---|---|
| Log returns | O(N·T) |
| Covariance | O(N²·T) |
| Average linkage | **O(N³)** ← dominates |
| Quasi-diagonalization | O(N²) |
| Recursive-bisection weights | O(N log N) |

Correlation and distance are computed but not timed. Synthetic price
generation is not timed. Times print as µs / ms / s.
