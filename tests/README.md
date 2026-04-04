# Tests

Unit tests for core project components. Tests use hand-rolled assertions (no external testing framework) — each test file contains a `main()` that runs all checks and returns a non-zero exit code on failure.

## Running Tests

```bash
make test
```

This compiles all `tests/test_*.cpp` files against the common source objects and runs each resulting binary. All tests must pass for the command to succeed.

## Test Coverage

| Test File | Component | What It Validates |
|-----------|-----------|-------------------|
| `test_dataset.cpp` | Dataset loading | MNIST IDX file parsing, pixel normalization to [0.0, 1.0], label range [0–9], correct sample/image dimensions |
| `test_loss.cpp` | Activations & loss | ReLU forward/derivative, softmax numerical stability, cross-entropy loss computation, softmax–cross-entropy gradient correctness |
| `test_mlp.cpp` | MLP network | Network creation and destruction, forward pass output dimensions, backward pass gradient accumulation, loss reduction over 50 training steps |

## Adding a New Test

1. Create `tests/test_<name>.cpp` with a `main()` function that returns `0` on success
2. The Makefile discovers test files automatically via `$(wildcard tests/test_*.cpp)` — no Makefile changes needed
3. Run `make test` to build and execute all tests including the new one
