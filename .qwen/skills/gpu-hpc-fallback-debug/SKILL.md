---
name: gpu-hpc-fallback-debug
description: Debug the GPU→HPC→CPU fallback chain in yishape-dl Trainer. When GPU is AVAILABLE but execution silently falls back to HPC or CPU, this skill covers the fallback chain, how to trace which path executed, and common causes for GPU silent failure.
source: auto-skill
extracted_at: '2026-06-04T10:39:59.260Z'
---

# GPU → HPC → CPU Fallback Chain Debugging

## Execution Chain

`Trainer.backwardAndStep()` in `yishape-dl` follows this fallback:

```
GpuGraphExecutor.tryExecute(loss)
  → returns NaN → HpcGraphExecutor.tryExecute(loss)
    → returns NaN → loss.backward() (pure Java CPU)
```

Each level returns `Double.NaN` to signal failure, causing silent fallback to the next level.

## Why "GPU: AVAILABLE" Doesn't Mean GPU Is Used

The backend status banner only checks DLL loading and device initialization. Actual graph execution can still fail at many points:

1. **Unsupported op in graph** — `GpuGraphExecutor.SUPPORTED_OPS` must contain every `opTag` in the computation graph. If any op is missing, `tryExecute()` returns NaN before sending to GPU.

2. **GPU kernel execution failure** — `GpuOptionalRuntime.tryExecuteGraph(json)` calls the Rust GPU DLL via reflection. If any dispatch function returns `Err(...)`, the result is null → NaN fallback.

3. **Graph export failure** — `GraphExporter.toJson()` returns null for degenerate graphs.

4. **Gradient parsing failure** — `applyGradientsFromJson()` catches exceptions silently and returns NaN.

## How to Trace Which Path Executed

### Check output patterns

| Symptom | Likely path |
|---------|-------------|
| No crash, correct loss, fast | GPU executing |
| No crash, correct loss, moderate speed | HPC executing |
| No crash, correct loss, slow | Java CPU fallback |
| Rust panic in `yishape_math_rust` | HPC path |
| Rust panic in `yishape_math_gpu` | GPU path |
| `thread '<unnamed>' panicked` | Either Rust DLL |

### Add logging to trace

In `Trainer.backwardAndStep()` (around the fallback chain):

```java
double gpuLoss = GpuGraphExecutor.tryExecute(rv);
if (!Double.isNaN(gpuLoss)) {
    // GPU path
} else {
    double hpcLoss = HpcGraphExecutor.tryExecute(rv);
    if (!Double.isNaN(hpcLoss)) {
        // HPC path
    } else {
        // CPU path
    }
}
```

Add `log.debug("GPU returned NaN, trying HPC")` between levels to confirm which path runs.

### Enable Rust logging

Set environment variable before running:
```bash
RUST_LOG=debug
```

The GPU Rust crate uses `log::error!()` for kernel failures. The HPC crate uses `eprintln!()` for graph errors.

## Common GPU Failure Causes

### 1. Missing op in SUPPORTED_OPS

Check `GpuGraphExecutor.SUPPORTED_OPS` in `yishape-math`. All DL layer ops must be present:

```
conv2d, maxpool2d, linear, batchNorm2d, softmaxCrossEntropy,
relu, sigmoid, tanh, gelu, flatten, dropout, mha, embedding, ...
```

### 2. JSON serialization issues

- `Infinity` in scalar/param2 fields (missing `isInfinite` guard)
- Subnormal doubles that don't round-trip correctly (rare, Rust Ryu handles them)
- Missing `shape` field for ops that need N-D shape (exportShape not set in Java)

### 3. Buffer size mismatch

If `exportShape` represents input dimensions but the operation produces a larger output (e.g., conv2d with channel expansion), the GPU buffer allocated from `shape.product()` is too small.

### 4. GPU initialization quirks

- `GpuSwitch.disable()` was called somewhere
- System property `-Dyishape.gpu=false`
- `GpuConfig.allowAttempts(n)` exhausted after n failures
- wgpu device selection failed (no Vulkan/DX12 backend)

## The HPC `scalar as u64` Bug Pattern

The most critical bug found in this codebase: HPC Rust's `graph.rs` used `scalar as u64` (numeric truncation) instead of `scalar.to_bits()` (bit reinterpretation) to decode bit-packed double parameters. Since bit-packed doubles are subnormal (~1e-319), `as u64` truncates to 0, making all conv2d/maxpool2d parameters zero.

**Key insight**: GPU Rust used `.to_bits()` correctly, but HPC Rust used `as u64`. When GPU execution silently fails and falls back to HPC, the HPC path crashes with a bug that looks like a GPU problem but is actually an HPC problem.

See the `bit-packed-double-debug` skill for full details.

## Gradient Convergence Debugging

When GPU/HPC graph execution succeeds but training doesn't converge (loss explodes or stays flat), the issue is likely **incorrect gradients** from the Rust backend.

### Diagnostic Approach

Add a `logGradDiag()` method to `Trainer.backwardAndStep()` that:

1. **After** graph execution but **before** `syncGradient()`, snapshot all leaf node gradients from the graph execution
2. Clear those leaf gradients (`leaf.gradient = null`)
3. Run `lossRv.backward()` (CPU backward on the same graph)
4. Compare: saved graph-exec gradients vs fresh CPU gradients
5. Restore graph-exec gradients for the optimizer step

### Key Output Metrics

For each parameter, compute:
- `graphL2` — L2 norm of graph-execution gradients
- `cpuL2` — L2 norm of CPU backward gradients
- `maxAbsDiff` — maximum absolute difference per element
- `maxRelDiff` — maximum relative difference (0-1 scale; 1.0 = completely different sign or magnitude)

### Known Issue: Sum vs Mean Loss

**Critical finding**: Rust GPU/HPC `softmaxCrossEntropy` backward computes gradients for **summed loss** (Σ L_i), while Java `CrossEntropyLoss.backward()` computes gradients for **mean loss** (Σ L_i / B). This causes all gradients to differ by a factor of `batchSize`.

Symptoms:
- `graphL2 / cpuL2 ≈ batchSize` (e.g., 32 for MNIST)
- `maxRelDiff ≈ 2.0` (signs may be opposite due to different scaling)
- First conv layer gets **zero gradients** from GPU (gradient magnitude too small to survive numerical propagation)

### SLF4J NOP Issue

yishape-dl uses SLF4J 2.x but has `slf4j-simple-1.7.x` on the classpath, causing SLF4J to fall into NOP mode (all `YishapeLogger` output is silently discarded). For diagnostic output, use `System.out.println()` instead of `log.info()`.

### Common Gradient Mismatch Patterns

| Pattern | Cause |
|---------|-------|
| All graph grads ≈ 0, CPU grads large | GPU/HPC backward didn't propagate to early layers |
| graphL2 / cpuL2 ≈ batchSize | Sum vs mean loss mismatch in softmaxCrossEntropy |
| graphL2 = 0 for specific params | Leaf node ordering mismatch between Java and Rust |
| maxRelDiff ≈ 1.0 | Sign error or completely wrong gradient values |
| cpuG = null for all params | `backward()` can't run after graph exec (graph state corrupted) |
