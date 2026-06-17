# Autodiff Operation Schema

> **AUTO-GENERATED from `OpRegistry.ALL_OPS`.** Do not edit by hand —
> edit `src/main/java/com/yishape/lab/math/codegen/OpRegistry.java` and re-run
> `com.yishape.lab.math.codegen.CodegenTool`. CI rejects an out-of-sync doc.

## Summary

| Metric | Value |
|---|---|
| Total ops (registry) | 88 |
| Leaf / constant nodes (arity 0) | 2 |
| Executable ops (arity ≥ 1) | 86 |
| GPU-supported | 82 |
| HPC-supported | 88 |
| Fusion base ops (`{unary}{Reduce}`) | 19 |
| Compound specials | 6 |

## Operation Matrix

Columns: **Arity** = tensor inputs consumed (0 = leaf/constant, no execution); **GPU** / **HPC** = backend implements the op; **Fused** = derived `{unary}{Reduce}` tags; **Param** = optional scalar parameter name.

### Binary

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `add`  | 2 | ✅ | ✅ | — |  | Element-wise Addition |
| `div`  | 2 | ✅ | ✅ | — |  | Element-wise Division |
| `mul`  | 2 | ✅ | ✅ | `mulMean`, `mulSum` |  | Element-wise Multiplication |
| `sub`  | 2 | ✅ | ✅ | — |  | Element-wise Subtraction |

### Binary scalar

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `addScalar`  | 1 | ✅ | ✅ | — |  | Add Scalar |
| `divScalar`  | 1 | ✅ | ✅ | — |  | Divide by Scalar |
| `mulScalar`  | 1 | ✅ | ✅ | — |  | Multiply by Scalar |
| `rdivScalar`  | 1 | ✅ | ✅ | — |  | Reverse Divide Scalar |
| `rsubScalar`  | 1 | ✅ | ✅ | — |  | Reverse Subtract Scalar |
| `subScalar`  | 1 | ✅ | ✅ | — |  | Subtract Scalar |

### Unary

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `abs`  | 1 | ✅ | ✅ | `absMean`, `absSum` |  | Absolute Value |
| `neg`  | 1 | ✅ | ✅ | — |  | Negate |
| `pow`  | 1 | ✅ | ✅ | `powMean`, `powSum` | `exponent` | Power |
| `reciprocal`  | 1 | — | ✅ | — |  | Reciprocal (1/x) |
| `sqrt`  | 1 | ✅ | ✅ | — |  | Square Root |
| `square`  | 1 | ✅ | ✅ | `squareMean`, `squareSum` |  | Square |

### Activation

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `clamp`  | 1 | ✅ | ✅ | — |  | Clamp |
| `cos`  | 1 | ✅ | ✅ | `cosMean`, `cosSum` |  | Cosine |
| `elu`  | 1 | ✅ | ✅ | `eluMean`, `eluSum` | `alpha` | ELU Activation |
| `exp`  | 1 | ✅ | ✅ | `expMean`, `expSum` |  | Exponential |
| `gelu`  | 1 | ✅ | ✅ | `geluMean`, `geluSum` |  | GELU Activation |
| `hardtanh`  | 1 | ✅ | ✅ | `hardtanhMean`, `hardtanhSum` |  | HardTanh Activation |
| `leakyRelu`  | 1 | ✅ | ✅ | `leakyReluMean`, `leakyReluSum` | `alpha` | Leaky ReLU Activation |
| `log`  | 1 | ✅ | ✅ | `logMean`, `logSum` |  | Natural Logarithm |
| `logSoftmax`  | 1 | ✅ | ✅ | — |  | Log Softmax |
| `mish`  | 1 | ✅ | ✅ | `mishMean`, `mishSum` |  | Mish Activation |
| `normalize`  | 1 | ✅ | ✅ | — |  | L2 Normalize |
| `relu`  | 1 | ✅ | ✅ | `reluMean`, `reluSum` |  | ReLU Activation |
| `selu`  | 1 | ✅ | ✅ | `seluMean`, `seluSum` |  | SELU Activation |
| `sigmoid`  | 1 | ✅ | ✅ | `sigmoidMean`, `sigmoidSum` |  | Sigmoid Activation |
| `silu`  | 1 | ✅ | ✅ | `siluMean`, `siluSum` |  | SiLU/Swish Activation |
| `sin`  | 1 | ✅ | ✅ | `sinMean`, `sinSum` |  | Sine |
| `softmax`  | 1 | ✅ | ✅ | — |  | Softmax |
| `softplus`  | 1 | ✅ | ✅ | `softplusMean`, `softplusSum` | `beta` | Softplus Activation |
| `tan`  | 1 | ✅ | ✅ | — |  | Tangent |
| `tanh`  | 1 | ✅ | ✅ | `tanhMean`, `tanhSum` |  | Tanh Activation |

### Reduce

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `logSumExp`  | 1 | ✅ | ✅ | — |  | Log-Sum-Exp |
| `mean`  | 1 | ✅ | ✅ | — |  | Mean Reduction |
| `sum`  | 1 | ✅ | ✅ | — |  | Sum Reduction |

### Linalg

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `bmm`  | 2 | — | ✅ | — |  | Batch Matrix Multiply |
| `cross`  | 2 | ✅ | ✅ | — |  | Cross Product |
| `dot`  | 2 | ✅ | ✅ | — |  | Dot Product |
| `matmul`  | 2 | ✅ | ✅ | — |  | Matrix Multiplication |
| `mmul`  | 2 | ✅ | ✅ | — |  | Matrix Multiply (alias) |

### View

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `broadcast`  | 1 | ✅ | ✅ | — |  | Broadcast |
| `cat`  | 2 | ✅ | ✅ | — |  | Concatenate |
| `contiguous`  | 1 | ✅ | ✅ | — |  | Contiguous |
| `expand`  | 1 | ✅ | ✅ | — |  | Expand |
| `flatten`  | 1 | ✅ | ✅ | — |  | Flatten |
| `gather`  | 2 | ✅ | ✅ | — |  | Gather |
| `gridSample`  | 1 | ✅ | ✅ | — |  | Grid Sample |
| `interpolate`  | 1 | ✅ | ✅ | — |  | Interpolate |
| `permute`  | 1 | ✅ | ✅ | — |  | Permute |
| `reshape`  | 1 | ✅ | ✅ | — |  | Reshape |
| `scatter`  | 2 | — | ✅ | — |  | Scatter |
| `select`  | 2 | ✅ | ✅ | — |  | Select |
| `slice`  | 1 | ✅ | ✅ | — |  | Slice |
| `squeeze`  | 1 | ✅ | ✅ | — |  | Squeeze |
| `transpose`  | 1 | ✅ | ✅ | — |  | Transpose |
| `unsqueeze`  | 1 | ✅ | ✅ | — |  | Unsqueeze |

### Random

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `dropout`  | 1 | ✅ | ✅ | — |  | Dropout |

### Normalization

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `batchNorm`  | 3 | — | ✅ | — |  | Batch Normalization 1D |
| `batchNorm2d`  | 3 | ✅ | ✅ | — |  | Batch Normalization 2D |
| `groupNorm`  | 3 | ✅ | ✅ | — |  | Group Normalization |
| `instanceNorm`  | 3 | ✅ | ✅ | — |  | Instance Normalization |
| `layerNorm`  | 3 | ✅ | ✅ | — | `epsilon` | Layer Normalization |
| `rmsNorm`  | 2 | ✅ | ✅ | — | `epsilon` | RMS Normalization |

### Dl

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `adaptiveAvgPool2d`  | 1 | ✅ | ✅ | — |  | Adaptive Average Pooling 2D |
| `avgpool2d`  | 1 | ✅ | ✅ | — |  | Average Pooling 2D |
| `conv2d`  | 3 | ✅ | ✅ | — |  | 2D Convolution |
| `convTranspose2d`  | 3 | ✅ | ✅ | — |  | 2D Transposed Convolution |
| `depthwiseConv1d`  | 3 | ✅ | ✅ | — |  | Depthwise 1D Convolution |
| `linear`  | 3 | ✅ | ✅ | — |  | Linear (Fully Connected) |
| `lstmStep`  | 5 | ✅ | ✅ | — |  | LSTM Timestep |
| `maxpool2d`  | 1 | ✅ | ✅ | — |  | Max Pooling 2D |

### Attention

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `mha`  | 3 | ✅ | ✅ | — |  | Multi-Head Attention |
| `scaledDotProductAttention`  | 4 | ✅ | ✅ | — |  | Scaled Dot-Product Attention |

### Ssm

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `selectiveScan`  | 6 | ✅ | ✅ | — |  | Selective Scan (Mamba SSM) |
| `selectiveScan2`  | 6 | ✅ | ✅ | — |  | Selective Scan 2 (Chunked) |
| `trapezoidalScan`  | 6 | ✅ | ✅ | — |  | Trapezoidal Scan (Mamba SSM) |

### Embedding

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `embedding`  | 2 | ✅ | ✅ | — |  | Embedding Lookup |

### Loss

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `bceLoss`  | 2 | ✅ | ✅ | — |  | Binary Cross Entropy Loss |
| `diceLoss`  | 2 | — | ✅ | — |  | Dice Loss |
| `focalLoss`  | 2 | — | ✅ | — |  | Focal Loss |
| `softmaxCrossEntropy`  | 2 | ✅ | ✅ | — |  | Softmax Cross-Entropy Loss |
| `softmaxCrossEntropySparse`  | 2 | ✅ | ✅ | — |  | Sparse Softmax Cross-Entropy Loss |

### Graph

| Tag | Arity | GPU | HPC | Fused variants | Param | Description |
|---|---:|:---:|:---:|---|---|---|
| `constant` 🍃 | 0 | ✅ | ✅ | — |  | Constant |
| `leaf` 🍃 | 0 | ✅ | ✅ | — |  | Leaf Variable |

## Fusion Patterns

Two classes of fused op carry native GPU/HPC implementations beyond the base ops:

### `{unary}{Reduce}` pattern

Any fusion-base unary op composed with `sum` or `mean` reduces to a single fused node named by concatenation (e.g. `square().sum()` → `squareSum`, `exp().mean()` → `expMean`). Built by `OpRegistry.fuseTag(unary, reduce)`.

Fusion-base unary ops (19):

| Unary tag | Fused tags |
|---|---|
| `mul` | `mulMean`, `mulSum` |
| `abs` | `absMean`, `absSum` |
| `square` | `squareMean`, `squareSum` |
| `pow` | `powMean`, `powSum` |
| `exp` | `expMean`, `expSum` |
| `log` | `logMean`, `logSum` |
| `relu` | `reluMean`, `reluSum` |
| `sigmoid` | `sigmoidMean`, `sigmoidSum` |
| `tanh` | `tanhMean`, `tanhSum` |
| `gelu` | `geluMean`, `geluSum` |
| `silu` | `siluMean`, `siluSum` |
| `mish` | `mishMean`, `mishSum` |
| `leakyRelu` | `leakyReluMean`, `leakyReluSum` |
| `elu` | `eluMean`, `eluSum` |
| `selu` | `seluMean`, `seluSum` |
| `softplus` | `softplusMean`, `softplusSum` |
| `hardtanh` | `hardtanhMean`, `hardtanhSum` |
| `sin` | `sinMean`, `sinSum` |
| `cos` | `cosMean`, `cosSum` |

### Compound specials

Loss / reduce compounds that do not follow the `{unary}{Reduce}` pattern. They are excluded from `BASE` and enter `SUPPORTED` via `FusedTagRegistry.<BACKEND>_COMPOUND`.

| Tag | GPU | HPC | Description |
|---|:---:|:---:|---|
| `logSumExp` | ✅ | ✅ | Log-Sum-Exp |
| `softmaxCrossEntropy` | ✅ | ✅ | Softmax Cross-Entropy Loss |
| `softmaxCrossEntropySparse` | ✅ | ✅ | Sparse Softmax Cross-Entropy Loss |
| `bceLoss` | ✅ | ✅ | Binary Cross Entropy Loss |
| `focalLoss` | — | ✅ | Focal Loss |
| `diceLoss` | — | ✅ | Dice Loss |

## Backend Coverage

`SUPPORTED = BASE ∪ <BACKEND>_PATTERN ∪ <BACKEND>_COMPOUND`. `BASE ∩ FUSED = ∅` by construction (compound specials are kept out of BASE).

| Backend | BASE ops | Pattern fused | Compound | Total SUPPORTED |
|---|---:|---:|---:|---:|
| GPU | 78 | 38 | 4 | 120 |
| HPC | 82 | 38 | 6 | 126 |

---
*Generated by `com.yishape.lab.math.codegen.DocGenerator` from `OpRegistry` (88 ops).*
