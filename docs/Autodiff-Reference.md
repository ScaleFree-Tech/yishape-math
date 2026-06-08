# 自动微分 API 参考 (Autodiff API Reference)

## 概述 / Overview

`AD` 是 `yishape-math` 的自动微分门面类，位于 `com.yishape.lab.math.autodiff` 包。提供反向模式 AD（tape-of-tape 高阶微分）、前向模式 AD（Jacobian/JVP）、混合模式 AD（HVP/Hessian）、VJP/vmap、算子融合、Neural ODE、稀疏/复数/混合精度/Tensor autodiff 等完整功能。

`AD` is the automatic differentiation facade class in `com.yishape.lab.math.autodiff`. Provides reverse-mode AD (tape-of-tape higher-order diffs), forward-mode AD (Jacobian/JVP), mixed-mode AD (HVP/Hessian), VJP/vmap, operator fusion, Neural ODE, sparse/complex/mixed-precision/tensor autodiff.

## 核心接口层次 / Core Interface Hierarchy

```
IDiffVector   — 可微向量（反向模式 AD），extends IDoubleVector
IDiffMatrix   — 可微矩阵，extends IDoubleMatrix
IDiffSparseMatrix — 可微稀疏矩阵
IDiffComplex  — 可微复数向量（Wirtinger 微积分）
IDiffTensor   — 可微张量，extends IDoubleTensor
```

在 `vjp` 子包中：
- `VjpFunction` — 可复用 VJP 算子：`f(g) = J^T @ g`
- `VjpResult` — VJP 结果：`(y, vjpFn)`
- `BatchVjpResult` — 批量 VJP 结果

在 `graph` 子包中：`GraphOptimizer`, `GraphRenderer`, `GraphExporter`, `HpcGraphExecutor`, `GpuGraphExecutor`。

## 核心概念 / Core Concepts

### 反向模式 AD / Reverse-mode AD

```java
var x = AD.vector(new double[]{3.0, 4.0});
var loss = x.pow(2).sum();  // f(x) = Σ x_i²
loss.backward();             // 自动计算 ∂f/∂x = 2x
var grad = x.getGradient();  // [6.0, 8.0]
```

梯度通过拓扑排序从输出节点反向传播至叶子变量。计算图由 `RereDiffVector` 节点构成的有向无环图（DAG）表示。`ThreadLocal` 拓扑排序支持并发 backward。

### 前向模式 AD / Forward-mode AD

```java
// Jacobian-vector product（双数传播）
var tangent = AD.tangent(primal, tangentVec);

// 完整 Jacobian 矩阵（通过 n 次 JVP 构造）
var J = AD.jacobian(fn, x);  // m×n Jacobian
```

### 高阶微分 / Higher-order Differentiation

tape-of-tape 架构：对 `AD.grad()` 的结果再次调用 `grad()`，支持任意阶导数。实现依赖节点上的 `symbolicBackwardFn` 字段。

## AD 门面完整方法列表 / AD Facade Complete Method Reference

### 变量工厂 / Variable Factories

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `AD.vector(double... data)` | `IDiffVector` | 从原始数组创建叶子向量 |
| `AD.vector(IDoubleVector data)` | `IDiffVector` | 从 IDoubleVector 创建（复制） |
| `AD.vector(double scalar)` | `IDiffVector` | 标量叶子变量 |
| `AD.zeros(int size)` | `IDiffVector` | 零向量 |
| `AD.ones(int size)` | `IDiffVector` | 全 1 向量 |
| `AD.matrix(double[][] data)` | `IDiffMatrix` | 从 2D 数组创建叶子矩阵 |
| `AD.matrix(IDoubleMatrix data)` | `IDiffMatrix` | 从 IDoubleMatrix 创建（复制） |
| `AD.matrixZeros(int rows, int cols)` | `IDiffMatrix` | 零矩阵 |
| `AD.matrixOnes(int rows, int cols)` | `IDiffMatrix` | 全 1 矩阵 |
| `AD.sparse(ISparseMatrix data)` | `IDiffSparseMatrix` | 稀疏矩阵叶子变量 |
| `AD.complex(IComplexVector data)` | `IDiffComplex` | 复数向量叶子变量（Wirtinger 导数） |
| `AD.diffFloat(float[] data)` | `IDiffVector` | FP32 前向 / FP64 梯度累积 |
| `AD.diffFloat(IFloatVector data)` | `IDiffVector` | 从 IFloatVector 创建混合精度 |
| `AD.constant(IDoubleVector value)` | `IDiffVector` | 常量节点（不传播梯度） |
| `AD.constant(double scalar)` | `IDiffVector` | 标量常量 |
| `AD.reuseNode(IDiffVector leaf, double[] newData)` | `IDiffVector` | 原地更新叶子数据，避免重建图节点 |

### 自定义操作 / Custom Operations

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `AD.op(CustomOp op, IDiffVector... inputs)` | `IDiffVector` | 应用自定义可微操作（推荐） |
| `AD.registerGradient(String, Function)` | `void` | ⚠️ 已废弃，使用 `CustomOp` 替代 |
| `AD.custom(String, Function, IDiffVector...)` | `IDiffVector` | ⚠️ 已废弃 |
| `AD.unregisterGradient(String)` | `void` | ⚠️ 已废弃 |

`CustomOp` 是自包含的抽象类，嵌入 backward 直接在图节点中：
```java
var op = new CustomOp() {
    protected ForwardResult forward(IDoubleVector[] rawInputs) { ... }
    protected IDoubleVector[] backward(IDoubleVector gradOutput, Object ctx) { ... }
};
var y = AD.op(op, x1, x2);
```

`TensorCustomOp` 类似但操作在 `IDoubleTensor`（带形状）上：
```java
var op = new TensorCustomOp() {
    protected ForwardResult forward(IDoubleTensor[] rawInputs) { ... }
    protected IDoubleTensor[] backward(IDoubleTensor gradOutput, Object ctx) { ... }
};
var y = op.apply(t1, t2);  // returns IDiffTensor
```

### 高阶微分 / Higher-Order Differentiation

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `AD.grad(IDiffVector output, IDiffVector... inputs)` | `IDiffVector[]` | 符号梯度（tape-of-tape，返回可微节点） |

### 算子融合 / Operator Fusion

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `AD.fuse(IDiffVector x)` | `FusedOps` | 开始逐元素融合链（向量） |
| `AD.fuseMatrix(IDiffMatrix x)` | `FusedMatrixOps` | 开始逐元素融合链（矩阵） |
| `AD.elementwise(IDiffVector x, Function<IDiffVector,IDiffVector> fn)` | `IDiffVector` | 自动融合：尝试追踪并融合，失败则回退 eager |

`FusedOps` 构建器支持链式调用：`AD.fuse(x).exp().add(y).mul(2).compute()`。支持 28 种 OpType：EXP, LOG, SQRT, SQUARE, SIGMOID, TANH, RELU, ABS, NEG, POW, ADD_C, SUB_C, MUL_C, DIV_C, RSUB_C, RDIV_C, CLAMP, ADD_V, SUB_V, MUL_V, DIV_V, LEAKY_RELU, ELU, SELU, SILU, MISH, SOFTPLUS, HARDTANH。

### 梯度检查点 / Gradient Checkpointing

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `AD.checkpoint(Function<IDiffVector,IDiffVector> fn, IDiffVector x)` | `IDiffVector` | 前向不存中间激活，反向重计算（O(√n) 内存） |

### Neural ODE

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `AD.odeint(Function<IDiffVector,IDiffVector> dynamics, IDiffVector z0, double t0, double t1, double dt)` | `IDiffVector` | RK4 前向积分 + 伴随法反向传播 |

### 前向模式 AD / Forward-Mode AD

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `AD.tangent(IDiffVector primal, IDiffVector tangent)` | `IDiffVector` | 种子前向模式 AD（JVP 传播） |
| `AD.jacobian(Function<IDiffVector,IDiffVector> fn, IDiffVector x)` | `IDoubleMatrix` | 完整 Jacobian（通过 n 次 JVP） |

### VJP（Vector-Jacobian Product）

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `AD.vjp(Function<IDiffVector,IDiffVector> fn, IDiffVector x)` | `VjpResult` | 计算 VJP 变换，返回可复用 `VjpFunction` |
| `AD.batchVjp(Function<IDiffVector,IDiffVector> fn, List<? extends IDiffVector> xs)` | `BatchVjpResult` | 批量 VJP |

`VjpResult` 包含 `y`（前向输出）和 `vjpFn`（可复用 VJP 算子）。`BatchVjpResult` 包含 `ys[]`、`vjpFns[]`，并提供 `applyAll(g)`、`sumGradients(g)`、`meanGradients(g)` 便捷方法。

### vmap（自动批处理）/ vmap (Automatic Batching)

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `AD.vmap(Function<IDiffVector,IDiffVector> fn, List<? extends IDiffVector> xs)` | `IDiffVector[]` | 对每个样本应用 fn（独立计算图） |
| `AD.vmapSum(Function, List)` | `IDiffVector` | vmap + 求和损失 |
| `AD.vmapMean(Function, List)` | `IDiffVector` | vmap + 平均损失 |
| `AD.vmapT(Function<IDiffTensor,IDiffTensor>, List)` | `IDiffTensor[]` | Tensor 版本 vmap |
| `AD.vmapSumT(Function, List)` | `IDiffTensor` | Tensor 版本 vmap + sum |
| `AD.vmapMeanT(Function, List)` | `IDiffTensor` | Tensor 版本 vmap + mean |

### 图工具 / Graph Tools

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `AD.render(IDiffVector root)` | `String` | 计算图 Graphviz DOT 格式可视化 |
| `AD.render(IDiffMatrix root)` | `String` | 矩阵计算图 DOT 可视化 |
| `AD.dumpGraphJson(IDiffVector root)` | `String` | JSON 调试导出（仅检查；执行路径使用二进制 YSGP） |
| `AD.tryHpcExecute(IDiffVector root)` | `boolean` | 尝试 Rust HPC 执行计算图 |
| `AD.optimize(IDiffVector x)` | `IDiffVector` | 图优化 pass（常量折叠：x+0→x, x×1→x, x×0→0 等） |
| `AD.graphStats(IDiffVector x)` | `GraphOptimizer.GraphStats` | 图统计：`totalNodes`, `leafNodes`, `nonLeafNodes`, `fusibleChains` |

### 梯度校验 / Gradient Checking

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `AD.checkGradient(Function<IDiffVector,IDiffVector> lossFn, IDiffVector x, double tolerance)` | `boolean` | 中心差分梯度校验（通过/失败） |
| `AD.checkGradientDetailed(Function<IDiffVector,IDiffVector> lossFn, IDiffVector x, double tolerance)` | `GradientCheckResult` | 详细校验报告 |

`GradientCheckResult` 记录：`passed`, `maxAbsoluteError`, `maxRelativeError`, `meanAbsoluteError`, `suspiciousIndices`, `analyticalGradient`, `numericalGradient`。提供 `toString()` 单行摘要和 `detailedReport()` 逐元素对比。

### 优化器集成 / Optimizer Integration

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `AD.optimize(IVector initX, Function<IDiffVector,IDiffVector> lossBuilder, IOptimizer optimizer)` | `OptResult` | 一行代码：loss 定义 + 自动梯度 + 批量优化 |
| `AD.autogradOptimizer(IOnlineOptimizer base, BiFunction<IDiffVector,T,IDiffVector> lossBuilder)` | `IOnlineOptimizer` | 包装在线优化器，自动求梯度 |
| `AD.onlineLearn(IVector initParams, List<T> data, BiFunction, IOnlineOptimizer, int epochs)` | `IVector` | 在线学习 + autodiff 一站式训练循环 |

## MixedMode — 混合模式 AD / Mixed-Mode AD

`com.yishape.lab.math.autodiff.MixedMode` 提供 forward-over-reverse 混合模式 AD：

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `MixedMode.hvp(Function<IDiffVector,IDiffVector> fn, IDiffVector x, IDoubleVector v)` | `double[]` | Hessian-vector product：H·v（tape-of-tape 实现） |
| `MixedMode.jvp(Function<IDiffVector,IDiffVector> fn, IDiffVector x, IDoubleVector v)` | `double[]` | Jacobian-vector product：J·v（forward-mode AD） |
| `MixedMode.vjp(Function<IDiffVector,IDiffVector> fn, IDiffVector x, IDoubleVector upstreamGradient)` | `double[]` | Vector-Jacobian product：J^T·g（通过 `AD.vjp()`） |
| `MixedMode.hessian(Function<IDiffVector,IDiffVector> fn, IDiffVector x)` | `IDoubleMatrix` | 完整 Hessian（n 次 HVP 调用，仅 n<100 可行） |
| `MixedMode.jacobianFull(Function<IDiffVector,IDiffVector> fn, IDiffVector x)` | `IDoubleMatrix` | 完整 Jacobian（n 次 JVP 调用，仅 n<100 可行） |

## IDiffVector 接口 / IDiffVector Interface

```java
// === 核心 AD 方法 ===
v.backward()                                  // 反向传播（单位初始梯度）
v.backward(IDoubleVector initialGradient)     // 反向传播（自定义上游梯度）
v.zeroGradient()                              // 清零累积梯度
v.getValue() → IDoubleVector                  // 前向值
v.getGradient() → IDoubleVector               // 累积梯度（null 如果未计算）
v.isLeaf() → boolean                          // 是否为叶子节点
v.grad() → IDiffVector                        // 梯度作为可微节点（tape-of-tape）
v.copy() → IDiffVector                        // 深拷贝

// === 逐元素二元运算（可微变量）===
v.add(IDiffVector other) / v.sub(other) / v.mul(other) / v.div(other)

// === 标量运算 ===
v.add(double scalar) / v.sub(double scalar) / v.mul(double scalar) / v.div(double scalar)
v.rsub(double scalar)         // scalar - v（反向减法）
v.rdiv(double scalar)         // scalar / v（反向除法）

// === 逐元素一元运算 ===
v.neg() / v.exp() / v.log() / v.sqrt() / v.square() / v.abs()
v.sin() / v.cos() / v.tan() / v.tanh() / v.sigmoid() / v.relu()
v.pow(double n)

// === 高级激活函数 ===
v.softmax()                   // Softmax
v.logSoftmax()                // Log-Softmax（数值稳定）
v.gelu()                      // Gaussian Error Linear Unit
v.leakyRelu(double alpha)     // Leaky ReLU
v.elu(double alpha)           // Exponential Linear Unit
v.selu()                      // Scaled ELU
v.silu()                      // Sigmoid Linear Unit (Swish)
v.mish()                      // Mish 激活
v.softplus(double beta)       // Softplus: log(1+exp(beta*x))/beta
v.hardtanh(double minVal, double maxVal)  // Hard Tanh
v.clamp(double min, double max)          // Clamp/clip
v.dropout(double p)           // Dropout（训练时随机置零）

// === 融合层归一化 ===
v.layerNorm(IDiffVector gamma, IDiffVector beta, double eps)

// === 向量运算 ===
v.dot(IDiffVector other) → IDiffVector     // 点积
v.broadcast(int n) → IDiffVector           // 广播标量到长度 n
v.slice(int start, int end) → IDiffVector  // 可微分切片

// === 归约 ===
v.sum() → IDiffVector                      // 求和归约（含模式融合：square().sum(), exp().sum() 等）
v.mean() → IDiffVector                     // 平均归约（含 square().mean() 融合）

// === 就地修改（仅叶子变量）===
v.addInPlace(IDiffVector other)            // 原地加法
v.mulInPlace(double scalar)                // 原地标量乘法
v.divideInPlace(double alpha)              // 原地除法
v.updateData(double[] newData)             // 更新叶子节点数据
```

## IDiffMatrix 接口 / IDiffMatrix Interface

```java
// === 核心 AD 方法 ===
M.backward() / M.backward(IDoubleMatrix initialGradient)
M.zeroGradient() / M.getValue() / M.getGradient() / M.isLeaf() / M.grad()

// === 矩阵乘法 ===
M.matmul(IDiffMatrix other) → IDiffMatrix  // 矩阵-矩阵乘法
M.matmul(IDiffVector v) → IDiffVector      // 矩阵-向量乘法

// === 形状操作 ===
M.transpose() → IDiffMatrix                // 转置
M.flatten() → IDiffVector                  // 展平为向量
M.reshape(int rows, int cols) → IDiffMatrix // 重塑

// === 逐元素运算（可微矩阵）===
M.add(IDiffMatrix) / M.sub(IDiffMatrix) / M.mul(IDiffMatrix) / M.div(IDiffMatrix)

// === 标量运算 ===
M.add(double) / M.sub(double) / M.mul(double) / M.div(double)
M.rsub(double) / M.rdiv(double)

// === 逐元素一元运算 ===
M.neg() / M.pow(double n)
M.exp() / M.log() / M.sigmoid() / M.relu() / M.tanh() / M.sqrt() / M.square() / M.abs()

// === 归约 ===
M.sum() → IDiffMatrix                      // 全矩阵求和（含模式融合）
M.mean() → IDiffMatrix                     // 全矩阵平均
M.sum(int axis) → IDiffVector              // 沿轴求和（axis=0: 列方向, axis=1: 行方向）
M.max(int axis) → IDiffVector              // 沿轴求最大值

// === 广播运算 ===
M.sub(IDiffVector vec, int axis) → IDiffMatrix  // 沿轴广播减法（如 bias 减法）
M.div(IDiffVector vec, int axis) → IDiffMatrix  // 沿轴广播除法

// === 融合损失 ===
M.softmaxCrossEntropy(IDiffMatrix oneHotLabels) → IDiffVector  // 融合 Softmax + 交叉熵（数值稳定）

// === 就地修改（仅叶子变量）===
M.addInPlace(IDiffMatrix) / M.mulInPlace(double)
```

## IDiffSparseMatrix 接口 / IDiffSparseMatrix Interface

```java
// === 核心 AD 方法 ===
A.backward() / A.backward(ISparseMatrix initialGradient)
A.zeroGradient() / A.getValue() / A.getGradient() / A.isLeaf() / A.grad()

// === 稀疏运算 ===
A.matmul(IDiffVector v) → IDiffVector      // 稀疏 @ 稠密向量（核心操作）
A.add(IDiffSparseMatrix other)             // 逐元素加法
A.mul(double scalar)                       // 标量乘法
A.transpose()                              // 转置

// === 归约 ===
A.sum() → IDiffVector                      // 求和归约
A.mean() → IDiffVector                     // 平均归约
```

## IDiffComplex 接口 / IDiffComplex Interface

```java
// === 核心 AD 方法 ===
z.backward() / z.backward(IComplexVector initialGradient)
z.zeroGradient() / z.getValue() / z.getGradient() / z.isLeaf() / z.grad()

// === 复数运算 ===
z.add(IDiffComplex) / z.sub(IDiffComplex) / z.mul(IDiffComplex) / z.div(IDiffComplex)
z.scale(Complex scalar)                    // 标量乘法
z.conjugate()                              // 共轭

// === 逐元素数学运算 ===
z.exp() / z.log() / z.sin() / z.cos() / z.tan()
z.tanh() / z.sigmoid() / z.relu()
z.abs() / z.sqrt() / z.square() / z.neg() / z.pow(double n)

// === 归约与内积 ===
z.sum() → Complex                          // 求和
z.innerProduct(IDiffComplex other) → Complex  // 内积
```

Wirtinger 导数自动处理：`∂f/∂z = (∂f/∂x - i·∂f/∂y)/2`。共轭对称反向规则（如 `mul` backward 使用 `conjugate()`）。

## IDiffTensor 接口 / IDiffTensor Interface

可微张量，将 `IDiffVector`（展平值）与 `RereDoubleTensor`（形状元数据）组合。所有 AD 委托给底层展平向量。

```java
// === 核心 AD 方法 ===
t.backward() / t.backward(IDoubleTensor gradient)
t.zeroGradient() / t.requiresGrad() / t.setRequiresGrad(boolean)
t.detach() → IDoubleTensor                 // 从计算图分离
t.grad() → IDoubleTensor                   // 梯度
t.flattenGrad() → IDiffVector              // 展平梯度（用于优化器）
t.flattenValue() → IDiffVector             // 展平值（用于优化器）

// === 工厂方法 ===
IDiffTensor.fromDiffVector(IDiffVector vec, int... shape)
IDiffTensor.fromTensor(IDoubleTensor tensor, boolean requiresGrad)

// === 形状/视图操作（共享底层向量）===
t.reshape(int...) / t.permute(int...) / t.transpose(int, int)
t.squeeze(int) / t.unsqueeze(int)
t.slice(int dim, int start, int end) / t.narrow(int dim, int start, int length)
t.select(int dim, int index)
t.expand(int...) / t.flatten() / t.contiguous() / t.tile(int...) / t.broadcastTo(int...)

// === 逐元素运算（全部可微）===
t.add(IDoubleTensor) / t.sub / t.mul / t.div
t.add(double) / t.sub(double) / t.mul(double) / t.div(double)
t.neg() / t.abs() / t.sqrt() / t.exp() / t.log()
t.sin() / t.cos() / t.tan()
t.tanh() / t.silu() / t.gelu() / t.softplus(double)
t.mish() / t.elu(double) / t.leakyRelu(double) / t.selu() / t.hardtanh(double,double)
t.sigmoid() / t.relu() / t.square() / t.pow(double) / t.clamp(double,double)
t.dropout(double p)

// === Softmax ===
t.softmax(int dim)                         // 沿维度 softmax（数值稳定）
t.logSoftmax(int dim)                      // Log-Softmax

// === 归约（可微，支持 dim + keepdim）===
t.sum(int dim, boolean keepdim) / t.mean(int dim, boolean keepdim)
t.max(int dim, boolean keepdim) / t.min(int dim, boolean keepdim)
t.prod(int dim, boolean keepdim)
t.sumAll() / t.meanAll() / t.maxAll() / t.minAll() / t.prodAll()
t.cumsum(int dim) / t.cumprod(int dim)
t.argmax(int dim) / t.argmin(int dim)
t.std(int dim) / t.var(int dim)

// === 线性代数 ===
t.mmul(IDoubleTensor other) → IDiffTensor  // 2D 矩阵乘法（可微）
t.bmm(IDoubleTensor other) → IDiffTensor   // 3D 批量矩阵乘法（可微）

// === 拼接 ===
t.cat(int dim, IDoubleTensor... others) → IDiffTensor  // 沿维度拼接（可微）

// === 就地修改 ===
t.add_(IDoubleTensor) / t.sub_ / t.mul_ / t.div_
t.fill_(double) / t.copy_(IDoubleTensor)
```

以下操作返回非可微结果（detach）：`gather`, `scatter`, `scatterAdd`, `where`, `topk`, `pad`, `unfold`, `nonzero`, `maskedSelect`, `maskedFill`, `stack`, `normalize`, `einsum`。

## 与优化器配合 / Integration with Optimizers

`AD.optimize()` 支持所有批量优化器（`Opts.lbfgs()`、`Opts.conjugateGradient()`、`Opts.dfp()`、`Opts.steepestDescent()`）和在线优化器（通过 `AD.autogradOptimizer()` / `AD.onlineLearn()` 包装 `Opts.onlineAdam()`, `Opts.onlineSGD()` 等）。

## 性能特性 / Performance Characteristics

- **模式融合**：`square().sum()`、`exp().sum()`、`pow(N).sum()`、`mul(x).sum()`、`square().mean()` 在 `RereDiffVector`/`RereDiffMatrix` 的 `sum()`/`mean()` 中自动检测并融合为单一反向节点
- **JIT 融合**：`AD.fuse()` 将逐元素链编译为单次前向/反向 kernel
- **自动融合**：`AD.elementwise()` 追踪运算，可融合则融合，不可则回退 eager
- **混合精度**：`AD.diffFloat()` 提供 FP32 前向 + FP64 梯度累积，适合显存受限场景
- **检查点**：`AD.checkpoint()` 用重计算换内存（O(√n) 内存）
- **HPC 桥接**：二进制 YSGP 协议（`tryGpuExecute`/`tryHpcExecute`）→ Rust 执行（28 种算子支持），不可用时自动回退 Java SIMD → SISD。JSON 调试导出：`AD.dumpGraphJson()`
- **GPU 桥接**：`GpuGraphExecutor`（25 种算子，执行路径开发中）
- **并发安全**：`ThreadLocal` 拓扑排序支持多线程同时 backward

---

**自动微分** — 告别手写梯度，让微分自动完成！