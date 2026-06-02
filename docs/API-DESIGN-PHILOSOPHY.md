# yishape-math API 设计哲学与最佳使用实践

> **面向对象**：使用本库进行数学计算的 Agent 和人类开发者  
> **目标**：避免盲目猜测 API 使用方式，减少反复修改造成的 token 消耗

---

## 目录

1. [核心设计哲学](#1-核心设计哲学)
2. [十大入口类架构](#2-十大入口类架构)
3. [接口与实现分离原则](#3-接口与实现分离原则)
4. [Rere* 命名规范的含义](#4-rere-命名规范的含义)
5. [泛型类型系统](#5-泛型类型系统)
6. [工厂方法使用规范](#6-工厂方法使用规范)
7. [优化器使用模式](#7-优化器使用模式)
8. [机器学习模块使用模式](#8-机器学习模块使用模式)
9. [向量索引模块使用模式](#9-向量索引模块使用模式)
10. [信号处理模块使用模式](#10-信号处理模块使用模式)
11. [时间序列分析模块使用模式](#11-时间序列分析模块使用模式)
12. [绘图模块使用模式](#12-绘图模块使用模式)
13. [常见错误与避免方法](#13-常见错误与避免方法)
14. [性能注意事项](#14-性能注意事项)

---

## 1. 核心设计哲学


### 1.1 委托链设计

本库采用**三层委托链**架构：

```
┌─────────────────────────────────────────────────────────────┐
│  用户代码                                                   │
│  Linalg.matrix(...) / ML.dr.pca(...) / Opts.lbfgs()       │
└─────────────────────┬─────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  入口类（工厂方法）                                           │
│  Linalg / ML / Opts / Stats                                 │
│  提供静态工厂方法，统一入口点                                  │
└─────────────────────┬─────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  接口层（抽象）                                              │
│  IMatrix / IVector / IOptimizer / IClassifier               │
│  定义规范，保证互换性                                         │
└─────────────────────┬─────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  实现层（具体算法）                                           │
│  RereDoubleMatrix / RereLBFGS / RerePCA                     │
│  自主研发实现                                                │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 十大入口类

| 门面入口类 / Facade class | 包路径 / package | 职责 / Functions |
|--------|--------|------|
| `Linalg` | `com.yishape.lab.math.linalg` | 线性代数：矩阵/向量创建、分解、求解 /Linear Algebra: Matrix/Vector Creation, Decomposition, and Solution |
| `DataFrame` | `com.yishape.lab.math.data` | 数据框操作：结构化数据处理、CSV文件读写、数据切片 / Data Frame Operations: Structured Data Processing, CSV file read/write, Flexible data slicing.|
| `Stats` | `com.yishape.lab.math.stats` | 统计学：概率分布、假设检验、参数估计 / Statistics: Probability Distribution, Hypothesis Testing, Parameter Estimation |
| `ML` | `com.yishape.lab.math.ml` | 机器学习：分类、回归、降维、聚类、度量学习 / Machine Learning: Classification, Regression, Dimensionality Reduction, Clustering, Metric Learning |
| `Opts` | `com.yishape.lab.math.optimize` | 优化：无约束优化、线性规划、在线优化 / Optimization: Unconstrained optimization, linear programming, online optimization |
| `AD` | `com.yishape.lab.math.autodiff` | 自动微分：反向/前向模式、高阶微分、Neural ODE / Automatic Differentiation: reverse/forward mode, higher-order, Neural ODE |
| `Signals` | `com.yishape.lab.math.signal` | 信号处理：生成、滤波、变换、分析 / Signal processing: generation, filtering, transformation, analysis|
| `TSA` | `com.yishape.lab.math.timeseries` | 时间序列：预测、分解、滤波、协整分析 / Time series: Prediction, decomposition, filtering, cointegration analysis |
| `Plots` | `com.yishape.lab.math.plot` | 绘图：静态工厂创建 `IPlot`，支持 JavaFX/ECharts/SVG 后端 / Drawing: Static factory creates `IPlot`, supporting JavaFX/ECharts/SVG backends|
| `VI` | `com.yishape.lab.math.vecidx` | 向量索引：最近邻搜索（hnsw、LSH、KD-Tree等）/ Vector Index: Nearest Neighbor Search (hnsw, LSH, KD-Tree, etc.) |


## 2. 十大入口类架构

### 2.1 Linalg - 线性代数入口

**位置**：`com.yishape.lab.math.linalg.Linalg`

**核心原则**：
- 所有矩阵/向量通过 `Linalg` 工厂方法创建
- `of` 方法重命名为 `matrix`/`vector` 方法
- 默认类型为 `Double`

```java
// ✅ 正确：通过 Linalg 工厂创建
IMatrix<Double> A = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
IVector<Double> v = Linalg.vector(new double[]{1, 2, 3});

// ✅ 正确：使用便捷方法
IMatrix<Double> zeros = Linalg.zeros(3, 3);
IMatrix<Double> eye = Linalg.eye(3);
IVector<Double> range = Linalg.range(0, 10, 2);  // [0, 2, 4, 6, 8]

// ✅ 正确：稀疏矩阵
ISparseMatrix sparse = Linalg.sparseFromCOO(rowIdx, colIdx, values, rows, cols);

// ❌ 错误：直接实例化实现类
// IMatrix<Double> A = new RereDoubleMatrix(...);
```

### 2.2 DataFrame - 数据框入口

**位置**：`com.yishape.lab.math.data.DataFrame`

**核心原则**：
- 通过 `new DataFrame()` 创建空数据框，再用 `addColumn(Column)` 填充数据
- 通过 `DataFrame.readCsv()` 静态工厂方法从 CSV 文件读取
- 数据框操作（切片、缺失值处理、统计摘要等）均在实例方法上完成

```java
import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.data.Column;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;

// ✅ 正确：从 CSV 文件读取（唯一静态工厂方法）
DataFrame df = DataFrame.readCsv("data.csv");

// ✅ 正确：手动创建 DataFrame
DataFrame df = new DataFrame();
Column col1 = new Column();
col1.setName("A");
col1.getData().add(1.0);
col1.getData().add(2.0);
col1.getData().add(3.0);
df.addColumn(col1);

Column col2 = new Column();
col2.setName("B");
col2.getData().add(4.0);
col2.getData().add(5.0);
col2.getData().add(6.0);
df.addColumn(col2);

// ✅ 正确：读取带分隔符、无表头的 CSV
DataFrame df2 = DataFrame.readCsv("data.tsv", "\t", false);

// ✅ 正确：导出为 CSV
df.toCsv("output.csv");

// ✅ 正确：转换为矩阵
IMatrix<Double> mat = df.toMatrix();

// ✅ 正确：数值统计摘要
Map<String, Double> stats = df.describe();

// ✅ 正确：缺失值处理
DataFrame cleaned = df.dropNa();       // 删除含缺失值的行
DataFrame filled = df.fillNa(0.0);     // 缺失值填充为 0

// ✅ 正确：转换为向量（数值列）
IVector<Double> vec = df.getColumns().get(0).toVec();
```

### 2.3 Stats - 统计学入口

**位置**：`com.yishape.lab.math.stats.Stats`

**核心原则**：
- 分布对象通过 `Stats.xxx()` 静态方法创建
- 支持连续分布和离散分布

```java
// ✅ 正确：创建分布对象
NormalDistribution norm = Stats.norm(0.0, 1.0);  // 标准正态分布
StudentDistribution t = Stats.t(10.0);          // t 分布
PoissonDistribution poi = Stats.poisson(2.5);    // 泊松分布

// ✅ 正确：使用分布方法
double pdf = norm.pdf(0.5);      // 概率密度
double cdf = norm.cdf(0.5);     // 累积分布
double sample = norm.sample();   // 随机采样

// ✅ 正确：统计计算
double corr = Stats.corr(x, y);  // 相关系数
double cov = Stats.cov(x, y);    // 协方差
```

### 2.4 ML - 机器学习入口

**位置**：`com.yishape.lab.math.ml.ML`

**核心原则**：
- 使用静态 Wrapper 字段组织子模块
- 每个子模块提供流式 API（链式调用）

```java
// ✅ 正确：分类
IClassifier clf = ML.clf.logisticRegression();
IClassifier rf = ML.clf.randomForest();

// ✅ 正确：降维（fit-transform 模式）
IMatrix<Double> Z = ML.dr.pca(2).fitTransform(X);

// ✅ 正确：聚类
var clusters = ML.clu.kMeans(3).fit(X).getResult();

// ✅ 正确：预处理
var scaler = ML.preproc.standardScaler().fit(X);
IMatrix<Double> Xscaled = scaler.transform(X);
```

### 2.5 Opts - 优化入口

**位置**：`com.yishape.lab.math.optimize.Opts`

**核心原则**：
- 优化器通过 `Opts.lbfgs()` 等静态方法创建
- 目标函数和梯度通过接口传递

```java
// ✅ 正确：创建优化器
IOptimizer optimizer = Opts.lbfgs();

// ✅ 正确：定义目标函数和梯度
IObjectiveFunction objFun = x -> x.dot(x);  // f(x) = x·x
IGradientFunction grdFun = x -> x.scale(2); // ∇f(x) = 2x

// ✅ 正确：执行优化
OptResult result = optimizer.optimize(initX, objFun, grdFun);
```

### 2.6 Plots - 绘图入口

**位置**：`com.yishape.lab.math.plot.Plots`

**核心原则**：
- 与 `Linalg`、`Stats` 相同：**只通过 `Plots` 静态方法创建图表**
- `Plots.of()` / `Plots.line(...)` 返回 `IPlot`，后续链式调用在接口上完成
- 后端切换：`Plots.setProvider(PlotProvider.Echarts)` 或 `Plots.ofEcharts()` / `ofJavaFx()` / `ofSvg()`

```java
// ✅ 正确
Plots.line(x, y).title("示例").show();

// ❌ 错误：IPlot 是接口，不是工厂入口
// IPlot p = new JavaFxPlot();  // 应使用 Plots.of() 或 Plots.ofJavaFx()
```

---

## 3. 接口与实现分离原则

### 3.1 核心接口一览

| 接口 | 路径 | 说明 |
|------|------|------|
| `IMatrix<T>` | `linalg/IMatrix.java` | 泛型矩阵接口 |
| `IVector<T>` | `linalg/IVector.java` | 泛型向量接口 |
| `IDoubleMatrix` | `linalg/IDoubleMatrix.java` | Double 矩阵 |
| `IFloatMatrix` | `linalg/IFloatMatrix.java` | Float 矩阵 |
| `IOptimizer` | `optimize/IOptimizer.java` | 优化器接口 |
| `IObjectiveFunction` | `optimize/IObjectiveFunction.java` | 目标函数 |
| `IGradientFunction` | `optimize/IGradientFunction.java` | 梯度函数 |
| `IClassifier` | `ml/clf/IClassifier.java` | 分类器接口 |
| `IVecIdx` | `vecidx/IVecIdx.java` | 向量索引接口 |

### 3.2 为什么使用接口

**好处**：
1. **互换性**：可以轻松切换实现（如 `RereLBFGS` → `RustLBFGS`）
2. **可测试性**：便于 mock 测试
3. **稳定性**：用户代码依赖接口而非实现

```java
// ✅ 正确：使用接口类型
IMatrix<Double> A = Linalg.eye(3);
IVector<Double> v = Linalg.zeros(3);

// ❌ 错误：依赖具体实现类
// RereDoubleMatrix A = new RereDoubleMatrix(...);
```

---

## 4. Rere* 命名规范的含义

### 4.1 什么是 Rere*

`Rere*` 是本库发起者网名RereMouse的缩写，基本为本库初始**自主实现**的标识，意为 "RereMouse-implemented"（由RereMouse实现）。

### 4.2 Rere* 系列分类

| 类别 | 示例 | 说明 |
|------|------|------|
| **Rere* 向量/矩阵** | `RereDoubleVector`, `RereFloatMatrix` | 底层数据结构 |
| **Rere* 分解** | `RereSVDDecomp`, `RereQRDecomposition` | 矩阵分解算法 |
| **Rere* 优化** | `RereLBFGS`, `RereSimplexLinProgSolver` | 优化算法 |
| **Rere* ML** | `RerePCA`, `RereKnn`, `RereTSNE` | 机器学习算法 |
| **Rere* 信号** | `RereFFT`, `RereDCT` | 信号处理算法 |

### 4.3 Rere* vs Rust*

部分功能同时提供 Java 实现（Rere*）和 Rust/HPC 加速实现（Rust*）：

```java
// Java 实现（纯 Java，可在任何环境运行）
IOptimizer javaLbfgs = Opts.lbfgs(LBFGSType.Java);  // 或 Opts.lbfgs()

// Rust/HPC 加速实现（需要原生库支持，自动回退）
IOptimizer rustLbfgs = Opts.lbfgs(LBFGSType.Rust);
```

---

## 5. 泛型类型系统

### 5.1 支持的类型

本库主要支持两种数值类型：

| 类型 | 接口 | 默认工厂方法 |
|------|------|-------------|
| `Double` | `IDoubleMatrix`, `IDoubleVector` | `Linalg.matrix(...)` 返回 `IDoubleMatrix` |
| `Float` | `IFloatMatrix`, `IFloatVector` | `Linalg.matrix(float[][], Float.class)` 返回 `IFloatMatrix` |

### 5.2 类型推断规则

```java
// ✅ 正确：double[] 推断为 IDoubleMatrix
IMatrix<Double> A = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});

// ✅ 正确：明确指定 Float 类型
IMatrix<Float> B = Linalg.matrix(new float[][]{{1f, 2f}, {3f, 4f}});

// ❌ 错误：类型不匹配
// IMatrix<Double> C = Linalg.matrix(floatData);  // 编译错误
```

### 5.3 类型转换

```java
// ✅ 正确：向量类型转换
IVector<Double> doubleVec = Linalg.vector(new double[]{1, 2, 3});
IVector<Float> floatVec = doubleVec.toFloatVector();

// ✅ 正确：矩阵类型转换
IMatrix<Double> doubleMat = Linalg.eye(3);
IMatrix<Float> floatMat = /* 需要手动转换 */;
```

---

## 6. 工厂方法使用规范

### 6.1 Linalg 工厂方法速查

#### 矩阵创建

```java
// 从数组创建
Linalg.matrix(double[][] data)    → IDoubleMatrix
Linalg.matrix(float[][] data)     → IFloatMatrix

// 特殊矩阵
Linalg.zeros(int rows, int cols)  → IDoubleMatrix（全零）
Linalg.ones(int rows, int cols)   → IDoubleMatrix（全一）
Linalg.eye(int size)              → IDoubleMatrix（单位阵）
Linalg.diag(double[] values)      → IDoubleMatrix（对角阵）

// 随机矩阵
Linalg.rand(int rows, int cols)   → IDoubleMatrix（均匀分布 [0,1)）
Linalg.randn(int rows, int cols)   → IDoubleMatrix（标准正态分布）

// 稀疏矩阵
Linalg.sparse(double[][] data)           → ISparseMatrix
Linalg.sparseFromCOO(...)               → ISparseMatrix
Linalg.sparseFromCSR(...)              → ISparseMatrix
Linalg.sparseFromCSC(...)              → ISparseMatrix
```

#### 向量创建

```java
// 从数组创建
Linalg.vector(double[] data)      → IDoubleVector
Linalg.vector(float[] data)       → IFloatVector

// 范围向量
Linalg.range(int end)             → IDoubleVector（0 到 end，步长 1）
Linalg.range(int start, int end)  → IDoubleVector
Linalg.range(int start, int end, int step) → IDoubleVector

// 特殊向量
Linalg.ones(int len)              → IDoubleVector
Linalg.zeros(int len)             → IDoubleVector
Linalg.linspace(double start, double stop, int num) → IDoubleVector
```

#### 线性代数运算

```java
// 求解线性方程组
Linalg.solve(A, b)                → IVector<Double>
Linalg.lstsq(A, b)               → Tuple2<IVector<Double>, Double>

// 特殊矩阵
Linalg.lowerTriMatrix(int m)      → IDoubleMatrix
Linalg.upperTriMatrix(int m)      → IDoubleMatrix
Linalg.tridiagonalMatrix(...)     → IDoubleMatrix
Linalg.blockDiagonalMatrix(...)    → IMatrix<Double>
```

### 6.2 Stats 工厂方法速查

#### 连续分布

```java
Stats.norm()                      → NormalDistribution（标准正态 N(0,1)）
Stats.norm(double mean, double std) → NormalDistribution（一般正态）
Stats.t(double dof)               → StudentDistribution
Stats.t(double dof, double loc, double scale) → StudentDistribution
Stats.uniform(double a, double b) → UniformDistribution
Stats.exponential(double rate)    → ExponentialDistribution
Stats.chi2(double dof)            → Chi2Distribution
Stats.f(double d1, double d2)     → FDistribution
Stats.beta(double a, double b)    → BetaDistribution
Stats.gamma(double shape, double scale) → GammaDistribution
```

#### 离散分布

```java
Stats.bernoulli(double p)         → BernoulliDistribution
Stats.binomial(int n, double p)   → BinomialDistribution
Stats.discreteUniform(int a, int b) → DiscreteUniformDistribution
Stats.geometric(double p)         → GeometricDistribution
Stats.negativeBinomial(int r, double p) → NegativeBinomialDistribution
Stats.poisson(double lambda)      → PoissonDistribution
```

#### 统计计算

```java
Stats.corr(x, y)                  → double（皮尔逊相关系数）
Stats.cov(x, y)                   → double（协方差）
Stats.estimator                   → ParameterEstimation（参数估计）
Stats.tester                      → HypothesisTesting（假设检验）
Stats.anova                       → ANOVA（方差分析）
```

---

## 7. 优化器使用模式

### 7.1 优化器创建

```java
// 无约束优化器
Opts.lbfgs()                      → IOptimizer（L-BFGS）
Opts.lbfgs(double tol, int maxIter) → IOptimizer（带参数）
Opts.conjugateGradient()           → IOptimizer（共轭梯度法）
Opts.dfp()                        → IOptimizer（DFP 拟牛顿法）
Opts.steepestDescent()            → IOptimizer（最速下降法）

// 在线优化器
Opts.onlineAdam()                 → IOnlineOptimizer
Opts.onlineSGD()                  → IOnlineOptimizer

// 线性/整数规划
Opts.linProgSolver()              → ILinProgSolver（默认 HiGHS）
Opts.simplexLinProgSolver()       → ILinProgSolver（单纯形法）
Opts.intLinProgSolver()           → IIntegerProg（整数规划）
```

### 7.2 目标函数定义

```java
// 使用 Lambda 表达式
IObjectiveFunction obj = x -> x.dot(x) - 5.0;

// 使用匿名内部类
IObjectiveFunction obj = new IObjectiveFunction() {
    @Override
    public double computeObjective(IVector x) {
        return x.dot(x);
    }
};

// 复合目标函数
IObjectiveFunction regularized = x -> loss(x) + lambda * x.dot(x);
```

### 7.3 梯度函数定义

```java
// 使用 Lambda 表达式
IGradientFunction grad = x -> x.scale(2.0);

// 使用匿名内部类
IGradientFunction grad = new IGradientFunction() {
    @Override
    public IVector computeGradient(IVector x) {
        return x.scale(2.0);
    }
};
```

### 7.4 执行优化

```java
// 1. 创建优化器
IOptimizer optimizer = Opts.lbfgs(1e-6, 100);

// 2. 定义初始点
IVector<Double> initX = Linalg.zeros(10);

// 3. 定义目标函数和梯度
IObjectiveFunction obj = x -> /* ... */;
IGradientFunction grad = x -> /* ... */;

// 4. 执行优化
OptResult result = optimizer.optimize(initX, obj, grad);

// 5. 获取结果
if (result.isConverged()) {
    IVector<Double> optimal = result.getOptimalPoint();
    double value = result.getOptimalValue();
}
```

### 7.5 OptResult 结果解读

```java
OptResult result = optimizer.optimize(initX, objFun, gradFun);

// 基本信息
result.getOptimalPoint();          // 最优点
result.getOptimalValue();          // 最优函数值
result.isConverged();              // 是否收敛
result.getConvergenceReason();     // 收敛原因
result.getIterations();            // 迭代次数

// 统计信息
result.getExecutionTimeMs();       // 执行时间（毫秒）
result.getFunctionEvaluations();   // 函数评估次数
result.getGradientEvaluations();   // 梯度评估次数

// 历史记录
result.getFunctionValueHistory();  // 函数值历史
result.getGradientNormHistory();   // 梯度范数历史

// 验证
result.validate();                 // 结果验证
result.getSummary();               // 格式化摘要
```

---

## 8. 机器学习模块使用模式

### 8.1 分类（ML.clf）

```java
// 逻辑回归
IClassifier lr = ML.clf.logisticRegression();

// 随机森林
IClassifier rf = ML.clf.randomForest();

// 决策树
IClassifier dt = ML.clf.decisionTree();

// 线性 SVM
IClassifier svm = ML.clf.linearSvm();

// XGBoost
IClassifier xgb = ML.clf.xGboost();

// K近邻
IClassifier knn = ML.clf.kNN(5);

// 集成分类器
IClassifier ensemble = ML.clf.ensembleClassifier(
    EnsembleClassifier.EnsembleStrategy.Bagging, 42L);

// 训练和预测
IClassifier clf = ML.clf.logisticRegression().fit(Xtrain, ytrain);
int[] predictions = clf.predict(Xtest);
double score = clf.score(Xtest, ytest);
```

### 8.2 回归（ML.reg）

```java
// 线性回归
IRegression reg = ML.reg.linear(0.0, 0.1);  // (正则化参数, 初始学习率)

// 训练和预测
IRegression reg = ML.reg.linear(0.0, 0.1).fit(Xtrain, ytrain);
double[] predictions = reg.predict(Xtest);
```

### 8.3 降维（ML.dr）

```java
// PCA
var pca = ML.dr.pca(2);  // 降到 2 维
IMatrix<Double> Z = pca.fitTransform(X);

// SVD 降维
var svd = ML.dr.svd(2);

// t-SNE
var tsne = ML.dr.tsne(2);

// UMAP
var umap = ML.dr.umap(2);
```

### 8.4 聚类（ML.clu）

```java
// K-Means
var kmeans = ML.clu.kMeans(3);
var result = kmeans.fit(X).getResult();
int[] labels = result.getClusters();  // 或 result.getResult()

// 高斯混合模型
var gmm = ML.clu.gaussianMixture(3);
var result = gmm.fit(X).getResult();
```

### 8.5 预处理（ML.preproc）

```java
// 标准化（均值0，方差1）
var scaler = ML.preproc.standardScaler();
IMatrix<Double> Xscaled = scaler.fitTransform(X);

// 归一化（Min-Max）
var normalizer = ML.preproc.minMaxScaler();
IMatrix<Double> Xnorm = normalizer.fitTransform(X);

// 缺失值填充
var imputer = ML.preproc.imputer().strategy(Imputer.Strategy.MEAN);
IMatrix<Double> Xfilled = imputer.fitTransform(X);

// 多项式特征
var poly = ML.preproc.polynomialFeatures(2);
IMatrix<Double> Xpoly = poly.fitTransform(X);
```

### 8.6 距离度量学习（ML.dml）

```java
// 监督度量学习
var metricLearner = ML.dml.ddml().fit(X, y);  // X: 特征, y: 标签
IDisMetric metric = metricLearner.getResult();

// 使用学到的度量进行最近邻搜索（通过 VI 工厂）
VecSearchOption options = new VecSearchOption().indexType(IdxType.HNSW);
IDoubleVecIdx index = VI.buildDouble(vectors, ids, metric, options);
int[] neighbors = index.search(query, k);
```

---

## 9. 向量索引模块使用模式

### 9.1 VI 工厂入口

**位置**：`com.yishape.lab.math.vecidx.VI`

**核心原则**：
- 使用 `VI.buildDouble()` / `VI.buildFloat()` 工厂方法创建索引
- 支持自动选择索引类型（`IdxType.AUTO`）
- 支持多种索引类型：`HNSW`、`KDTree`、`LSH`、`PQ`、`BRUTE_FORCE`

### 9.2 支持的索引类型

| 索引类型 | 说明 | 适用场景 |
|----------|------|----------|
| `HNSW` | 分层可导航小世界图 | 高召回、低延迟、大规模数据 |
| `KDTree` | KD 树 | 低维数据（dim ≤ 20） |
| `LSH` | 局部敏感哈希 | 近似最近邻搜索 |
| `PQ` | 乘积量化 | 内存受限的大规模数据 |
| `PQ_HNSW` | PQ + HNSW 混合 | 超大规模数据 |
| `BRUTE_FORCE` | 暴力搜索 | 小数据集或基准测试 |
| `AUTO` | 自动选择 | 根据数据规模自动决定 |

### 9.3 使用示例

```java
// ✅ 正确：使用 VI 工厂创建索引
double[][] data = /* 加载数据 */;
String[] ids = /* 文档 ID */;
VecSearchOption options = new VecSearchOption()
    .indexType(IdxType.HNSW)
    .efConstruction(100)
    .m(16);

// 构建 Double 类型向量索引
IDoubleVecIdx index = VI.buildDouble(data, ids, options);

// ✅ 正确：使用自定义距离度量
IDisMetric<Double> metric = new CosineMetric();
IDoubleVecIdx index2 = VI.buildDouble(data, ids, metric, options);

// ✅ 正确：自动选择索引类型
options.indexType(IdxType.AUTO);  // 根据数据规模自动选择
IDoubleVecIdx autoIndex = VI.buildDouble(data, ids, options);

// ✅ 正确：搜索
String[] query = new String[]{"doc1", "doc2"};
int k = 10;
SearchResult result = index.search(query, k);
String[] neighborIds = result.getNeighborIds();
double[] distances = result.getDistances();

// ❌ 错误：直接实例化
// IVecIdx index = new HnswDoubleVecIdx(data, ids, metric, options);
```

### 9.4 距离度量

```java
// 欧氏距离
IDisMetric<Double> euclidean = EuclideanMetric.DOUBLE;

// 平方欧氏距离
IDisMetric<Double> squaredEuclidean = SquaredEuclideanMetric.DOUBLE;

// 余弦距离
IDisMetric<Double> cosine = CosineMetric.DOUBLE;

// 从度量学习得到
IDisMetric<Double> learnedMetric = metricLearner.getResult();
```

---

## 10. 信号处理模块使用模式

### 10.1 Signals 入口

**位置**：`com.yishape.lab.math.signal.Signals`

**核心原则**：
- 采用**两级门面模式**：顶层 `Signals` 类提供静态 `Wrapper` 子域
- 子域包括：`gen`（生成）、`filt`（滤波）、`xform`（变换）、`analyze`（分析）、`plot`（可视化）

### 10.2 信号生成（Signals.gen）

```java
// 正弦波
IVector<Double> sine = Signals.gen.sineWave(1000, 10.0, 1000.0, 1.0, 0.0);
// 参数：长度、频率、采样率、幅值、相位

// 余弦波
IVector<Double> cos = Signals.gen.cosineWave(1000, 10.0, 1000.0, 1.0, 0.0);

// 方波
IVector<Double> square = Signals.gen.squareWave(1000, 10.0, 1000.0, 1.0, 0.5);
// 参数：长度、频率、采样率、幅值、占空比

// 三角波
IVector<Double> tri = Signals.gen.triangularWave(1000, 10.0, 1000.0, 1.0, 0.5);

// 白噪声
IVector<Double> noise = Signals.gen.whiteNoise(1000, 0.1);
// 参数：长度、功率

// 粉噪声
IVector<Double> pink = Signals.gen.pinkNoise(1000, 0.1);

// 复合信号
IVector<Double> composite = Signals.gen.compositeSignal(signalTypes, length, parameters);

// 添加噪声
IVector<Double> noisySignal = Signals.gen.addNoise(signal, noiseType, parameters);
```

### 10.3 信号滤波（Signals.filt）

```java
// 移动平均滤波
IVector<Double> smoothed = Signals.filt.movingAverage(signal, windowSize);

// 中值滤波
IVector<Double> median = Signals.filt.medianFilter(signal, windowSize);

// 高斯滤波
IVector<Double> gaussian = Signals.filt.gaussianFilter(signal, sigma);
IVector<Double> gaussian2 = Signals.filt.gaussianFilter(signal, sigma, kernelSize);

// Butterworth 低通滤波
IVector<Double> lowpassed = Signals.filt.butterworthLowPass(signal, cutoffFreq, samplingRate, order);

// Butterworth 高通滤波
IVector<Double> highpassed = Signals.filt.butterworthHighPass(signal, cutoffFreq, samplingRate, order);

// 带通滤波
IVector<Double> bandpassed = Signals.filt.bandPass(signal, lowFreq, highFreq, samplingRate, order);

// 带阻滤波
IVector<Double> bandstop = Signals.filt.bandStop(signal, lowFreq, highFreq, samplingRate, order);

// 卡尔曼滤波
IVector<Double> kalman = Signals.filt.kalmanFilter(signal, processNoise, measurementNoise);

// 维纳滤波
IVector<Double> wiener = Signals.filt.wienerFilter(signal, signalPower, noisePower, filterLength);
```

### 10.4 信号变换（Signals.xform）

```java
// 快速傅里叶变换（FFT）
Complex[] spectrum = Signals.xform.fft(complexSignal);

// 逆 FFT
Complex[] timeSignal = Signals.xform.ifft(freqSignal);

// 幅度谱
double[] magnitude = Signals.xform.magnitudeSpectrum(fftResult);

// 相位谱
double[] phase = Signals.xform.phaseSpectrum(fftResult);

// 功率谱
double[] power = Signals.xform.powerSpectrum(fftResult);

// 离散余弦变换（DCT）
IVector<Double> dct = Signals.xform.dct2(signal);
IVector<Double> idct = Signals.xform.idct2(dctSignal);

// Hilbert 变换
IVector<Double> hilbert = Signals.xform.hilbertTransform(signal);

// 解析信号
Complex[] analytic = Signals.xform.analyticSignal(signal);

// 瞬时幅度、相位、频率
IVector<Double> amp = Signals.xform.instantaneousAmplitude(signal);
IVector<Double> ph = Signals.xform.instantaneousPhase(signal);
IVector<Double> freq = Signals.xform.instantaneousFrequency(signal, samplingRate);

// 离散小波变换
WaveletCoefficients coeffs = Signals.xform.discreteWaveletTransform(signal, waveletType, levels, param);
IVector<Double> reconstructed = Signals.xform.inverseDiscreteWaveletTransform(coeffs, waveletType, param);
```

### 10.5 信号分析（Signals.analyze）

```java
// 功率谱密度
Tuple2<IVector<Double>, IVector<Double>> psd = Signals.analyze.powerSpectralDensity(signal, windowSize, overlap, samplingRate);
// 返回：(频率向量, 功率谱密度向量)

// 自相关
IVector<Double> autocorr = Signals.analyze.autocorrelation(signal);

// 互相关
IVector<Double> crosscorr = Signals.analyze.crossCorrelation(signal1, signal2);
IVector<Double> crosscorr2 = Signals.analyze.crossCorrelation(signal1, signal2, maxLag);

// 频谱分析
Tuple3<IVector<Double>, IVector<Double>, IVector<Double>> spec = Signals.analyze.spectrum(signal, samplingRate);
// 返回：(频率向量, 幅度向量, 相位向量)

// 短时傅里叶变换
IMatrix<Double> stft = Signals.analyze.shortTimeFourierTransform(signal, windowSize, hopSize, samplingRate);

// 信噪比
double snr = Signals.analyze.signalToNoiseRatio(signal, noise);

// 峰值信噪比
double psnr = Signals.analyze.peakSignalToNoiseRatio(original, reconstructed);
```

---

## 11. 时间序列分析模块使用模式

### 11.1 TSA 入口

**位置**：`com.yishape.lab.math.timeseries.TSA`

**核心原则**：
- 采用**两级门面模式**：顶层 `TSA` 类提供静态工厂方法和 `Wrapper` 子域
- `TSA.data()` 创建 `TimeSeriesData`（类似 `Linalg.vector()` 模式）
- 子域包括：`forecast`（预测）、`filter`（滤波）、`decompose`（分解）、`cointegrate`（协整）、`plot`（可视化）

### 11.2 时间序列创建

```java
// 从 IVector 创建（类似 Linalg.vector 模式）
TimeSeriesData ts = TSA.data(values, "price");

// 从 double[] 创建
TimeSeriesData ts2 = TSA.data(new double[]{1.0, 2.0, 3.0}, "price");

// 带时间戳
LocalDateTime[] timestamps = { ... };
TimeSeriesData ts3 = TSA.data(values, "price", timestamps);

// 也可直接用 TimeSeriesData 静态工厂
TimeSeriesData ts4 = TimeSeriesData.of(values, "price");
TimeSeriesData ts5 = TimeSeriesData.of(values, 1.0, "price", LocalDateTime.now());
```

### 11.3 时间序列预测（TSA.forecast）

```java
IVector<Double> y = ts.getVariable(0);

// 移动平均预测（默认 95% 置信度）
ForecastResult sma = TSA.forecast.movingAverage(y, windowSize, forecastSteps);

// 指数平滑（可指定置信度）
ForecastResult exp = TSA.forecast.expSmooth(y, alpha, forecastSteps, 0.95);

// 线性趋势预测
ForecastResult linear = TSA.forecast.linearTrend(y, forecastSteps);

// ARIMA 预测（p=自回归阶数, d=差分阶数, q=移动平均阶数）
ForecastResult arima = TSA.forecast.arima(y, p, d, q, forecastSteps);

// 季节性预测
ForecastResult seasonal = TSA.forecast.seasonal(y, period, forecastSteps);

// Holt-Winters 三次指数平滑
ForecastResult hw = TSA.forecast.holtWinters(y, alpha, beta, gamma, period, forecastSteps);

// GARCH 模型预测
ForecastResult garch = TSA.forecast.garch(y, p, q, forecastSteps);

// 状态空间模型预测
ForecastResult ssm = TSA.forecast.stateSpace(y, sigmaEta, sigmaZeta, sigmaEpsilon, forecastSteps);

// 自动预测（自动选择最佳模型）
ForecastResult auto = TSA.forecast.auto(y, forecastSteps);
```

### 11.4 时间序列分解（TSA.decompose）

```java
IVector<Double> y = ts.getVariable(0);

// 经典分解（加法模型）
DecompositionResult classical = TSA.decompose.classical(y, period,
    TimeSeriesDecomposition.DecompositionModel.ADDITIVE);
// 或 MULTIPLICATIVE

// X-13 分解
DecompositionResult x13 = TSA.decompose.x13(y, period);

// STL 分解
DecompositionResult stl = TSA.decompose.stl(y, period, seasonalWindow, trendWindow);

// 小波分解
DecompositionResult wavelet = TSA.decompose.wavelet(y, "haar", levels);
```

### 11.5 时间序列滤波（TSA.filter）

```java
IVector<Double> y = ts.getVariable(0);

// 移动平均滤波
FilterResult ma = TSA.filter.movingAverage(y, windowSize);

// 指数平滑滤波
FilterResult exp = TSA.filter.expSmooth(y, alpha);

// 高斯滤波
FilterResult gauss = TSA.filter.gaussian(y, sigma);

// 中值滤波
FilterResult median = TSA.filter.median(y, windowSize);

// 低通/高通/带通滤波（默认采样率=1.0，也可显式指定）
FilterResult lp = TSA.filter.lowPass(y, cutoffFreq, order);
FilterResult hp = TSA.filter.highPass(y, cutoffFreq, order);
FilterResult bp = TSA.filter.bandPass(y, lowFreq, highFreq, order);

// 自适应滤波
FilterResult adaptive = TSA.filter.adaptive(y, learningRate);
```

### 11.6 协整分析（TSA.cointegrate）

```java
// Engle-Granger 协整检验
CointegrationAnalysis.EngleGrangerResult eg = TSA.cointegrate.engleGrangerTest(y, x, maxLags);

// Johansen 协整检验
CointegrationAnalysis.JohansenResult joh = TSA.cointegrate.johansenTest(data, maxLags, TrendType.CONSTANT);
// TrendType: CONSTANT, TREND, NONE

// 估计协整关系
CointegrationAnalysis.CointegratingRelationship cr = TSA.cointegrate.estimateCointegratingRelationship(y, x);

// 估计误差修正模型
CointegrationAnalysis.ErrorCorrectionModel ecm = TSA.cointegrate.estimateECM(deltaY, deltaX, residuals, maxLags);
```

---

## 12. 绘图模块使用模式

### 12.1 Plots 入口

**位置**：`com.yishape.lab.math.plot.Plots`

**核心原则**：
- **`Plots` 是绘图的唯一静态工厂入口**（与 `Linalg`、`Stats` 同级），不要直接 `new` 具体后端类
- `Plots.of()`、`Plots.line(x, y)` 等返回 **`IPlot` 接口**，用于链式配置与展示
- 实际渲染由 **`PlotProvider`** 决定：`JavaFx`（默认）、`Echarts`、`Svg`
- 3D 绘图使用 `Plots.of3d()`，后端由 `PlotProvider3d`（`JavaFx` / `EchartsGL`）控制

```java
// ✅ 正确：通过 Plots 创建图表
Plots.line(x, y)
    .title("折线图示例")
    .xlabel("X轴")
    .ylabel("Y轴")
    .show();

// ✅ 正确：切换全局后端（影响后续 of() 及 line/scatter 等工厂）
Plots.setProvider(PlotProvider.Echarts);

// ✅ 正确：一次性指定后端（不改变全局默认）
IPlot webPlot = Plots.ofEcharts();
IPlot desktopPlot = Plots.ofJavaFx();
SvgPlot vectorPlot = Plots.ofSvg();

// ✅ 正确：查看可用主题名
List<String> themes = Plots.listThemes();

// ❌ 错误：把 IPlot 当作入口（IPlot 是接口，不是工厂）
// IPlot plot = ???;  // 应使用 Plots.of() 或 Plots.line(...)
```

### 12.2 基础图表

```java
// 折线图（Plots.line 已创建并绘制第一条序列，可继续链式追加）
Plots.line(x, y)
   .title("折线图示例")
   .show();

// 多折线图：在已有 IPlot 上追加，或多次调用 line 工厂
IPlot plot = Plots.of();
plot.line(x, y1, labels1)
   .line(x, y2, labels2)
   .show();

// 散点图
Plots.scatter(x, y)
   .title("散点图示例")
   .show();

// 柱状图
Plots.bar(x)
   .title("柱状图示例")
   .show();

// 分组柱状图
Plots.bar(xticks, y, hue)
   .title("分组柱状图")
   .show();

// 饼图
Plots.pie(values)
   .title("饼图示例")
   .show();

// 带标签饼图
Plots.pie(values, labels)
   .show();

// 直方图
Plots.hist(data, true)  // true = 显示拟合线
   .title("直方图示例")
   .show();
```

### 12.3 统计图表

```java
// 箱线图
Plots.boxplot(data, labels)
   .title("箱线图")
   .show();

// 小提琴图
Plots.violinplot(data, labels)
   .title("小提琴图")
   .show();

// K线图（OHLC）
IMatrix ohlc = /* 开盘、收盘、最高、最低价 */;
Plots.candlestick(ohlc, dates)
   .title("K线图")
   .show();
```

### 12.4 极坐标图表

```java
// 极坐标柱状图
Plots.polarBar(data, categories)
   .title("极坐标柱状图")
   .show();

// 极坐标线图
Plots.polarLine(data, categories)
   .show();

// 极坐标散点图
Plots.polarScatter(data, categories)
   .show();
```

### 12.5 高级图表

```java
// 热力图
Plots.heatmap(data, xLabels, yLabels)
   .title("热力图")
   .show();

// 雷达图
Plots.radar(data, indicators)
   .title("雷达图")
   .show();

// 仪表盘
Plots.gauge(value, max, min)
   .title("仪表盘")
   .show();

// 漏斗图
Plots.funnel(data, labels)
   .title("漏斗图")
   .show();

// 关系图
Plots.graph(nodes, links)
   .title("关系图")
   .show();

// 平行坐标图
Plots.parallel(data, dimensions)
   .title("平行坐标图")
   .show();
```

### 12.6 流式 API 配置

```java
Plots.line(x, y)
   .title("标题", "副标题")      // 标题 + 副标题
   .xlabel("X轴标签")
   .ylabel("Y轴标签")
   .size(800, 600)             // 图表尺寸
   .theme("dark")              // 主题
   .setPalette("viridis")      // 色板
   .show();

// 保存图表（方法在 IPlot 上，由 Plots 工厂返回的实例调用）
Plots.line(x, y).saveAsHtml("chart.html");
Plots.line(x, y).saveAsPng("chart.png");
Plots.line(x, y).saveAsSvg("chart.svg");
Plots.line(x, y).saveAsPdf("chart.pdf");
```

### 12.7 特殊图表（Map/树结构）

```java
// 桑基图
List<Map<String, Object>> nodes = /* ... */;
List<Map<String, Object>> links = /* ... */;
Plots.sankey(nodes, links)
   .title("桑基图")
   .show();

// 旭日图
Plots.sunburst(hierarchicalData)
   .show();

// 主题河流图
Plots.themeRiver(multiSeriesData, categories)
   .show();

// 树图
Plots.tree(treeData)
   .show();

// 矩形树图
Plots.treemap(hierarchicalData)
   .show();
```

### 12.8 Plots 与 IPlot 的分工

| 层级 | 类型 | 职责 |
|------|------|------|
| 入口 | `Plots` | 静态工厂、`setProvider`、常用图表一行创建 |
| 契约 | `IPlot` | 链式配置、`show()`、导出 HTML/PNG/SVG/PDF |
| 实现 | `JavaFxPlot` / `EchartsPlot` / `SvgPlot` | 用户不应直接依赖，由 `PlotProvider` 选择 |

---

## 13. 常见错误与避免方法

### 13.1 矩阵向量运算错误

```java
// ❌ 错误：维度不匹配
IMatrix<Double> A = Linalg.rand(3, 4);
IVector<Double> v = Linalg.rand(5);  // 长度 5 ≠ A 的列数 4
IMatrix<Double> C = A.mmul(v);       // 运行时错误

// ✅ 正确：确保维度匹配
IVector<Double> v = Linalg.rand(4); // 长度 4 = A 的列数
IMatrix<Double> C = A.mmul(v);       // OK
```

### 13.2 类型不匹配

```java
// ❌ 错误：类型混淆
IMatrix<Double> A = Linalg.eye(3);
IMatrix<Float> B = /* 错误：不能直接将 Double 赋给 Float */;

// ✅ 正确：显式转换
IMatrix<Float> B = /* 手动转换或重新创建 */;
```

### 13.3 优化器使用错误

```java
// ❌ 错误：目标函数和梯度不一致
IObjectiveFunction obj = x -> x.dot(x);
IGradientFunction grad = x -> x.scale(3.0);  // 应该是 2x，不是 3x

// ✅ 正确：确保数学一致性
IObjectiveFunction obj = x -> x.dot(x);      // f(x) = x·x
IGradientFunction grad = x -> x.scale(2.0); // ∇f(x) = 2x
```

### 13.4 稀疏矩阵使用错误

```java
// ❌ 错误：稀疏索引越界
ISparseMatrix sparse = Linalg.sparseEye(5);
sparse.set(10, 10, 1.0);  // 越界！

// ✅ 正确：检查索引范围
int n = sparse.rows();  // 或 cols()
sparse.set(Math.min(10, n-1), Math.min(10, n-1), 1.0);
```

### 13.5 机器学习流程错误

```java
// ❌ 错误：训练集测试集混淆
IClassifier clf = ML.clf.logisticRegression().fit(Xtest, ytest);  // 用错了！
int[] pred = clf.predict(Xtrain);  // 应该用 Xtest

// ✅ 正确：正确的训练预测流程
IClassifier clf = ML.clf.logisticRegression().fit(Xtrain, ytrain);
int[] pred = clf.predict(Xtest);
```

### 13.6 绘图入口错误

```java
// ❌ 错误：把 IPlot 当作工厂或自行 new 后端实现
// IPlot plot = new JavaFxPlot();
// plot.line(x, y).show();  // 应统一从 Plots 进入

// ✅ 正确：Plots 创建，IPlot 仅用于链式配置
Plots.line(x, y).title("示例").show();
Plots.setProvider(PlotProvider.Echarts);  // 需要 Web 图表时切换后端
```

---

## 14. 性能注意事项

### 14.1 热点路径优化

对于热点计算，可使用原始数组（`double[]`/`float[]`）避免 `IVector` 包装开销：

```java
// 高性能场景：使用原始数组
float[] a = new float[]{1f, 2f, 3f};
float[] b = new float[]{4f, 5f, 6f};
float dotProduct = Linalg.dot(a, b);  // SIMD 加速，不创建 IVector

float sqDist = Linalg.squaredDistance(a, b);  // 平方欧氏距离
```

### 14.2 Double vs Float

- **默认使用 `Double`**：精度更高
- **内存敏感场景使用 `Float`**：节省 50% 内存
- **HPC 加速场景优先 `Float`**：SIMD 效率更高

### 14.3 避免不必要的拷贝

```java
// ❌ 错误：频繁拷贝
IMatrix<Double> A = Linalg.rand(1000, 1000);
for (int i = 0; i < 100; i++) {
    IMatrix<Double> B = A.copy();  // 每次都拷贝
    // ...
}

// ✅ 正确：按需拷贝
IMatrix<Double> A = Linalg.rand(1000, 1000);
IMatrix<Double> B = A.copy();  // 只在需要时拷贝
```

### 14.4 批量操作优先

```java
// ✅ 正确：批量添加向量（通过 VI 工厂）
VecSearchOption options = new VecSearchOption().indexType(IdxType.HNSW);
IDoubleVecIdx index = VI.buildDouble(vectors, ids, options);
index.addAll(vectors);  // 批量添加，比逐个添加高效

// ❌ 低效：逐个添加
for (Vector v : vectors) {
    index.add(v);
}
```

---

## 附录 A：完整导入示例

```java
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;

import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.math.stats.distribution.NormalDistribution;

import com.yishape.lab.math.ml.ML;
import com.yishape.lab.math.ml.clf.IClassifier;
import com.yishape.lab.math.ml.reg.IRegression;
import com.yishape.lab.math.ml.dr.IDimReduction;
import com.yishape.lab.math.ml.clu.IClustering;
import com.yishape.lab.math.ml.preproc.IPreprocessor;

import com.yishape.lab.math.optimize.Opts;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.OptResult;

import com.yishape.lab.math.vecidx.VI;
import com.yishape.lab.math.vecidx.IVecIdx;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.math.vecidx.IdxType;

import com.yishape.lab.math.signal.Signals;

import com.yishape.lab.math.timeseries.TSA;
import com.yishape.lab.math.timeseries.TimeSeriesData;
import com.yishape.lab.math.timeseries.ForecastResult;

import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.PlotProvider;
```

---

## 附录 B：快速参考卡

### Linalg 常用方法

| 操作 | 代码 |
|------|------|
| 创建 3×3 全零矩阵 | `Linalg.zeros(3, 3)` |
| 创建 3×3 单位矩阵 | `Linalg.eye(3)` |
| 创建随机矩阵 | `Linalg.rand(3, 4)` |
| 创建向量 | `Linalg.vector(new double[]{1,2,3})` |
| 矩阵乘法 | `A.mmul(B)` |
| 矩阵转置 | `A.t()` |
| 矩阵求逆 | `A.inv()` |
| 解线性方程 | `Linalg.solve(A, b)` |

### ML 常用方法

| 操作 | 代码 |
|------|------|
| PCA 降维 | `ML.dr.pca(k).fitTransform(X)` |
| K-Means 聚类 | `ML.clu.kMeans(k).fit(X).getResult()` |
| 逻辑回归 | `ML.clf.logisticRegression().fit(X, y)` |
| 数据标准化 | `ML.preproc.standardScaler().fitTransform(X)` |

### Opts 常用方法

| 操作 | 代码 |
|------|------|
| 创建 LBFGS | `Opts.lbfgs()` |
| 创建 Adam | `Opts.onlineAdam()` |
| 创建 LP 求解器 | `Opts.linProgSolver()` |
| 执行优化 | `optimizer.optimize(x0, obj, grad)` |

### VI 常用方法

| 操作 | 代码 |
|------|------|
| 构建向量索引 | `VI.buildDouble(data, ids, options)` |
| 自动选择索引类型 | `VI.buildDouble(data, ids, IdxType.AUTO, options)` |
| 搜索最近邻 | `index.search(query, k)` |

### Signals 常用方法

| 操作 | 代码 |
|------|------|
| 生成正弦波 | `Signals.gen.sineWave(1000, 10.0, 1000.0, 1.0, 0.0)` |
| FFT 变换 | `Signals.xform.fft(complexSignal)` |
| 功率谱密度 | `Signals.analyze.powerSpectralDensity(signal, window, overlap, samplingRate)` |
| 低通滤波 | `Signals.filt.butterworthLowPass(signal, cutoffFreq, samplingRate, order)` |

### TSA 常用方法

| 操作 | 代码 |
|------|------|
| 创建时间序列 | `TSA.data(values, "price")` |
| ARIMA 预测 | `TSA.forecast.arima(y, p, d, q, steps)` |
| 指数平滑 | `TSA.forecast.expSmooth(y, alpha, steps)` |
| STL 分解 | `TSA.decompose.stl(y, period, seasonalWindow, trendWindow)` |
| 低通滤波 | `TSA.filter.lowPass(y, cutoffFreq, order)` |
| 协整检验 | `TSA.cointegrate.engleGrangerTest(y, x, maxLags)` |

### Plots 常用方法

| 操作 | 代码 |
|------|------|
| 创建绘图实例 | `Plots.of()` |
| 折线图 | `Plots.line(x, y).title("标题").show()` |
| 散点图 | `Plots.scatter(x, y).show()` |
| 直方图 | `Plots.hist(data, true).show()` |
| 切换 ECharts 后端 | `Plots.setProvider(PlotProvider.Echarts)` |
| 保存 HTML | `Plots.line(x, y).saveAsHtml("chart.html")` |

---

*文档版本：1.2*  
*最后更新：2026-05-17*
