# 向量操作 (Vector Operations)

## 概述 / Overview

`IVector<T>` 接口提供了完整的泛型向量数学运算功能，支持 `Float` 和 `Double` 类型，包括基本数学运算、统计运算、切片索引、通用函数等。该接口设计简洁，功能强大，支持链式操作。推荐使用 `Linalg` 工厂类来创建向量实例。

The `IVector<T>` interface provides comprehensive generic vector mathematical operations supporting `Float` and `Double` types, including basic mathematical operations, statistical operations, slicing and indexing, universal functions, and more. The interface features clean design, powerful functionality, and supports method chaining. It is recommended to use the `Linalg` factory class to create vector instances.

与矩阵共用的广播、`.npy` 等入口速查见 [`examples/Matrix-Examples.md`](examples/Matrix-Examples.md) 文首。

For shared broadcasting and `.npy` entry points with matrices, see the opening section of [`examples/Matrix-Examples.md`](examples/Matrix-Examples.md).

## 快速上手 / Quick Start

```java
// 创建向量（推荐用 Linalg）/ Create vectors (recommended via Linalg)
IVector<Double> v = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
IVector<Double> ones = Linalg.ones(5);          // [1,1,1,1,1]
IVector<Double> range = Linalg.range(1, 6);     // [1,2,3,4,5]

// 链式操作 / Method chaining
Double mean = v.multiplyScalar(2.0).addScalar(1.0).mean();
System.out.println("均值: " + mean);  // 均值: 8.0

// 向量运算 / Vector operations
IVector<Double> a = Linalg.vector(new double[]{1.0, 2.0, 3.0});
IVector<Double> b = Linalg.vector(new double[]{0.1, 0.2, 0.3});
IVector<Double> sum = a.add(b);           // [1.1, 2.2, 3.3]
Double dot = a.innerProduct(b);            // 内积: 1.4
IVector<Double> norm = a.normalize();      // 归一化（返回新向量，不修改原向量）
```

## 核心接口 / Core Interface

### IVector 接口 / IVector Interface

`IVector<T>` 是向量操作的核心泛型接口，定义了所有向量运算的抽象方法，支持 `Float` 和 `Double` 类型。

`IVector<T>` is the core generic interface for vector operations, defining abstract methods for all vector operations, supporting `Float` and `Double` types.

### Linalg 工厂类 / Linalg Factory Class

`Linalg` 类提供了统一的向量和矩阵创建入口，推荐使用此类来创建向量实例。该类采用委托模式，将创建请求委托给相应的接口实现。

`Linalg` class provides a unified entry point for vector and matrix creation, recommended for creating vector instances. This class uses delegation pattern, delegating creation requests to corresponding interface implementations.

**架构设计 / Architecture Design:**
- **委托链** / **Delegation chain**: `Linalg` → `IVector` → `IFloatVector/IDoubleVector`
- **类型推断** / **Type inference**: 根据输入数据类型自动选择合适的实现
- **API统一** / **API consistency**: 提供一致的命名和使用模式

```java
// 推荐使用 Linalg 创建向量 / Recommended to use Linalg for vector creation
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0});
IVector<Float> v2 = Linalg.vector(new float[]{1.0f, 2.0f, 3.0f, 4.0f});
```

## 主要功能 / Main Features

### 1. 向量创建 / Vector Creation

#### 静态工厂方法 / Static Factory Methods

```java
// 推荐使用 Linalg 工厂类 / Recommended to use Linalg factory class

// 从数组创建向量 / Create vector from array
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0});
IVector<Float> v2 = Linalg.vector(new float[]{1.0f, 2.0f, 3.0f, 4.0f});

// 从包装类数组创建 / Create from wrapper array
IVector<Double> v3 = Linalg.vector(new Double[]{1.0, 2.0, 3.0, 4.0});
IVector<Float> v4 = Linalg.vector(new Float[]{1.0f, 2.0f, 3.0f, 4.0f});

// 从基本类型数组创建 / Create from primitive arrays
IVector<Double> v5 = Linalg.vector(new int[]{1, 2, 3, 4});        // 自动转换为Double
IVector<Double> v6 = Linalg.vector(new Integer[]{1, 2, 3, 4});    // 自动转换为Double

// 从单个值创建向量 / Create vector from single values
IVector<Double> v7 = Linalg.vector(5.0);                          // [5.0]
IVector<Double> v8 = Linalg.vector(new double[]{1.0, 2.0});        // [1.0, 2.0]
IVector<Float> v9 = Linalg.vector(3.0f);                          // [3.0f]
IVector<Float> v10 = Linalg.vector(new float[]{1.0f, 2.0f});      // [1.0f, 2.0f]

// 创建范围向量 / Create range vector
IVector<Double> v11 = Linalg.range(10);                           // [0.0, 1.0, 2.0, ..., 9.0]
IVector<Double> v12 = Linalg.range(1, 11);                        // [1.0, 2.0, 3.0, ..., 10.0]
IVector<Double> v13 = Linalg.range(0, 20, 2);                     // [0.0, 2.0, 4.0, ..., 18.0]
IVector<Float> v14 = Linalg.range(5, Float.class);                // [0.0f, 1.0f, 2.0f, 3.0f, 4.0f]

// 创建特殊向量 / Create special vectors
IVector<Double> v15 = Linalg.ones(5);                             // [1.0, 1.0, 1.0, 1.0, 1.0]
IVector<Float> v16 = Linalg.zeros(5, Float.class);                // [0.0f, 0.0f, 0.0f, 0.0f, 0.0f]
IVector<Double> v17 = Linalg.rand(5);                             // 随机向量 / Random vector
IVector<Float> v18 = Linalg.rand(5, Float.class);                 // Float类型随机向量

// 正态分布随机向量 / Normal distribution random vectors
IVector<Double> v19 = Linalg.randn(5);                            // 标准正态分布
IVector<Double> v20 = Linalg.randn(5, 0.0, 1.0);                  // 指定均值和标准差
IVector<Float> v21 = Linalg.randn(5, 0.0f, 1.0f, Float.class);    // Float类型正态分布

// 创建线性空间向量 / Create linear space vectors
IVector<Double> v22 = Linalg.linspace(0.0, 1.0, 5);               // [0.0, 0.25, 0.5, 0.75, 1.0]
IVector<Float> v23 = Linalg.linspace(0.0f, 1.0f, 5, Float.class); // Float类型线性空间
IVector<Double> v24 = Linalg.logspace(0.0, 2.0, 4);               // [1.0, 10.0, 100.0, 1000.0]
IVector<Float> v25 = Linalg.logspace(0.0f, 2.0f, 4, Float.class); // Float类型对数空间

// 直接使用 IVector 接口（不推荐） / Direct use of IVector interface (not recommended)
IVector<Double> v26 = IVector.of(new double[]{1.0, 2.0, 3.0, 4.0});
IVector<Float> v27 = IVector.of(new float[]{1.0f, 2.0f, 3.0f, 4.0f});

// 使用 IDoubleVector 和 IFloatVector 接口 / Use IDoubleVector and IFloatVector interfaces
IDoubleVector v28 = IDoubleVector.of(new double[]{1.0, 2.0, 3.0, 4.0});
IFloatVector v29 = IFloatVector.of(new float[]{1.0f, 2.0f, 3.0f, 4.0f});

// IDoubleVector 专用工厂方法 / IDoubleVector specific factory methods
IDoubleVector v30 = IDoubleVector.range(10);                      // [0.0, 1.0, 2.0, ..., 9.0]
IDoubleVector v31 = IDoubleVector.ones(5);                        // [1.0, 1.0, 1.0, 1.0, 1.0]
IDoubleVector v32 = IDoubleVector.zeros(5);                       // [0.0, 0.0, 0.0, 0.0, 0.0]
IDoubleVector v33 = IDoubleVector.rand(5);                        // 随机向量
IDoubleVector v34 = IDoubleVector.randn(5);                       // 正态分布随机向量
IDoubleVector v35 = IDoubleVector.linspace(0.0, 1.0, 5);         // 线性空间向量
IDoubleVector v36 = IDoubleVector.logspace(0.0, 2.0, 4);         // 对数空间向量

// IFloatVector 专用工厂方法 / IFloatVector specific factory methods
IFloatVector v37 = IFloatVector.range(10);                        // [0.0f, 1.0f, 2.0f, ..., 9.0f]
IFloatVector v38 = IFloatVector.ones(5);                          // [1.0f, 1.0f, 1.0f, 1.0f, 1.0f]
IFloatVector v39 = IFloatVector.zeros(5);                         // [0.0f, 0.0f, 0.0f, 0.0f, 0.0f]
IFloatVector v40 = IFloatVector.rand(5);                          // 随机向量
IFloatVector v41 = IFloatVector.randn(5);                         // 正态分布随机向量
IFloatVector v42 = IFloatVector.linspace(0.0f, 1.0f, 5);         // 线性空间向量
IFloatVector v43 = IFloatVector.logspace(0.0f, 2.0f, 4);         // 对数空间向量
```

### 2. 基本数学运算 / Basic Mathematical Operations

#### 向量间运算 / Vector-to-Vector Operations

```java
// 创建向量 / Create vectors
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0});
IVector<Double> v2 = Linalg.vector(new double[]{5.0, 6.0, 7.0, 8.0});

// 加法 / Addition
IVector<Double> sum = v1.add(v2);

// 减法 / Subtraction
IVector<Double> diff = v1.sub(v2);

// 元素级乘法 / Element-wise multiplication
IVector<Double> product = v1.multiply(v2);

// 内积 / Inner product (dot product)
Double dotProduct = v1.innerProduct(v2);
Double dotProduct2 = v1.dot(v2);  // 简写形式

// 向量与矩阵乘法（行向量 × 矩阵，对齐 NumPy v @ M、np.dot(v, M)；向量长度须等于矩阵行数）
// Vector-matrix: row length must equal #matrix rows
IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}});
IVector<Double> result = v1.mmul(matrix);
IVector<Double> same = v1.dot(matrix);       // 与 mmul(matrix) 等价 / same as mmul(matrix)

// 向量外积 / Vector outer product
IMatrix<Double> outerProduct = v1.outer(v2);
```

#### 标量运算 / Scalar Operations

```java
// 标量加法 / Scalar addition
IVector<Double> result1 = v1.addScalar(5.0);

// 标量减法 / Scalar subtraction
IVector<Double> result2 = v1.subScalar(2.0);

// 标量乘法 / Scalar multiplication
IVector<Double> result3 = v1.multiplyScalar(3.0);

// 标量除法 / Scalar division
IVector<Double> result4 = v1.divideByScalar(2.0);
```

### 3. 统计运算 / Statistical Operations

```java
// 创建向量 / Create vector
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});

// 基本统计 / Basic statistics
Double sum = v1.sum();           // 求和 / Sum
Double mean = v1.mean();         // 均值 / Mean
Double variance = v1.var();      // 方差 / Variance
Double std = v1.std();           // 标准差 / Standard deviation
Double prod = v1.prod();         // 元素乘积 / Product

// 最值 / Min/Max
Double min = v1.min();           // 最小值 / Minimum
Double max = v1.max();           // 最大值 / Maximum
int minIndex = v1.argMin();      // 最小值索引 / Index of minimum
int maxIndex = v1.argMax();      // 最大值索引 / Index of maximum

// 范数 / Norms
Double norm1 = v1.norm1();       // L1范数 / L1 norm
Double norm2 = v1.norm2();       // L2范数 / L2 norm
Double normInf = v1.normInf();   // 无穷范数 / Infinity norm
Double normP = v1.norm(3.0);     // Lp范数 / Lp norm

// 其他统计量 / Other statistics
Double ptp = v1.ptp();           // 峰峰值 / Peak-to-peak value
Double median = v1.median();     // 中位数 / Median
Double mode = v1.mode();         // 众数 / Mode
Double percentile = v1.percentile(75.0); // 75%分位数 / 75th percentile
Double skewness = v1.skewness(); // 偏度 / Skewness
Double kurtosis = v1.kurtosis(); // 峰度 / Kurtosis

// 分位数 / Quantiles
Double q1 = v1.q1();             // 第一四分位数 / First quartile
Double q3 = v1.q3();             // 第三四分位数 / Third quartile
```

### 4. 通用函数 / Universal Functions

```java
// 创建向量 / Create vector
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0});

// 数学函数 / Mathematical functions
IVector<Double> squared = v1.square();          // 平方 / Square
IVector<Double> sqrt = v1.sqrt();               // 平方根 / Square root
IVector<Double> exp = v1.exp();                 // 指数 / Exponential
IVector<Double> log = v1.log();                 // 自然对数 / Natural logarithm
IVector<Double> log10 = v1.log10();             // 以10为底对数 / Base-10 logarithm
IVector<Double> abs = v1.abs();                 // 绝对值 / Absolute value
IVector<Double> pow = v1.pow(2.0);              // 幂运算 / Power
IVector<Double> remainder = v1.remainder(2.0);  // 取余运算 / Remainder
IVector<Double> reciprocal = v1.reciprocal();   // 倒数 / Reciprocal

// 三角函数 / Trigonometric functions
IVector<Double> sin = v1.sin();                 // 正弦 / Sine
IVector<Double> cos = v1.cos();                 // 余弦 / Cosine
IVector<Double> tan = v1.tan();                 // 正切 / Tangent
IVector<Double> arcsin = v1.arcsin();           // 反正弦 / Arcsine
IVector<Double> arccos = v1.arccos();           // 反余弦 / Arccosine
IVector<Double> arctan = v1.arctan();           // 反正切 / Arctangent

// 双曲函数 / Hyperbolic functions
IVector<Double> sinh = v1.sinh();               // 双曲正弦 / Hyperbolic sine
IVector<Double> cosh = v1.cosh();               // 双曲余弦 / Hyperbolic cosine
IVector<Double> tanh = v1.tanh();               // 双曲正切 / Hyperbolic tangent

// 舍入函数 / Rounding functions
IVector<Double> round = v1.round();             // 四舍五入 / Round
IVector<Double> floor = v1.floor();             // 向下取整 / Floor
IVector<Double> ceil = v1.ceil();               // 向上取整 / Ceiling
IVector<Double> trunc = v1.trunc();             // 截断取整 / Truncate
```

### 5. 切片和索引 / Slicing and Indexing

#### 范围切片 / Range Slicing

```java
// 创建向量 / Create vector
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

// 基本切片 / Basic slicing
IVector<Double> slice1 = v1.slice(2);          // 从索引2到末尾 / From index 2 to end
IVector<Double> slice2 = v1.slice(1, 4);       // 从索引1到4（不包含） / From index 1 to 4 (exclusive)
IVector<Double> slice3 = v1.slice(0, 6, 2);    // 从索引0到6，步长为2 / From index 0 to 6 with step 2

// 负数索引切片 / Negative indexing slicing
IVector<Double> slice4 = v1.slice(-3);         // 从倒数第3个到末尾 / From 3rd to last to end
IVector<Double> slice5 = v1.slice(1, -1);      // 从索引1到倒数第1个（不包含） / From index 1 to 2nd to last (exclusive)
IVector<Double> slice6 = v1.slice(-4, -1, 2);  // 从倒数第4个到倒数第1个，步长为2 / From 4th to last to 2nd to last with step 2

// 字符串切片表达式 / String slice expressions
IVector<Double> slice7 = v1.slice("1:4");      // 从索引1到4（不包含） / From index 1 to 4 (exclusive)
IVector<Double> slice8 = v1.slice(":-1");      // 从开始到倒数第1个（不包含） / From start to 2nd to last (exclusive)
IVector<Double> slice9 = v1.slice("::2");      // 从开始到末尾，步长为2 / From start to end with step 2
IVector<Double> slice10 = v1.slice("1:-1:2");  // 从索引1到倒数第1个，步长为2 / From index 1 to 2nd to last with step 2

// 花式索引 / Fancy indexing
IVector<Double> fancy1 = v1.fancyGet(new int[]{0, 2, 3});  // 获取索引0, 2, 3的元素
IVector<Double> fancy2 = v1.fancyGet(new int[]{-1, -2, 0}); // 获取倒数第1、2个和第一个元素

// 布尔索引 / Boolean indexing
boolean[] mask = {true, false, true, false, true, false};
IVector<Double> fancy3 = v1.booleanGet(mask);  // 获取mask为true的元素
```

#### 切片表达式语法 / Slice Expression Syntax

YiShape支持类似NumPy的切片表达式语法，支持负数索引：

YiShape supports NumPy-like slice expression syntax with negative indexing:

```java
// 创建向量 / Create vector
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

// 基本格式：start:end:step / Basic format: start:end:step
// 其中start、end、step都可以省略，支持负数索引
// Where start, end, step can be omitted, supports negative indexing

// 完整格式示例 / Complete format examples
v1.slice("1:5:2");     // 从索引1到5，步长为2 / From index 1 to 5 with step 2
v1.slice("1:5");       // 从索引1到5，步长为1 / From index 1 to 5 with step 1
v1.slice("1:");        // 从索引1到末尾 / From index 1 to end
v1.slice(":5");        // 从开始到索引5 / From start to index 5
v1.slice("::2");       // 从开始到末尾，步长为2 / From start to end with step 2
v1.slice(":");         // 整个向量 / Entire vector

// 负数索引示例 / Negative indexing examples
v1.slice("-3:");       // 从倒数第3个到末尾 / From 3rd to last to end
v1.slice(":-2");       // 从开始到倒数第2个（不包含） / From start to 2nd to last (exclusive)
v1.slice("-4:-1");     // 从倒数第4个到倒数第1个（不包含） / From 4th to last to 2nd to last (exclusive)
v1.slice("-4:-1:2");   // 从倒数第4个到倒数第1个，步长为2 / From 4th to last to 2nd to last with step 2
v1.slice("::-1");      // 反转向量 / Reverse vector
```

#### 负数索引规则 / Negative Indexing Rules

负数索引从-1开始，表示最后一个元素：

Negative indexing starts from -1, representing the last element:

```java
// 创建向量 / Create vector
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});

// 对于长度为5的向量 [1.0, 2.0, 3.0, 4.0, 5.0]
// For a vector of length 5 [1.0, 2.0, 3.0, 4.0, 5.0]

// 正数索引 / Positive indices: 0, 1, 2, 3, 4
// 负数索引 / Negative indices: -5, -4, -3, -2, -1

Double first = v1.get(0);    // 获取第一个元素 / Get first element (1.0)
Double last = v1.get(-1);    // 获取最后一个元素 / Get last element (5.0)
Double secondLast = v1.get(-2); // 获取倒数第二个元素 / Get second to last element (4.0)

IVector<Double> middle = v1.slice("1:-1");  // 从索引1到倒数第1个（不包含），即[2.0, 3.0, 4.0]
IVector<Double> lastTwo = v1.slice("-3:-1"); // 从倒数第3个到倒数第1个（不包含），即[3.0, 4.0]
```

### 6. 比较运算 / Comparison Operations

```java
// 创建向量 / Create vectors
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0});
IVector<Double> v2 = Linalg.vector(new double[]{2.0, 1.0, 4.0, 3.0});

// 元素级比较 / Element-wise comparison
boolean[] greater = v1.gt(v2);              // 大于 / Greater than
boolean[] less = v1.lt(v2);                // 小于 / Less than
boolean[] equal = v1.eq(v2);               // 等于 / Equal
```

### 7. 数据转换 / Data Conversion

```java
// 创建向量 / Create vector
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0});

// 类型转换 / Type conversion
double[] doubleArray = v1.toDoubleArray(); // 转换为double数组 / Convert to double array
float[] floatArray = v1.toFloatArray();    // 转换为float数组 / Convert to float array
int[] intArray = v1.toIntArray();          // 转换为int数组 / Convert to int array

// 数据访问 / Data access
Double element = v1.get(2);                // 获取指定索引的元素 / Get element at specific index
int length = v1.length();                  // 获取向量长度 / Get vector length
```

### 8. 高级操作 / Advanced Operations

#### 距离计算 / Distance Calculations

```java
// 创建向量 / Create vectors
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0});
IVector<Double> v2 = Linalg.vector(new double[]{2.0, 1.0, 4.0, 3.0});

// 距离度量 / Distance metrics
Double euclideanDist = v1.euclideanDistance(v2);  // 欧几里得距离 / Euclidean distance
Double manhattanDist = v1.manhattanDistance(v2);  // 曼哈顿距离 / Manhattan distance
Double cosineSim = v1.cosineSimilarity(v2);       // 余弦相似度 / Cosine similarity
```

#### 条件操作 / Conditional Operations

```java
// 创建向量 / Create vector
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0});

// 条件选择 / Conditional selection
boolean[] condition = {true, false, true, false};
IVector<Double> result = v1.where(condition, 10.0, 20.0);  // 根据条件选择值
IVector<Double> result2 = v1.where(condition, v2, v1);     // 根据条件选择向量
```

#### 重复和连接 / Repeat and Concatenation

```java
// 创建向量 / Create vector
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0});

// 重复操作 / Repeat operations
IVector<Double> repeated = v1.repeat(3);              // 重复向量3次
IVector<Double> tiled = v1.tile(2);                   // 平铺向量2次
```

#### 排序和查找 / Sorting and Searching

```java
// 创建向量 / Create vector
IVector<Double> v1 = Linalg.vector(new double[]{3.0, 1.0, 4.0, 2.0});

// 排序操作 / Sorting operations
IVector<Double> sorted = v1.sort();                   // 升序排序 / Ascending sort
IVector<Double> reversed = v1.reverse();               // 反转向量 / Reverse vector
```

#### 数据修改 / Data Modification

```java
// 创建向量 / Create vector
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0});

// 元素设置 / Element setting
IVector<Double> modified = v1.set(2, 10.0);          // 设置指定位置的值
IVector<Double> rangeSet = v1.setFromTo(1, 4, new double[]{10.0, 20.0, 30.0}); // 范围设置

// 数据填充 / Data filling
IVector<Double> filled = v1.fill(0.0);               // 填充所有元素为0
IVector<Double> clipped = v1.clip(0.0, 100.0);      // 裁剪值到指定范围
```

#### 累积操作 / Cumulative Operations

```java
// 创建向量 / Create vector
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0});

// 累积运算 / Cumulative operations
IVector<Double> cumsum = v1.cumsum();                 // 累积求和 / Cumulative sum
IVector<Double> cumprod = v1.cumprod();               // 累积乘积 / Cumulative product
```

#### 差分操作 / Difference Operations

```java
// 创建向量 / Create vector
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 4.0, 7.0, 11.0});

// 差分运算 / Difference operations
IVector<Double> diff = v1.diff();                     // 一阶差分 / First order difference
IVector<Double> diff2 = v1.diff(2);                   // 二阶差分 / Second order difference
```

#### 逻辑运算 / Logical Operations

```java
// 创建向量 / Create vectors
IVector<Double> v1 = Linalg.vector(new double[]{1.0, 0.0, 1.0, 0.0});
IVector<Double> v2 = Linalg.vector(new double[]{0.0, 1.0, 1.0, 0.0});

// 逻辑运算 / Logical operations
IVector<Double> logicalAnd = v1.logicalAnd(v2);       // 逻辑与 / Logical AND
IVector<Double> logicalOr = v1.logicalOr(v2);         // 逻辑或 / Logical OR
IVector<Double> logicalNot = v1.logicalNot();         // 逻辑非 / Logical NOT
IVector<Double> logicalXor = v1.logicalXor(v2);       // 逻辑异或 / Logical XOR
```

#### 线性代数扩展 / Extended Linear Algebra

```java
// 创建向量 / Create vector
IVector<Double> v1 = Linalg.vector(new double[]{3.0, 4.0, 0.0});

// 线性代数扩展 / Extended linear algebra
IVector<Double> normalized = v1.normalize();           // 向量归一化 / Vector normalization

// 向量转换为列矩阵 / Convert vector to column matrix
IMatrix<Double> columnMatrix = v1.asColumnVector();    // 将向量转换为列矩阵
```


## 使用示例 / Usage Examples

详细的代码示例请参考 [Vector-Examples.md](examples/Vector-Examples.md) 文档。

For detailed code examples, please refer to the [Vector-Examples.md](examples/Vector-Examples.md) document.

## 性能特性 / Performance Features

### 内存优化 / Memory Optimization
- 使用基本类型数组，避免装箱拆箱开销 / Use primitive type arrays to avoid boxing/unboxing overhead
- 高效的数组复制和切片操作 / Efficient array copying and slicing operations
- 最小化临时对象创建 / Minimize temporary object creation

### 算法优化 / Algorithm Optimization
- 优化的数学函数实现 / Optimized mathematical function implementations
- 高效的统计计算算法 / Efficient statistical computation algorithms
- 智能的索引计算 / Smart index calculations

### 类型安全 / Type Safety
- 强类型系统，避免运行时错误 / Strong type system to avoid runtime errors
- 泛型支持，编译时类型检查 / Generic support with compile-time type checking
- 参数验证和异常处理 / Parameter validation and exception handling
- 清晰的接口契约 / Clear interface contracts

## 注意事项 / Notes

1. **索引范围** / **Index Range**: 所有索引操作都会进行边界检查 / All index operations perform boundary checking
2. **空值处理** / **Null Handling**: 输入参数不能为null / Input parameters cannot be null
3. **精度** / **Precision**: 支持Float和Double类型，注意精度限制 / Supports Float and Double types, pay attention to precision limitations
4. **内存** / **Memory**: 大量向量操作时注意内存使用 / Pay attention to memory usage for large vector operations
5. **就地操作** / **In-place Operations**: 某些方法会修改原向量，注意是否需要复制 / Some methods modify the original vector, pay attention to whether copying is needed
6. **泛型类型** / **Generic Types**: 使用泛型时确保类型一致性 / Ensure type consistency when using generics

## 扩展性 / Extensibility

`IVector<T>` 接口设计支持扩展，可以轻松添加新的向量类型实现：
The `IVector<T>` interface is designed to support extensions, making it easy to add new vector type implementations:
- 稀疏向量 / Sparse vectors
- GPU加速向量 / GPU-accelerated vectors
- 分布式向量 / Distributed vectors
- 复数向量 / Complex vectors
- 高精度向量 / High-precision vectors



## 与NumPy功能对照表 / NumPy Function Mapping

| YiShape IVector | NumPy | 功能描述 / Description |
|-----------------|-------|----------------------|
| `Linalg.vector(double[])` | `np.array()` | 从数组创建向量 / Create vector from array |
| `Linalg.zeros(n)` | `np.zeros(n)` | 创建全零向量 / Create zero vector |
| `Linalg.ones(n)` | `np.ones(n)` | 创建全一向量 / Create ones vector |
| `Linalg.range(n)` | `np.arange(n)` | 创建范围向量 / Create range vector |
| `Linalg.range(start, end)` | `np.arange(start, end)` | 创建指定范围向量 / Create range with start/end |
| `Linalg.range(start, end, step)` | `np.arange(start, end, step)` | 创建带步长范围向量 / Create range with step |
| `Linalg.rand(n)` | `np.random.rand(n)` | 创建随机向量 / Create random vector |
| `Linalg.randn(n)` | `np.random.randn(n)` | 创建正态分布随机向量 / Create normal random vector |
| `Linalg.randn(n, mean, std, seed)` | `np.random.normal(mean, std, n)` | 创建带参数的正态分布随机向量 / Create normal random vector with parameters |
| `Linalg.linspace(start, stop, num)` | `np.linspace(start, stop, num)` | 创建线性空间向量 / Create linear space vector |
| `Linalg.logspace(start, stop, num)` | `np.logspace(start, stop, num)` | 创建对数空间向量 / Create logarithmic space vector |
| `v1.add(v2)` | `v1 + v2` | 向量加法 / Vector addition |
| `v1.sub(v2)` | `v1 - v2` | 向量减法 / Vector subtraction |
| `v1.multiply(v2)` | `v1 * v2` | 元素级乘法 / Element-wise multiplication |
| `v1.innerProduct(v2)、v1.dot(v2)` | `np.dot(v1, v2)` | 内积 / Inner product |
| `v1.mmul(matrix)`、`v1.dot(matrix)` | `np.dot(v1, matrix)` 或 `v1 @ matrix` | 行向量×矩阵；`dot` 与 `mmul` 等价 / Row × matrix; `dot` ≡ `mmul` |
| `v1.outer(v2)` | `np.outer(v1, v2)` | 外积 / Outer product |
| `v1.addScalar(s)` | `v1 + s` | 标量加法 / Scalar addition |
| `v1.subScalar(s)` | `v1 - s` | 标量减法 / Scalar subtraction |
| `v1.multiplyScalar(s)` | `v1 * s` | 标量乘法 / Scalar multiplication |
| `v1.divideByScalar(s)` | `v1 / s` | 标量除法 / Scalar division |
| `v1.sum()` | `np.sum(v1)` | 求和 / Sum |
| `v1.mean()` | `np.mean(v1)` | 均值 / Mean |
| `v1.var()` | `np.var(v1)` | 方差 / Variance |
| `v1.std()` | `np.std(v1)` | 标准差 / Standard deviation |
| `v1.min()` | `np.min(v1)` | 最小值 / Minimum |
| `v1.max()` | `np.max(v1)` | 最大值 / Maximum |
| `v1.argMin()` | `np.argmin(v1)` | 最小值索引 / Index of minimum |
| `v1.argMax()` | `np.argmax(v1)` | 最大值索引 / Index of maximum |
| `v1.norm1()` | `np.linalg.norm(v1, 1)` | L1范数 / L1 norm |
| `v1.norm2()` | `np.linalg.norm(v1, 2)` | L2范数 / L2 norm |
| `v1.normInf()` | `np.linalg.norm(v1, np.inf)` | 无穷范数 / Infinity norm |
| `v1.norm(p)` | `np.linalg.norm(v1, p)` | Lp范数 / Lp norm |
| `v1.square()` | `v1 ** 2` | 平方 / Square |
| `v1.sqrt()` | `np.sqrt(v1)` | 平方根 / Square root |
| `v1.exp()` | `np.exp(v1)` | 指数 / Exponential |
| `v1.log()` | `np.log(v1)` | 自然对数 / Natural logarithm |
| `v1.log10()` | `np.log10(v1)` | 以10为底对数 / Base-10 logarithm |
| `v1.abs()` | `np.abs(v1)` | 绝对值 / Absolute value |
| `v1.pow(exp)` | `v1 ** exp` | 幂运算 / Power |
| `v1.remainder(value)` | `v1 % value` | 取余运算 / Remainder |
| `v1.sin()` | `np.sin(v1)` | 正弦 / Sine |
| `v1.cos()` | `np.cos(v1)` | 余弦 / Cosine |
| `v1.tan()` | `np.tan(v1)` | 正切 / Tangent |
| `v1.arcsin()` | `np.arcsin(v1)` | 反正弦 / Arcsine |
| `v1.arccos()` | `np.arccos(v1)` | 反余弦 / Arccosine |
| `v1.arctan()` | `np.arctan(v1)` | 反正切 / Arctangent |
| `v1.sinh()` | `np.sinh(v1)` | 双曲正弦 / Hyperbolic sine |
| `v1.cosh()` | `np.cosh(v1)` | 双曲余弦 / Hyperbolic cosine |
| `v1.tanh()` | `np.tanh(v1)` | 双曲正切 / Hyperbolic tangent |
| `v1.round()` | `np.round(v1)` | 四舍五入 / Round |
| `v1.floor()` | `np.floor(v1)` | 向下取整 / Floor |
| `v1.ceil()` | `np.ceil(v1)` | 向上取整 / Ceiling |
| `v1.trunc()` | `np.trunc(v1)` | 截断取整 / Truncate |
| `v1.slice(start)` | `v1[start:]` | 切片 / Slicing |
| `v1.slice(start, end)` | `v1[start:end]` | 切片 / Slicing |
| `v1.slice(start, end, step)` | `v1[start:end:step]` | 带步长切片 / Slicing with step |
| `v1.slice("1:5")` | `v1[1:5]` | 字符串切片表达式 / String slice expression |
| `v1.slice(":-1")` | `v1[:-1]` | 负数索引切片 / Negative indexing slicing |
| `v1.slice("::2")` | `v1[::2]` | 步长切片 / Step slicing |
| `v1.slice("::-1")` | `v1[::-1]` | 反转向量 / Reverse vector |
| `v1.get(-1)` | `v1[-1]` | 负数索引访问 / Negative indexing access |
| `v1.fancyGet(indices)` | `v1[indices]` | 花式索引 / Fancy indexing |
| `v1.fancyGet([-1, -2, 0])` | `v1[[-1, -2, 0]]` | 负数花式索引 / Negative fancy indexing |
| `v1.booleanGet(mask)` | `v1[mask]` | 布尔索引 / Boolean indexing |
| `v1.gt(v2)` | `v1 > v2` | 大于比较，返回 `boolean[]` / Greater than comparison, returns `boolean[]` |
| `v1.lt(v2)` | `v1 < v2` | 小于比较，返回 `boolean[]` / Less than comparison, returns `boolean[]` |
| `v1.eq(v2)` | `v1 == v2` | 相等比较，返回 `boolean[]` / Equality comparison, returns `boolean[]` |
| `v1.sort()` | `np.sort(v1)` | 排序（返回新向量，不修改原向量）/ Sort (returns new vector, non-mutating) |
| `v1.reverse()` | `v1[::-1]` | 反转（返回新向量，不修改原向量）/ Reverse (returns new vector, non-mutating) |
| `v1.copy()` | `v1.copy()` | 深拷贝（返回独立副本）/ Deep copy (returns independent copy) |
| `v1.euclideanDistance(v2)` | `np.linalg.norm(v1 - v2)` | 欧几里得距离 / Euclidean distance |
| `v1.manhattanDistance(v2)` | `np.sum(np.abs(v1 - v2))` | 曼哈顿距离 / Manhattan distance |
| `v1.cosineSimilarity(v2)` | `np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))` | 余弦相似度 / Cosine similarity |
| `v1.where(cond, x, y)` | `np.where(cond, x, y)` | 条件选择 / Conditional selection |
| `v1.repeat(n)` | `np.repeat(v1, n)` | 重复 / Repeat |
| `v1.tile(n)` | `np.tile(v1, n)` | 平铺 / Tile |
| `v1.clip(min, max)` | `np.clip(v1, min, max)` | 裁剪 / Clip |
| `v1.ptp()` | `np.ptp(v1)` | 峰峰值 / Peak-to-peak |
| `v1.median()` | `np.median(v1)` | 中位数 / Median |
| `v1.percentile(q)` | `np.percentile(v1, q)` | 百分位数 / Percentile |
| `v1.q1()` | `np.percentile(v1, 25)` | 第一四分位数 / First quartile |
| `v1.q3()` | `np.percentile(v1, 75)` | 第三四分位数 / Third quartile |
| `v1.mode()` | `scipy.stats.mode(v1)` | 众数 / Mode |
| `v1.cumsum()` | `np.cumsum(v1)` | 累积求和 / Cumulative sum |
| `v1.cumprod()` | `np.cumprod(v1)` | 累积乘积 / Cumulative product |
| `v1.diff()` | `np.diff(v1)` | 差分 / Difference |
| `v1.diff(n=1)` | `np.diff(v1, n)` | n阶差分，默认一阶 / n-th order difference, default 1st |
| `v1.logicalAnd(v2)` | `np.logical_and(v1, v2)` | 逻辑与 / Logical AND |
| `v1.logicalOr(v2)` | `np.logical_or(v1, v2)` | 逻辑或 / Logical OR |
| `v1.logicalNot()` | `np.logical_not(v1)` | 逻辑非 / Logical NOT |
| `v1.logicalXor(v2)` | `np.logical_xor(v1, v2)` | 逻辑异或 / Logical XOR |
| `v1.normalize()` | `v1 / np.linalg.norm(v1)` | 向量归一化（返回新向量，不修改原向量）/ Vector normalization (returns new vector, non-mutating) |
| `v1.asColumnVector()` | `v1.reshape(-1, 1)` | 向量转换为列矩阵 / Convert vector to column matrix |


## 与NumPy的兼容性 / NumPy Compatibility

YiShape的向量操作设计参考了NumPy的API设计，提供了类似的接口和功能：
YiShape's vector operations design references NumPy's API design, providing similar interfaces and functionality:

- **`np.dot` 对照** / **`np.dot` mapping**: 一维×一维内积用 `innerProduct`/`dot(IVector)`；一维×二维（行向量×矩阵）用 `mmul(IMatrix)` 或 `dot(IMatrix)`（二者等价）。二维矩阵乘请使用 `IMatrix.mmul`，不要用 `frobeniusInnerProduct`。/ For 1D×1D use `innerProduct`/`dot(IVector)`; for 1D×2D row×matrix use `mmul(IMatrix)` or `dot(IMatrix)` (equivalent). For 2D matrix multiply use `IMatrix.mmul`, not `frobeniusInnerProduct`.
- **语法相似性** / **Syntax Similarity**: 方法命名和参数设计与NumPy保持一致 / Method naming and parameter design consistent with NumPy
- **功能对等性** / **Functional Equivalence**: 核心功能与NumPy向量操作完全对等 / Core functionality is completely equivalent to NumPy vector operations
- **性能优化** / **Performance Optimization**: 针对Java环境进行了性能优化 / Performance optimized for Java environment
- **类型安全** / **Type Safety**: 提供比NumPy更强的类型安全保障，支持泛型 / Provides stronger type safety guarantees than NumPy with generic support
- **统一入口** / **Unified Entry**: 通过Linalg工厂类提供统一的API入口 / Provides unified API entry through Linalg factory class

这使得从Python/NumPy迁移到Java/YiShape变得相对容易，同时保持了Java的类型安全和性能优势。
This makes migration from Python/NumPy to Java/YiShape relatively easy while maintaining Java's type safety and performance advantages.

---

**向量操作** - 数学计算的基础，让数据处理更高效！

---

**向量操作** - 数学计算的基础，让数据处理更高效！

**Vector Operations** - Mathematical computing foundation, making data processing more efficient!

## 常见问题 / FAQ

### Q1: Float 和 Double 选哪个？

大多数情况下用 `Double`（双精度）。`Float`（单精度）仅在：
- 数据量极大（如数百万维向量），内存紧张时
- 与外部系统交互（某些 C 库只支持 float32）
- 精度要求不高的近似计算

**注意**：混合使用 Float 和 Double 会触发自动类型转换，有性能开销。建议选定一种后统一使用。

### Q2: 向量运算后原向量被修改了？

YiShape Math 的向量操作**默认返回新向量，不修改原向量**（非 in-place）：

```java
IVector<Double> a = Linalg.vector(new double[]{1.0, 2.0, 3.0});
IVector<Double> b = a.multiplyScalar(2.0);  // a 不变，b 是新向量
System.out.println(a);  // [1.0, 2.0, 3.0]
System.out.println(b);  // [2.0, 4.0, 6.0]
```

若发现原向量被改变，检查是否误用了带 `Into` 后缀的方法（如 `a.multiplyScalarInto(target, scalar)`），这类方法会将结果写入目标向量。

### Q3: 向量索引从 0 还是 1 开始？

**从 0 开始**，与 NumPy/Python 一致：

```java
IVector<Double> v = Linalg.range(5);  // [0.0, 1.0, 2.0, 3.0, 4.0]
v.get(0);   // 第 1 个元素 = 0.0
v.get(4);   // 第 5 个元素 = 4.0
```

切片也使用左闭右开区间 `[start, end)`：

```java
v.slice(1, 4);  // [1.0, 2.0, 3.0]，不含索引 4
```

### Q4: 归一化（normalize）和标准化（standardize）有什么区别？

| 操作 | 公式 | 适用场景 |
|------|------|---------|
| **归一化** | x / \|\|x\|\|₂ | 距离度量（KNN、余弦相似度） |
| **标准化** | (x - μ) / σ | 梯度下降类算法（线性回归、神经网络） |

- 归一化：向量长度变为 1，方向不变
- 标准化：均值=0，标准差=1（按特征计算）

### Q5: `v.std()` 为什么不用 ddof 参数？

`IVector.std()` 默认计算的是**样本标准差**（分母为 n-1，ddof=1），与统计学中默认的无偏估计一致。若需要总体标准差：

```java
v.std();     // 样本标准差，ddof=1
v.std(0);    // 总体标准差，ddof=0
```
