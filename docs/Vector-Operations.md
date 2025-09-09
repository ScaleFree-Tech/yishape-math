# 向量操作 (Vector Operations)

## 概述 / Overview

`IVector` 接口和 `RereVector` 实现类提供了完整的向量数学运算功能，包括基本数学运算、统计运算、切片索引、通用函数等。该接口设计简洁，功能强大，支持链式操作。

The `IVector` interface and `RereVector` implementation class provide comprehensive vector mathematical operations including basic mathematical operations, statistical operations, slicing and indexing, universal functions, and more. The interface features clean design, powerful functionality, and supports method chaining.

## 核心接口 / Core Interface

### IVector 接口 / IVector Interface

`IVector` 是向量操作的核心接口，定义了所有向量运算的抽象方法。

`IVector` is the core interface for vector operations, defining abstract methods for all vector operations.

## 主要功能 / Main Features

### 1. 向量创建 / Vector Creation

#### 静态工厂方法 / Static Factory Methods

```java
// 从float数组创建 / Create from float array
IVector v1 = IVector.of(new float[]{1, 2, 3, 4});

// 从Float包装类数组创建 / Create from Float wrapper array
IVector v2 = IVector.of(new Float[]{1.0f, 2.0f, 3.0f, 4.0f});

// 从double数组创建 / Create from double array
IVector v3 = IVector.of(new double[]{1.0, 2.0, 3.0, 4.0});

// 从int数组创建 / Create from int array
IVector v4 = IVector.of(new int[]{1, 2, 3, 4});

// 创建范围向量 / Create range vector
IVector v5 = IVector.range(10);        // [0, 1, 2, ..., 9]
IVector v6 = IVector.range(1, 11);     // [1, 2, 3, ..., 10]
IVector v7 = IVector.range(0, 20, 2); // [0, 2, 4, ..., 18]

// 创建特殊向量 / Create special vectors
IVector v8 = IVector.ones(5);          // [1, 1, 1, 1, 1]
IVector v9 = IVector.zeros(5);         // [0, 0, 0, 0, 0]
IVector v10 = IVector.random(5);       // 随机向量 / Random vector
IVector v11 = IVector.randomNormal(5, 0.0f, 1.0f); // 正态分布随机向量 / Normal distribution random vector

// 创建线性空间向量 / Create linear space vectors
IVector v12 = IVector.linspace(0.0f, 1.0f, 5);     // [0.0, 0.25, 0.5, 0.75, 1.0]
IVector v13 = IVector.logspace(0.0f, 2.0f, 4);     // [1.0, 10.0, 100.0, 1000.0]
```

### 2. 基本数学运算 / Basic Mathematical Operations

#### 向量间运算 / Vector-to-Vector Operations

```java
// 加法 / Addition
IVector sum = v1.add(v2);

// 减法 / Subtraction
IVector diff = v1.sub(v2);

// 元素级乘法 / Element-wise multiplication
IVector product = v1.multiply(v2);

// 内积 / Inner product (dot product)
float dotProduct = v1.innerProduct(v2);

// 向量与矩阵点积 / Vector-matrix dot product
IMatrix result = v1.dot(matrix);
```

#### 标量运算 / Scalar Operations

```java
// 标量加法 / Scalar addition
IVector result1 = v1.addScalar(5.0f);

// 标量减法 / Scalar subtraction
IVector result2 = v1.subScalar(2.0f);

// 标量乘法 / Scalar multiplication
IVector result3 = v1.multiplyScalar(3.0f);

// 标量除法 / Scalar division
IVector result4 = v1.divideByScalar(2.0f);
```

### 3. 统计运算 / Statistical Operations

```java
// 基本统计 / Basic statistics
float sum = v1.sum();           // 求和 / Sum
float mean = v1.mean();         // 均值 / Mean
float variance = v1.var();      // 方差 / Variance
float std = v1.std();           // 标准差 / Standard deviation
float prod = v1.prod();         // 元素乘积 / Product

// 最值 / Min/Max
float min = v1.min();           // 最小值 / Minimum
float max = v1.max();           // 最大值 / Maximum
int minIndex = v1.argMin();     // 最小值索引 / Index of minimum
int maxIndex = v1.argMax();     // 最大值索引 / Index of maximum

// 范数 / Norms
float norm1 = v1.norm1();       // L1范数 / L1 norm
float norm2 = v1.norm2();       // L2范数 / L2 norm
float normInf = v1.normInf();   // 无穷范数 / Infinity norm
float normP = v1.norm(3.0f);    // Lp范数 / Lp norm

// 其他统计量 / Other statistics
float ptp = v1.ptp();           // 峰峰值 / Peak-to-peak value
float median = v1.median();      // 中位数 / Median
float mode = v1.mode();          // 众数 / Mode
float percentile = v1.percentile(75.0f); // 75%分位数 / 75th percentile
```

### 4. 通用函数 / Universal Functions

```java
// 数学函数 / Mathematical functions
IVector squared = v1.squre();           // 平方 / Square
IVector sqrt = v1.sqrt();               // 平方根 / Square root
IVector exp = v1.exp();                 // 指数 / Exponential
IVector log = v1.log();                 // 自然对数 / Natural logarithm
IVector log10 = v1.log10();             // 以10为底对数 / Base-10 logarithm
IVector abs = v1.abs();                 // 绝对值 / Absolute value
IVector pow = v1.pow(2.0f);             // 幂运算 / Power
IVector remainder = v1.remainder(2.0f); // 取余运算 / Remainder

// 三角函数 / Trigonometric functions
IVector sin = v1.sin();                 // 正弦 / Sine
IVector cos = v1.cos();                 // 余弦 / Cosine
IVector tan = v1.tan();                 // 正切 / Tangent
IVector arcsin = v1.arcsin();           // 反正弦 / Arcsine
IVector arccos = v1.arccos();           // 反余弦 / Arccosine
IVector arctan = v1.arctan();           // 反正切 / Arctangent

// 双曲函数 / Hyperbolic functions
IVector sinh = v1.sinh();               // 双曲正弦 / Hyperbolic sine
IVector cosh = v1.cosh();               // 双曲余弦 / Hyperbolic cosine
IVector tanh = v1.tanh();               // 双曲正切 / Hyperbolic tangent

// 舍入函数 / Rounding functions
IVector round = v1.round();             // 四舍五入 / Round
IVector floor = v1.floor();             // 向下取整 / Floor
IVector ceil = v1.ceil();               // 向上取整 / Ceiling
IVector trunc = v1.trunc();             // 截断取整 / Truncate
```

### 5. 切片和索引 / Slicing and Indexing

#### 范围切片 / Range Slicing

```java
// 基本切片 / Basic slicing
IVector slice1 = v1.slice(2);          // 从索引2到末尾 / From index 2 to end
IVector slice2 = v1.slice(1, 4);       // 从索引1到4（不包含） / From index 1 to 4 (exclusive)
IVector slice3 = v1.slice(0, 6, 2);    // 从索引0到6，步长为2 / From index 0 to 6 with step 2

// 负数索引切片 / Negative indexing slicing
IVector slice4 = v1.slice(-3);         // 从倒数第3个到末尾 / From 3rd to last to end
IVector slice5 = v1.slice(1, -1);      // 从索引1到倒数第1个（不包含） / From index 1 to 2nd to last (exclusive)
IVector slice6 = v1.slice(-4, -1, 2);  // 从倒数第4个到倒数第1个，步长为2 / From 4th to last to 2nd to last with step 2

// 字符串切片表达式 / String slice expressions
IVector slice7 = v1.slice("1:4");      // 从索引1到4（不包含） / From index 1 to 4 (exclusive)
IVector slice8 = v1.slice(":-1");      // 从开始到倒数第1个（不包含） / From start to 2nd to last (exclusive)
IVector slice9 = v1.slice("::2");      // 从开始到末尾，步长为2 / From start to end with step 2
IVector slice10 = v1.slice("1:-1:2");  // 从索引1到倒数第1个，步长为2 / From index 1 to 2nd to last with step 2

// 花式索引 / Fancy indexing
IVector fancy1 = v1.fancyGet(new int[]{0, 2, 3});  // 获取索引0, 2, 3的元素
IVector fancy2 = v1.fancyGet(new int[]{-1, -2, 0}); // 获取倒数第1、2个和第一个元素

// 布尔索引 / Boolean indexing
boolean[] mask = {true, false, true, false};
IVector fancy3 = v1.booleanGet(mask);  // 获取mask为true的元素
```

#### 切片表达式语法 / Slice Expression Syntax

YiShape支持类似NumPy的切片表达式语法，支持负数索引：

YiShape supports NumPy-like slice expression syntax with negative indexing:

```java
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
// 对于长度为5的向量 [0, 1, 2, 3, 4]
// For a vector of length 5 [0, 1, 2, 3, 4]

// 正数索引 / Positive indices: 0, 1, 2, 3, 4
// 负数索引 / Negative indices: -5, -4, -3, -2, -1

v1.get(0);    // 获取第一个元素 / Get first element
v1.get(-1);   // 获取最后一个元素 / Get last element
v1.get(-2);   // 获取倒数第二个元素 / Get second to last element

v1.slice("1:-1");  // 从索引1到倒数第1个（不包含），即[1, 2, 3]
v1.slice("-3:-1"); // 从倒数第3个到倒数第1个（不包含），即[2, 3]
```

### 6. 比较运算 / Comparison Operations

```java
// 元素级比较 / Element-wise comparison
boolean[] greater = v1.greaterThan(v2);      // 大于 / Greater than
boolean[] less = v1.lessThan(v2);            // 小于 / Less than
boolean[] equal = v1.equals(v2);             // 等于 / Equal
```

### 7. 数据转换 / Data Conversion

```java
// 类型转换 / Type conversion
float[] floatArray = v1.getData();         // 获取float数组 / Get float array
double[] doubleArray = v1.asDoubleArray(); // 转换为double数组 / Convert to double array
int[] intArray = v1.asIntArray();          // 转换为int数组 / Convert to int array

// 数据访问 / Data access
float element = v1.get(2);                 // 获取指定索引的元素 / Get element at specific index
int length = v1.length();                  // 获取向量长度 / Get vector length
```

### 8. 高级操作 / Advanced Operations

#### 距离计算 / Distance Calculations

```java
// 距离度量 / Distance metrics
float euclideanDist = v1.euclideanDistance(v2);  // 欧几里得距离 / Euclidean distance
float manhattanDist = v1.manhattanDistance(v2);  // 曼哈顿距离 / Manhattan distance
float cosineSim = v1.cosineSimilarity(v2);       // 余弦相似度 / Cosine similarity
```

#### 条件操作 / Conditional Operations

```java
// 条件选择 / Conditional selection
boolean[] condition = {true, false, true, false};
IVector result = v1.where(condition, 10.0f, 20.0f);  // 根据条件选择值
IVector result2 = v1.where(condition, v2, v3);        // 根据条件选择向量
```

#### 重复和连接 / Repeat and Concatenation

```java
// 重复操作 / Repeat operations
IVector repeated = v1.repeat(3);              // 重复向量3次
IVector tiled = v1.tile(2);                   // 平铺向量2次
```

#### 排序和查找 / Sorting and Searching

```java
// 排序操作 / Sorting operations
IVector sorted = v1.sort();                   // 升序排序 / Ascending sort
IVector reversed = v1.reverse();               // 反转向量 / Reverse vector
```

#### 数据修改 / Data Modification

```java
// 元素设置 / Element setting
IVector modified = v1.set(2, 10.0f);          // 设置指定位置的值
IVector rangeSet = v1.setFromTo(1, 4, new float[]{10, 20, 30}); // 范围设置

// 数据填充 / Data filling
IVector filled = v1.fill(0.0f);               // 填充所有元素为0
IVector clipped = v1.clip(0.0f, 100.0f);     // 裁剪值到指定范围
```

#### 累积操作 / Cumulative Operations

```java
// 累积运算 / Cumulative operations
IVector cumsum = v1.cumsum();                 // 累积求和 / Cumulative sum
IVector cumprod = v1.cumprod();               // 累积乘积 / Cumulative product
```

#### 差分操作 / Difference Operations

```java
// 差分运算 / Difference operations
IVector diff = v1.diff();                     // 一阶差分 / First order difference
IVector diff2 = v1.diff(2);                   // 二阶差分 / Second order difference
```

#### 逻辑运算 / Logical Operations

```java
// 逻辑运算 / Logical operations
IVector logicalAnd = v1.logicalAnd(v2);       // 逻辑与 / Logical AND
IVector logicalOr = v1.logicalOr(v2);         // 逻辑或 / Logical OR
IVector logicalNot = v1.logicalNot();         // 逻辑非 / Logical NOT
IVector logicalXor = v1.logicalXor(v2);       // 逻辑异或 / Logical XOR
```

#### 线性代数扩展 / Extended Linear Algebra

```java
// 线性代数扩展 / Extended linear algebra
IVector normalized = v1.normalize();           // 向量归一化 / Vector normalization
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
- 参数验证和异常处理 / Parameter validation and exception handling
- 清晰的接口契约 / Clear interface contracts

## 注意事项 / Notes

1. **索引范围** / **Index Range**: 所有索引操作都会进行边界检查 / All index operations perform boundary checking
2. **空值处理** / **Null Handling**: 输入参数不能为null / Input parameters cannot be null
3. **精度** / **Precision**: 使用float类型，注意精度限制 / Uses float type, pay attention to precision limitations
4. **内存** / **Memory**: 大量向量操作时注意内存使用 / Pay attention to memory usage for large vector operations
5. **就地操作** / **In-place Operations**: 某些方法会修改原向量，注意是否需要复制 / Some methods modify the original vector, pay attention to whether copying is needed

## 扩展性 / Extensibility

`IVector` 接口设计支持扩展，可以轻松添加新的向量类型实现：
The `IVector` interface is designed to support extensions, making it easy to add new vector type implementations:
- 稀疏向量 / Sparse vectors
- GPU加速向量 / GPU-accelerated vectors
- 分布式向量 / Distributed vectors
- 复数向量 / Complex vectors
- 高精度向量 / High-precision vectors



## 与NumPy功能对照表 / NumPy Function Mapping

| YiShape IVector | NumPy | 功能描述 / Description |
|-----------------|-------|----------------------|
| `IVector.of(float[])` | `np.array()` | 从数组创建向量 / Create vector from array |
| `IVector.zeros(n)` | `np.zeros(n)` | 创建全零向量 / Create zero vector |
| `IVector.ones(n)` | `np.ones(n)` | 创建全一向量 / Create ones vector |
| `IVector.range(n)` | `np.arange(n)` | 创建范围向量 / Create range vector |
| `IVector.range(start, end)` | `np.arange(start, end)` | 创建指定范围向量 / Create range with start/end |
| `IVector.range(start, end, step)` | `np.arange(start, end, step)` | 创建带步长范围向量 / Create range with step |
| `IVector.rand(n)` | `np.random.rand(n)` | 创建随机向量 / Create random vector |
| `IVector.randn(n, mean, std)` | `np.random.randn(mean, std, n)` | 创建正态分布随机向量 / Create normal random vector |
| `IVector.linspace(start, stop, num)` | `np.linspace(start, stop, num)` | 创建线性空间向量 / Create linear space vector |
| `IVector.logspace(start, stop, num)` | `np.logspace(start, stop, num)` | 创建对数空间向量 / Create logarithmic space vector |
| `v1.add(v2)` | `v1 + v2` | 向量加法 / Vector addition |
| `v1.sub(v2)` | `v1 - v2` | 向量减法 / Vector subtraction |
| `v1.multiply(v2)` | `v1 * v2` | 元素级乘法 / Element-wise multiplication |
| `v1.innerProduct(v2)、v1.dot(v2)` | `np.dot(v1, v2)` | 内积 / Inner product |
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
| `v1.squre()` | `v1 ** 2` | 平方 / Square |
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
| `v1.greaterThan(v2)` | `v1 > v2` | 大于比较 / Greater than comparison |
| `v1.lessThan(v2)` | `v1 < v2` | 小于比较 / Less than comparison |
| `v1.equals(v2)` | `v1 == v2` | 相等比较 / Equality comparison |
| `v1.sort()` | `np.sort(v1)` | 排序 / Sort |
| `v1.reverse()` | `v1[::-1]` | 反转 / Reverse |
| `v1.copy()` | `v1.copy()` | 复制 / Copy |
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
| `v1.mode()` | `scipy.stats.mode(v1)` | 众数 / Mode |
| `v1.cumsum()` | `np.cumsum(v1)` | 累积求和 / Cumulative sum |
| `v1.cumprod()` | `np.cumprod(v1)` | 累积乘积 / Cumulative product |
| `v1.diff()` | `np.diff(v1)` | 差分 / Difference |
| `v1.diff(n)` | `np.diff(v1, n)` | n阶差分 / n-th order difference |
| `v1.logicalAnd(v2)` | `np.logical_and(v1, v2)` | 逻辑与 / Logical AND |
| `v1.logicalOr(v2)` | `np.logical_or(v1, v2)` | 逻辑或 / Logical OR |
| `v1.logicalNot()` | `np.logical_not(v1, v2)` | 逻辑非 / Logical NOT |
| `v1.logicalXor(v2)` | `np.logical_xor(v1, v2)` | 逻辑异或 / Logical XOR |
| `v1.normalize()` | `v1 / np.linalg.norm(v1)` | 向量归一化 / Vector normalization |


## 与NumPy的兼容性 / NumPy Compatibility

YiShape的向量操作设计参考了NumPy的API设计，提供了类似的接口和功能：
YiShape's vector operations design references NumPy's API design, providing similar interfaces and functionality:

- **语法相似性** / **Syntax Similarity**: 方法命名和参数设计与NumPy保持一致 / Method naming and parameter design consistent with NumPy
- **功能对等性** / **Functional Equivalence**: 核心功能与NumPy向量操作完全对等 / Core functionality is completely equivalent to NumPy vector operations
- **性能优化** / **Performance Optimization**: 针对Java环境进行了性能优化 / Performance optimized for Java environment
- **类型安全** / **Type Safety**: 提供比NumPy更强的类型安全保障 / Provides stronger type safety guarantees than NumPy

这使得从Python/NumPy迁移到Java/YiShape变得相对容易，同时保持了Java的类型安全和性能优势。
This makes migration from Python/NumPy to Java/YiShape relatively easy while maintaining Java's type safety and performance advantages.

---

**向量操作** - 数学计算的基础，让数据处理更高效！

**Vector Operations** - The foundation of mathematical computing, making data processing more efficient!
