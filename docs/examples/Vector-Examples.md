# 向量运算示例 (Vector Operation Examples)

## 概述 / Overview

本文档提供了 `IVector<T>` 泛型接口的详细使用示例，涵盖各种实际应用场景，帮助用户快速掌握向量操作的使用方法。推荐使用 `Linalg` 工厂类来创建向量实例。

This document provides detailed usage examples for the `IVector<T>` generic interface, covering various practical application scenarios to help users quickly master vector operations. It is recommended to use the `Linalg` factory class to create vector instances.

## 基础示例 / Basic Examples

### 示例1：向量创建和基本操作 / Example 1: Vector Creation and Basic Operations

```java
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

public class VectorBasicExample {
    public static void main(String[] args) {
        // 1. 创建向量 / Create vectors
        System.out.println("=== 向量创建 / Vector Creation ===");

        // 从数组创建 / Create from array
        IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        System.out.println("v1: " + v1);

        // 创建范围向量 / Create range vector
        IVector<Double> v2 = Linalg.range(10);
        System.out.println("v2 (range 10): " + v2);

        // 创建特殊向量 / Create special vectors
        IVector<Double> v3 = Linalg.ones(5);
        System.out.println("v3 (ones 5): " + v3);

        IVector<Float> v4 = Linalg.zeros(5, Float.class);
        System.out.println("v4 (zeros 5): " + v4);

        IVector<Double> v5 = Linalg.rand(5);
        System.out.println("v5 (random 5): " + v5);

        // 从单个值创建向量 / Create vector from single values
        IVector<Double> v6 = Linalg.vector(5.0);
        System.out.println("v6 (single value): " + v6);

        IVector<Double> v7 = Linalg.vector(new double[]{1.0, 2.0});
        System.out.println("v7 (two values): " + v7);

        // 从基本类型数组创建 / Create from primitive arrays
        IVector<Double> v8 = Linalg.vector(new int[]{1, 2, 3, 4});
        System.out.println("v8 (from int array): " + v8);

        // 2. 基本数学运算 / Basic mathematical operations
        System.out.println("\n=== 基本数学运算 / Basic Mathematical Operations ===");

        // 向量加法 / Vector addition
        IVector<Double> sum = v1.add(v3);
        System.out.println("v1 + v3 = " + sum);

        // 向量减法 / Vector subtraction
        IVector<Double> diff = v1.sub(v3);
        System.out.println("v1 - v3 = " + diff);

        // 向量乘法 / Vector multiplication
        IVector<Double> product = v1.multiply(v3);
        System.out.println("v1 * v3 = " + product);

        // 内积 / Inner product
        Double dotProduct = v1.innerProduct(v3);
        System.out.println("v1 · v3 = " + dotProduct);

        // 外积 / Outer product
        IMatrix<Double> outerProduct = v1.outer(v3);
        System.out.println("v1 ⊗ v3 = " + outerProduct);

        // 3. 标量运算 / Scalar operations
        System.out.println("\n=== 标量运算 / Scalar Operations ===");

        // 标量加法 / Scalar addition
        IVector<Double> scalarAdd = v1.addScalar(10.0);
        System.out.println("v1 + 10 = " + scalarAdd);

        // 标量乘法 / Scalar multiplication
        IVector<Double> scalarMul = v1.multiplyScalar(2.0);
        System.out.println("v1 * 2 = " + scalarMul);

        // 标量除法 / Scalar division
        IVector<Double> scalarDiv = v1.divideByScalar(2.0);
        System.out.println("v1 / 2 = " + scalarDiv);
    }
}
```

### 示例2：统计运算 / Example 2: Statistical Operations

```java
public class VectorStatisticsExample {
    public static void main(String[] args) {
        // 创建测试数据 / Create test data
        double[] data = {2.5, 4.8, 6.2, 3.9, 5.1, 7.3, 4.2, 6.8, 3.5, 5.9};
        IVector<Double> vector = Linalg.vector(data);
        
        System.out.println("=== 统计运算示例 / Statistical Operations Example ===");
        System.out.println("数据: " + vector);
        
        // 基本统计 / Basic statistics
        Double sum = vector.sum();
        Double mean = vector.mean();
        Double variance = vector.var();
        Double std = vector.std();
        
        System.out.println("求和: " + sum);
        System.out.println("均值: " + mean);
        System.out.println("方差: " + variance);
        System.out.println("标准差: " + std);
        
        // 最值 / Min/Max
        Double min = vector.min();
        Double max = vector.max();
        int minIndex = vector.argMin();
        int maxIndex = vector.argMax();
        
        System.out.println("最小值: " + min + " (索引: " + minIndex + ")");
        System.out.println("最大值: " + max + " (索引: " + maxIndex + ")");
        
        // 范数 / Norms
        Double norm1 = vector.norm1();
        Double norm2 = vector.norm2();
        Double normInf = vector.normInf();
        
        System.out.println("L1范数: " + norm1);
        System.out.println("L2范数: " + norm2);
        System.out.println("无穷范数: " + normInf);
        
        // 扩展统计 / Extended statistics
        Double median = vector.median();
        Double ptp = vector.ptp();
        Double skewness = vector.skewness();
        Double kurtosis = vector.kurtosis();
        
        System.out.println("中位数: " + median);
        System.out.println("峰峰值: " + ptp);
        System.out.println("偏度: " + skewness);
        System.out.println("峰度: " + kurtosis);
    }
}
```

### 示例3：切片和索引 / Example 3: Slicing and Indexing

```java
public class VectorSlicingExample {
    public static void main(String[] args) {
        // 创建测试向量 / Create test vector
        IVector<Double> vector = Linalg.range(20);
        System.out.println("原始向量: " + vector);
        
        System.out.println("\n=== 切片操作 / Slicing Operations ===");
        
        // 基本切片 / Basic slicing
        IVector<Double> slice1 = vector.slice(5);
        System.out.println("slice(5): " + slice1);
        
        IVector<Double> slice2 = vector.slice(5, 15);
        System.out.println("slice(5, 15): " + slice2);
        
        IVector<Double> slice3 = vector.slice(0, 20, 2);
        System.out.println("slice(0, 20, 2): " + slice3);
        
        System.out.println("\n=== 花式索引 / Fancy Indexing ===");
        
        // 整数索引 / Integer indexing
        int[] indices = {0, 3, 7, 12, 18};
        IVector<Double> fancy1 = vector.fancyGet(indices);
        System.out.println("fancyGet([0, 3, 7, 12, 18]): " + fancy1);
        
        // 负数索引 / Negative indexing
        int[] negativeIndices = {-1, -5, -10, -15, -20}; // 倒数第1,5,10,15,20个元素
        IVector<Double> fancy3 = vector.fancyGet(negativeIndices);
        System.out.println("fancyGet([-1, -5, -10, -15, -20]): " + fancy3);
        
        // 混合索引 / Mixed indexing
        int[] mixedIndices = {0, -1, 5, -5, 10, -10}; // 正数和负数索引混合
        IVector<Double> fancy4 = vector.fancyGet(mixedIndices);
        System.out.println("fancyGet([0, -1, 5, -5, 10, -10]): " + fancy4);
        
        // 布尔索引 / Boolean indexing
        boolean[] mask = new boolean[20];
        for (int i = 0; i < 20; i++) {
            mask[i] = (i % 2 == 0); // 偶数索引 / Even indices
        }
        IVector<Double> fancy2 = vector.booleanGet(mask);
        System.out.println("booleanGet(even indices): " + fancy2);
        
        // 条件索引 / Conditional indexing
        System.out.println("\n=== 条件索引 / Conditional Indexing ===");
        
        // 找出大于10的元素 / Find elements greater than 10
        boolean[] greaterThan10 = new boolean[20];
        for (int i = 0; i < 20; i++) {
            greaterThan10[i] = (vector.get(i) > 10);
        }
        IVector<Double> result = vector.booleanGet(greaterThan10);
        System.out.println("大于10的元素: " + result);
    }
}
```

### 示例4：切片表达式和负数索引 / Example 4: Slice Expressions and Negative Indexing

```java
public class VectorSliceExpressionExample {
    public static void main(String[] args) {
        // 创建测试向量 / Create test vector
        IVector<Double> vector = Linalg.vector(new double[]{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
        System.out.println("原始向量: " + vector);
        System.out.println("向量长度: " + vector.length());
        
        System.out.println("\n=== 基本切片表达式 / Basic Slice Expressions ===");
        
        // 基本切片 / Basic slicing
        IVector<Double> slice1 = vector.slice("1:5");      // 从索引1到4（不包含5）
        System.out.println("slice(\"1:5\"): " + slice1);
        
        IVector<Double> slice2 = vector.slice("2:");       // 从索引2到末尾
        System.out.println("slice(\"2:\"): " + slice2);
        
        IVector<Double> slice3 = vector.slice(":5");       // 从开始到索引4（不包含5）
        System.out.println("slice(\":5\"): " + slice3);
        
        IVector<Double> slice4 = vector.slice("::2");      // 从开始到末尾，步长为2
        System.out.println("slice(\"::2\"): " + slice4);
        
        IVector<Double> slice5 = vector.slice("1:8:2");    // 从索引1到7，步长为2
        System.out.println("slice(\"1:8:2\"): " + slice5);
        
        System.out.println("\n=== 负数索引切片 / Negative Indexing Slicing ===");
        
        // 负数索引切片 / Negative indexing slicing
        IVector<Double> slice6 = vector.slice("-3:");      // 从倒数第3个到末尾
        System.out.println("slice(\"-3:\"): " + slice6);
        
        IVector<Double> slice7 = vector.slice(":-2");      // 从开始到倒数第2个（不包含）
        System.out.println("slice(\":-2\"): " + slice7);
        
        IVector<Double> slice8 = vector.slice("-5:-1");    // 从倒数第5个到倒数第1个（不包含）
        System.out.println("slice(\"-5:-1\"): " + slice8);
        
        IVector<Double> slice9 = vector.slice("-6:-1:2");  // 从倒数第6个到倒数第1个，步长为2
        System.out.println("slice(\"-6:-1:2\"): " + slice9);
        
        IVector<Double> slice10 = vector.slice("::-1");    // 反转向量
        System.out.println("slice(\"::-1\"): " + slice10);
        
        System.out.println("\n=== 负数索引访问 / Negative Indexing Access ===");
        
        // 负数索引访问 / Negative indexing access
        System.out.println("vector.get(0): " + vector.get(0));     // 第一个元素
        System.out.println("vector.get(-1): " + vector.get(-1));   // 最后一个元素
        System.out.println("vector.get(-2): " + vector.get(-2));   // 倒数第二个元素
        System.out.println("vector.get(-5): " + vector.get(-5));   // 倒数第五个元素
        
        System.out.println("\n=== 切片表达式边界情况 / Slice Expression Edge Cases ===");
        
        // 边界情况 / Edge cases
        IVector<Double> slice11 = vector.slice("0:0");     // 空切片
        System.out.println("slice(\"0:0\"): " + slice11 + " (长度: " + slice11.length() + ")");
        
        IVector<Double> slice12 = vector.slice("5:5");     // 空切片
        System.out.println("slice(\"5:5\"): " + slice12 + " (长度: " + slice12.length() + ")");
        
        IVector<Double> slice13 = vector.slice("1:1");     // 空切片
        System.out.println("slice(\"1:1\"): " + slice13 + " (长度: " + slice13.length() + ")");
        
        IVector<Double> slice14 = vector.slice("-1:-1");   // 空切片
        System.out.println("slice(\"-1:-1\"): " + slice14 + " (长度: " + slice14.length() + ")");
        
        System.out.println("\n=== 切片表达式错误处理 / Slice Expression Error Handling ===");
        
        try {
            // 无效的切片表达式 / Invalid slice expression
            vector.slice("1:2:3:4");  // 超过3个冒号
        } catch (IllegalArgumentException e) {
            System.out.println("捕获异常: " + e.getMessage());
        }
        
        try {
            // 无效的步长 / Invalid step
            vector.slice("1:5:0");    // 步长为0
        } catch (IllegalArgumentException e) {
            System.out.println("捕获异常: " + e.getMessage());
        }
        
        try {
            // 索引超出范围 / Index out of bounds
            vector.slice("1:20");     // 结束索引超出范围
        } catch (IllegalArgumentException e) {
            System.out.println("捕获异常: " + e.getMessage());
        }
        
        System.out.println("\n=== 实际应用示例 / Practical Application Examples ===");
        
        // 获取前一半和后一半 / Get first half and second half
        IVector<Double> firstHalf = vector.slice(":" + (vector.length() / 2));
        IVector<Double> secondHalf = vector.slice((vector.length() / 2) + ":");
        System.out.println("前一半: " + firstHalf);
        System.out.println("后一半: " + secondHalf);
        
        // 获取奇数位置和偶数位置的元素 / Get odd and even positioned elements
        IVector<Double> oddPositions = vector.slice("1::2");  // 奇数位置（索引1,3,5,7,9）
        IVector<Double> evenPositions = vector.slice("::2");  // 偶数位置（索引0,2,4,6,8）
        System.out.println("奇数位置: " + oddPositions);
        System.out.println("偶数位置: " + evenPositions);
        
        // 获取中间部分（去除首尾） / Get middle part (excluding first and last)
        IVector<Double> middle = vector.slice("1:-1");
        System.out.println("中间部分: " + middle);
        
        // 获取最后几个元素 / Get last few elements
        IVector<Double> lastThree = vector.slice("-3:");
        System.out.println("最后三个: " + lastThree);
    }
}
```

### 示例5：通用函数应用 / Example 5: Universal Function Applications

```java
public class VectorUniversalFunctionsExample {
    public static void main(String[] args) {
        // 创建测试向量 / Create test vector
        IVector<Double> vector = Linalg.vector(new double[]{0.5, 1.0, 1.5, 2.0, 2.5});
        System.out.println("原始向量: " + vector);
        
        System.out.println("\n=== 数学函数 / Mathematical Functions ===");
        
        // 基本数学函数 / Basic mathematical functions
        IVector<Double> squared = vector.squre();
        System.out.println("平方: " + squared);
        
        IVector<Double> sqrt = vector.sqrt();
        System.out.println("平方根: " + sqrt);
        
        IVector<Double> exp = vector.exp();
        System.out.println("指数: " + exp);
        
        IVector<Double> log = vector.log();
        System.out.println("自然对数: " + log);
        
        IVector<Double> abs = vector.abs();
        System.out.println("绝对值: " + abs);
        
        System.out.println("\n=== 三角函数 / Trigonometric Functions ===");
        
        // 三角函数 / Trigonometric functions
        IVector<Double> sin = vector.sin();
        System.out.println("正弦: " + sin);
        
        IVector<Double> cos = vector.cos();
        System.out.println("余弦: " + cos);
        
        IVector<Double> tan = vector.tan();
        System.out.println("正切: " + tan);
        
        System.out.println("\n=== 舍入函数 / Rounding Functions ===");
        
        // 舍入函数 / Rounding functions
        IVector<Double> rounded = vector.round();
        System.out.println("四舍五入: " + rounded);
        
        IVector<Double> floored = vector.floor();
        System.out.println("向下取整: " + floored);
        
        IVector<Double> ceiled = vector.ceil();
        System.out.println("向上取整: " + ceiled);
        
        System.out.println("\n=== 统计函数 / Statistical Functions ===");
        
        // 标准化 / Standardization
        IVector<Double> standardized = vector.sub(vector.mean()).divideByScalar(vector.std());
        System.out.println("标准化: " + standardized);
        
        // 验证标准化结果 / Verify standardization results
        Double stdMean = standardized.mean();
        Double stdStd = standardized.std();
        System.out.println("标准化后均值: " + stdMean + " (应该接近0)");
        System.out.println("标准化后标准差: " + stdStd + " (应该接近1)");
    }
}
```

### 示例5：链式操作 / Example 5: Method Chaining

```java
public class VectorChainingExample {
    public static void main(String[] args) {
        System.out.println("=== 链式操作示例 / Method Chaining Example ===");
        
        // 复杂的链式操作 / Complex method chaining
        IVector<Double> result = Linalg.range(100)                    // 创建0-99的向量
            .slice(10, 90)                                              // 取10-89
            .multiplyScalar(0.1)                                        // 乘以0.1
            .addScalar(5.0)                                             // 加5
            .squre()                                                    // 平方
            .sqrt()                                                     // 开方
            .multiplyScalar(2.0);                                       // 乘以2
        
        System.out.println("链式操作结果: " + result);
        
        // 数据预处理管道 / Data preprocessing pipeline
        System.out.println("\n=== 数据预处理管道 / Data Preprocessing Pipeline ===");
        
        // 模拟传感器数据 / Simulate sensor data
        IVector<Double> sensorData = Linalg.rand(1000).multiplyScalar(100.0).addScalar(50.0);
        
        // 数据预处理 / Data preprocessing
        IVector<Double> processedData = sensorData
            .sub(sensorData.mean())                                   // 中心化
            .divideByScalar(sensorData.std())                         // 标准化
            .abs()                                                    // 取绝对值
            .multiplyScalar(10.0);                                    // 放大10倍
        
        System.out.println("原始传感器数据统计:");
        System.out.println("  均值: " + sensorData.mean());
        System.out.println("  标准差: " + sensorData.std());
        System.out.println("  范围: [" + sensorData.min() + ", " + sensorData.max() + "]");
        
        System.out.println("处理后数据统计:");
        System.out.println("  均值: " + processedData.mean());
        System.out.println("  标准差: " + processedData.std());
        System.out.println("  范围: [" + processedData.min() + ", " + processedData.max() + "]");
    }
}
```

### 示例6：高级操作 / Example 6: Advanced Operations

```java
public class VectorAdvancedOperationsExample {
    public static void main(String[] args) {
        IVector<Double> v1 = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0});
        IVector<Double> v2 = Linalg.vector(new double[]{5.0, 6.0, 7.0, 8.0});
        
        System.out.println("=== 高级操作示例 / Advanced Operations Example ===");
        
        // 距离计算 / Distance calculations
        Double euclideanDist = v1.euclideanDistance(v2);
        Double manhattanDist = v1.manhattanDistance(v2);
        Double cosineSim = v1.cosineSimilarity(v2);
        
        System.out.println("欧几里得距离: " + euclideanDist);
        System.out.println("曼哈顿距离: " + manhattanDist);
        System.out.println("余弦相似度: " + cosineSim);
        
        // 条件操作 / Conditional operations
        boolean[] condition = {true, false, true, false};
        IVector<Double> x = Linalg.vector(new double[]{10.0, 20.0, 30.0, 40.0});
        IVector<Double> y = Linalg.vector(new double[]{100.0, 200.0, 300.0, 400.0});
        IVector<Double> result = v1.where(condition, x, y);
        
        System.out.println("条件选择结果: " + result);
        
        // 重复和连接 / Repeat and concatenation
        IVector<Double> repeated = v1.repeat(2);
        IVector<Double> tiled = v1.tile(2);
        
        System.out.println("重复结果: " + repeated);
        System.out.println("平铺结果: " + tiled);
        
        // 数据修改 / Data modification
        IVector<Double> clipped = v1.copy().clip(2.0, 3.0);
        IVector<Double> filled = v1.copy().fill(0.0);
        
        System.out.println("裁剪结果: " + clipped);
        System.out.println("填充结果: " + filled);
        
        // 累积操作 / Cumulative operations
        IVector<Double> cumsum = v1.cumsum();
        IVector<Double> cumprod = v1.cumprod();
        
        System.out.println("累积求和: " + cumsum);
        System.out.println("累积乘积: " + cumprod);
        
        // 差分操作 / Difference operations
        IVector<Double> diff = v1.diff();
        IVector<Double> diff2 = v1.diff(2);
        
        System.out.println("一阶差分: " + diff);
        System.out.println("二阶差分: " + diff2);
        
        // 逻辑运算 / Logical operations
        IVector<Double> logicalAnd = v1.logicalAnd(v2);
        IVector<Double> logicalOr = v1.logicalOr(v2);
        IVector<Double> logicalNot = v1.logicalNot();
        
        System.out.println("逻辑与: " + logicalAnd);
        System.out.println("逻辑或: " + logicalOr);
        System.out.println("逻辑非: " + logicalNot);
        
        // 线性代数扩展 / Extended linear algebra
        IVector<Double> normalized = v1.normalize();
        System.out.println("归一化结果: " + normalized);
        System.out.println("归一化后L2范数: " + normalized.norm2());
    }
}
```

### 示例7：随机向量生成 / Example 7: Random Vector Generation

```java
public class RandomVectorGenerationExample {
    public static void main(String[] args) {
        System.out.println("=== 随机向量生成示例 / Random Vector Generation Example ===");
        
        // 生成随机向量 / Generate random vectors
        IVector<Double> randomVec = Linalg.rand(1000);                    // 1000个[0,1)随机数
        IVector<Double> normalVec = Linalg.randn(1000, 0.0, 1.0); // 1000个标准正态分布随机数
        
        // 生成线性空间向量 / Generate linear space vectors
        IVector<Double> linVec = Linalg.linspace(0.0, 10.0, 100);        // 100个等间距值
        IVector<Double> logVec = Linalg.logspace(0.0, 3.0, 50);          // 50个对数等间距值
        IVector<Float> linVecF = Linalg.linspace(0.0f, 10.0f, 100, Float.class); // Float类型线性空间
        IVector<Float> logVecF = Linalg.logspace(0.0f, 3.0f, 50, Float.class);   // Float类型对数空间
        
        System.out.println("随机向量统计:");
        System.out.println("  均值: " + randomVec.mean());
        System.out.println("  标准差: " + randomVec.std());
        System.out.println("  范围: [" + randomVec.min() + ", " + randomVec.max() + "]");
        
        System.out.println("正态分布向量统计:");
        System.out.println("  均值: " + normalVec.mean());
        System.out.println("  标准差: " + normalVec.std());
        System.out.println("  范围: [" + normalVec.min() + ", " + normalVec.max() + "]");
        
        System.out.println("线性空间向量:");
        System.out.println("  前5个值: " + linVec.slice(5));
        System.out.println("  后5个值: " + linVec.slice(-5));
        
        System.out.println("对数空间向量:");
        System.out.println("  前5个值: " + logVec.slice(5));
        System.out.println("  后5个值: " + logVec.slice(-5));
    }
}
```

## 实际应用场景示例 / Real-world Application Scenarios

### 场景1：信号处理 / Scenario 1: Signal Processing

```java
public class SignalProcessingExample {
    public static void main(String[] args) {
        System.out.println("=== 信号处理示例 / Signal Processing Example ===");
        
        // 生成模拟信号 / Generate simulated signal
        int signalLength = 1000;
        IVector<Double> time = Linalg.range(signalLength).multiplyScalar(0.01); // 时间轴 / Time axis
        
        // 生成复合信号：正弦波 + 噪声 / Generate composite signal: sine wave + noise
        IVector<Double> signal = time.multiplyScalar(2 * Math.PI * 10)  // 10Hz正弦波 / 10Hz sine wave
            .sin()
            .add(Linalg.rand(signalLength).multiplyScalar(0.1)); // 添加噪声 / Add noise
        
        System.out.println("信号统计:");
        System.out.println("  长度: " + signal.length());
        System.out.println("  均值: " + signal.mean());
        System.out.println("  标准差: " + signal.std());
        System.out.println("  峰值: " + signal.max());
        
        // 信号滤波 / Signal filtering
        System.out.println("\n=== 信号滤波 / Signal Filtering ===");
        
        // 简单的移动平均滤波 / Simple moving average filter
        int windowSize = 20;
        IVector<Double> filteredSignal = movingAverageFilter(signal, windowSize);
        
        System.out.println("滤波后信号统计:");
        System.out.println("  均值: " + filteredSignal.mean());
        System.out.println("  标准差: " + filteredSignal.std());
        System.out.println("  峰值: " + filteredSignal.max());
        
        // 计算信噪比 / Calculate signal-to-noise ratio
        Double snr = calculateSNR(signal, filteredSignal);
        System.out.println("信噪比改善: " + snr + " dB");
    }
    
    // 移动平均滤波 / Moving average filter
    private static IVector<Double> movingAverageFilter(IVector<Double> signal, int windowSize) {
        int length = signal.length();
        double[] filtered = new double[length];
        
        for (int i = 0; i < length; i++) {
            int start = Math.max(0, i - windowSize / 2);
            int end = Math.min(length, i + windowSize / 2 + 1);
            
            double sum = 0;
            int count = 0;
            for (int j = start; j < end; j++) {
                sum += signal.get(j);
                count++;
            }
            filtered[i] = sum / count;
        }
        
        return Linalg.vector(filtered);
    }
    
    // 计算信噪比 / Calculate signal-to-noise ratio
    private static Double calculateSNR(IVector<Double> original, IVector<Double> filtered) {
        IVector<Double> noise = original.sub(filtered);
        Double signalPower = filtered.squre().mean();
        Double noisePower = noise.squre().mean();
        
        return 10 * Math.log10(signalPower / noisePower);
    }
}
```

### 场景2：图像处理 / Scenario 2: Image Processing

```java
public class ImageProcessingExample {
    public static void main(String[] args) {
        System.out.println("=== 图像处理示例 / Image Processing Example ===");
        
        // 模拟图像数据（灰度图）/ Simulate image data (grayscale)
        int width = 100;
        int height = 100;
        int totalPixels = width * height;
        
        // 生成测试图像：渐变 + 噪声 / Generate test image: gradient + noise
        IVector<Double> image = generateTestImage(width, height);
        
        System.out.println("图像统计:");
        System.out.println("  尺寸: " + width + " x " + height);
        System.out.println("  像素数: " + totalPixels);
        System.out.println("  最小像素值: " + image.min());
        System.out.println("  最大像素值: " + image.max());
        System.out.println("  平均像素值: " + image.mean());
        
        // 图像增强 / Image enhancement
        System.out.println("\n=== 图像增强 / Image Enhancement ===");
        
        // 对比度增强 / Contrast enhancement
        IVector<Double> enhancedImage = enhanceContrast(image);
        
        System.out.println("增强后图像统计:");
        System.out.println("  最小像素值: " + enhancedImage.min());
        System.out.println("  最大像素值: " + enhancedImage.max());
        System.out.println("  平均像素值: " + enhancedImage.mean());
        
        // 边缘检测 / Edge detection
        System.out.println("\n=== 边缘检测 / Edge Detection ===");
        
        IVector<Double> edgeImage = detectEdges(image, width, height);
        
        System.out.println("边缘图像统计:");
        System.out.println("  边缘像素比例: " + (edgeImage.greaterThan(0.5).sum() / totalPixels * 100) + "%");
    }
    
    // 生成测试图像 / Generate test image
    private static IVector<Double> generateTestImage(int width, int height) {
        double[] pixels = new double[width * height];
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 创建渐变 / Create gradient
                double gradient = (x + y) / (double) (width + height);
                
                // 添加噪声 / Add noise
                double noise = (Math.random() - 0.5) * 0.1;
                
                pixels[y * width + x] = Math.max(0, Math.min(1, gradient + noise));
            }
        }
        
        return Linalg.vector(pixels);
    }
    
    // 对比度增强 / Contrast enhancement
    private static IVector<Double> enhanceContrast(IVector<Double> image) {
        Double min = image.min();
        Double max = image.max();
        Double range = max - min;
        
        if (range > 0) {
            return image.sub(min).divideByScalar(range);
        }
        return image;
    }
    
    // 简单的边缘检测 / Simple edge detection
    private static IVector<Double> detectEdges(IVector<Double> image, int width, int height) {
        double[] edges = new double[image.length()];
        
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = y * width + x;
                
                // 计算梯度 / Calculate gradient
                double gx = image.get(idx + 1) - image.get(idx - 1);
                double gy = image.get(idx + width) - image.get(idx - width);
                
                // 梯度幅值 / Gradient magnitude
                edges[idx] = Math.sqrt(gx * gx + gy * gy);
            }
        }
        
        return Linalg.vector(edges);
    }
}
```

## 性能优化示例 / Performance Optimization Examples

### 示例8：批量操作优化 / Example 8: Batch Operation Optimization

```java
public class BatchOperationExample {
    public static void main(String[] args) {
        System.out.println("=== 批量操作优化示例 / Batch Operation Optimization Example ===");
        
        int dataSize = 1000000;
        System.out.println("数据大小: " + dataSize);
        
        // 生成大量数据 / Generate large amount of data
        IVector<Double> largeData = Linalg.rand(dataSize);
        
        // 测量批量操作性能 / Measure batch operation performance
        long startTime = System.currentTimeMillis();
        
        // 批量操作 / Batch operations
        IVector<Double> result = largeData
            .addScalar(100.0)
            .multiplyScalar(2.0)
            .squre()
            .sqrt();
        
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;
        
        System.out.println("批量操作耗时: " + duration + " ms");
        System.out.println("结果统计:");
        System.out.println("  均值: " + result.mean());
        System.out.println("  标准差: " + result.std());
        
        // 比较逐元素操作 / Compare element-wise operations
        startTime = System.currentTimeMillis();
        
        double[] manualResult = new double[dataSize];
        for (int i = 0; i < dataSize; i++) {
            double value = largeData.get(i);
            value = value + 100.0;
            value = value * 2.0;
            value = value * value;
            value = Math.sqrt(value);
            manualResult[i] = value;
        }
        
        endTime = System.currentTimeMillis();
        long manualDuration = endTime - startTime;
        
        System.out.println("逐元素操作耗时: " + manualDuration + " ms");
        System.out.println("性能提升: " + (manualDuration / (double) duration) + "x");
    }
}
```

## 总结 / Summary

本文档展示了 `IVector<T>` 接口的各种使用方法和实际应用场景。通过合理使用向量操作，可以：

1. **提高代码可读性** / **Improve code readability**: 链式操作使代码更清晰
2. **提升性能** / **Enhance performance**: 批量操作比循环操作更快
3. **简化复杂计算** / **Simplify complex calculations**: 内置函数处理常见数学运算
4. **支持大规模数据** / **Support large-scale data**: 高效的内存管理和算法优化
5. **类型安全** / **Type safety**: 泛型设计确保编译时类型检查

建议在实际使用中：
- 优先使用 `Linalg` 工厂类创建向量和矩阵
- 注意泛型类型声明，确保类型安全
- 优先使用向量化操作而不是循环
- 合理使用链式操作提高代码可读性
- 注意内存使用，避免创建过多临时对象
- 根据具体需求选择合适的向量创建方法

---

**向量运算示例** - 从基础到高级，掌握向量操作的精髓！

**Vector Operation Examples** - From basic to advanced, master the essence of vector operations!
