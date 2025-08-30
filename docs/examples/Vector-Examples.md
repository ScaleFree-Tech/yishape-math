# 向量运算示例 (Vector Operation Examples)

## 概述 / Overview

本文档提供了 `IVector` 接口和 `RereVector` 实现类的详细使用示例，涵盖各种实际应用场景，帮助用户快速掌握向量操作的使用方法。

This document provides detailed usage examples for the `IVector` interface and `RereVector` implementation class, covering various practical application scenarios to help users quickly master vector operations.

## 基础示例 / Basic Examples

### 示例1：向量创建和基本操作 / Example 1: Vector Creation and Basic Operations

```java
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.RereVector;

public class VectorBasicExample {
    public static void main(String[] args) {
        // 1. 创建向量 / Create vectors
        System.out.println("=== 向量创建 / Vector Creation ===");
        
        // 从数组创建 / Create from array
        IVector v1 = IVector.of(new float[]{1, 2, 3, 4, 5});
        System.out.println("v1: " + v1);
        
        // 创建范围向量 / Create range vector
        IVector v2 = IVector.range(10);
        System.out.println("v2 (range 10): " + v2);
        
        // 创建特殊向量 / Create special vectors
        IVector v3 = IVector.ones(5);
        System.out.println("v3 (ones 5): " + v3);
        
        IVector v4 = IVector.zeros(5);
        System.out.println("v4 (zeros 5): " + v4);
        
        IVector v5 = IVector.random(5);
        System.out.println("v5 (random 5): " + v5);
        
        // 2. 基本数学运算 / Basic mathematical operations
        System.out.println("\n=== 基本数学运算 / Basic Mathematical Operations ===");
        
        // 向量加法 / Vector addition
        IVector sum = v1.add(v3);
        System.out.println("v1 + v3 = " + sum);
        
        // 向量减法 / Vector subtraction
        IVector diff = v1.sub(v3);
        System.out.println("v1 - v3 = " + diff);
        
        // 向量乘法 / Vector multiplication
        IVector product = v1.multiply(v3);
        System.out.println("v1 * v3 = " + product);
        
        // 内积 / Inner product
        float dotProduct = v1.innerProduct(v3);
        System.out.println("v1 · v3 = " + dotProduct);
        
        // 3. 标量运算 / Scalar operations
        System.out.println("\n=== 标量运算 / Scalar Operations ===");
        
        // 标量加法 / Scalar addition
        IVector scalarAdd = v1.addScalar(10.0f);
        System.out.println("v1 + 10 = " + scalarAdd);
        
        // 标量乘法 / Scalar multiplication
        IVector scalarMul = v1.multiplyScalar(2.0f);
        System.out.println("v1 * 2 = " + scalarMul);
        
        // 标量除法 / Scalar division
        IVector scalarDiv = v1.divideByScalar(2.0f);
        System.out.println("v1 / 2 = " + scalarDiv);
    }
}
```

### 示例2：统计运算 / Example 2: Statistical Operations

```java
public class VectorStatisticsExample {
    public static void main(String[] args) {
        // 创建测试数据 / Create test data
        float[] data = {2.5f, 4.8f, 6.2f, 3.9f, 5.1f, 7.3f, 4.2f, 6.8f, 3.5f, 5.9f};
        IVector vector = IVector.of(data);
        
        System.out.println("=== 统计运算示例 / Statistical Operations Example ===");
        System.out.println("数据: " + vector);
        
        // 基本统计 / Basic statistics
        float sum = vector.sum();
        float mean = vector.mean();
        float variance = vector.var();
        float std = vector.std();
        
        System.out.println("求和: " + sum);
        System.out.println("均值: " + mean);
        System.out.println("方差: " + variance);
        System.out.println("标准差: " + std);
        
        // 最值 / Min/Max
        float min = vector.min();
        float max = vector.max();
        int minIndex = vector.argMin();
        int maxIndex = vector.argMax();
        
        System.out.println("最小值: " + min + " (索引: " + minIndex + ")");
        System.out.println("最大值: " + max + " (索引: " + maxIndex + ")");
        
        // 范数 / Norms
        float norm1 = vector.norm1();
        float norm2 = vector.norm2();
        float normInf = vector.normInf();
        
        System.out.println("L1范数: " + norm1);
        System.out.println("L2范数: " + norm2);
        System.out.println("无穷范数: " + normInf);
        
        // 扩展统计 / Extended statistics
        float median = vector.median();
        float ptp = vector.ptp();
        
        System.out.println("中位数: " + median);
        System.out.println("峰峰值: " + ptp);
    }
}
```

### 示例3：切片和索引 / Example 3: Slicing and Indexing

```java
public class VectorSlicingExample {
    public static void main(String[] args) {
        // 创建测试向量 / Create test vector
        IVector vector = IVector.range(20);
        System.out.println("原始向量: " + vector);
        
        System.out.println("\n=== 切片操作 / Slicing Operations ===");
        
        // 基本切片 / Basic slicing
        IVector slice1 = vector.slice(5);
        System.out.println("slice(5): " + slice1);
        
        IVector slice2 = vector.slice(5, 15);
        System.out.println("slice(5, 15): " + slice2);
        
        IVector slice3 = vector.slice(0, 20, 2);
        System.out.println("slice(0, 20, 2): " + slice3);
        
        System.out.println("\n=== 花式索引 / Fancy Indexing ===");
        
        // 整数索引 / Integer indexing
        int[] indices = {0, 3, 7, 12, 18};
        IVector fancy1 = vector.fancyGet(indices);
        System.out.println("fancyGet([0, 3, 7, 12, 18]): " + fancy1);
        
        // 布尔索引 / Boolean indexing
        boolean[] mask = new boolean[20];
        for (int i = 0; i < 20; i++) {
            mask[i] = (i % 2 == 0); // 偶数索引 / Even indices
        }
        IVector fancy2 = vector.booleanGet(mask);
        System.out.println("booleanGet(even indices): " + fancy2);
        
        // 条件索引 / Conditional indexing
        System.out.println("\n=== 条件索引 / Conditional Indexing ===");
        
        // 找出大于10的元素 / Find elements greater than 10
        boolean[] greaterThan10 = new boolean[20];
        for (int i = 0; i < 20; i++) {
            greaterThan10[i] = (vector.get(i) > 10);
        }
        IVector result = vector.booleanGet(greaterThan10);
        System.out.println("大于10的元素: " + result);
    }
}
```

### 示例4：通用函数应用 / Example 4: Universal Function Applications

```java
public class VectorUniversalFunctionsExample {
    public static void main(String[] args) {
        // 创建测试向量 / Create test vector
        IVector vector = IVector.of(new float[]{0.5f, 1.0f, 1.5f, 2.0f, 2.5f});
        System.out.println("原始向量: " + vector);
        
        System.out.println("\n=== 数学函数 / Mathematical Functions ===");
        
        // 基本数学函数 / Basic mathematical functions
        IVector squared = vector.squre();
        System.out.println("平方: " + squared);
        
        IVector sqrt = vector.sqrt();
        System.out.println("平方根: " + sqrt);
        
        IVector exp = vector.exp();
        System.out.println("指数: " + exp);
        
        IVector log = vector.log();
        System.out.println("自然对数: " + log);
        
        IVector abs = vector.abs();
        System.out.println("绝对值: " + abs);
        
        System.out.println("\n=== 三角函数 / Trigonometric Functions ===");
        
        // 三角函数 / Trigonometric functions
        IVector sin = vector.sin();
        System.out.println("正弦: " + sin);
        
        IVector cos = vector.cos();
        System.out.println("余弦: " + cos);
        
        IVector tan = vector.tan();
        System.out.println("正切: " + tan);
        
        System.out.println("\n=== 舍入函数 / Rounding Functions ===");
        
        // 舍入函数 / Rounding functions
        IVector rounded = vector.round();
        System.out.println("四舍五入: " + rounded);
        
        IVector floored = vector.floor();
        System.out.println("向下取整: " + floored);
        
        IVector ceiled = vector.ceil();
        System.out.println("向上取整: " + ceiled);
        
        System.out.println("\n=== 统计函数 / Statistical Functions ===");
        
        // 标准化 / Standardization
        IVector standardized = vector.sub(vector.mean()).divideByScalar(vector.std());
        System.out.println("标准化: " + standardized);
        
        // 验证标准化结果 / Verify standardization results
        float stdMean = standardized.mean();
        float stdStd = standardized.std();
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
        IVector result = IVector.range(100)                    // 创建0-99的向量
            .slice(10, 90)                                      // 取10-89
            .multiplyScalar(0.1f)                               // 乘以0.1
            .addScala(5.0f)                                    // 加5
            .squre()                                            // 平方
            .sqrt()                                             // 开方
            .multiplyScalar(2.0f);                              // 乘以2
        
        System.out.println("链式操作结果: " + result);
        
        // 数据预处理管道 / Data preprocessing pipeline
        System.out.println("\n=== 数据预处理管道 / Data Preprocessing Pipeline ===");
        
        // 模拟传感器数据 / Simulate sensor data
        IVector sensorData = IVector.random(1000).multiplyScalar(100.0f).addScala(50.0f);
        
        // 数据预处理 / Data preprocessing
        IVector processedData = sensorData
            .sub(sensorData.mean())                           // 中心化
            .divideByScalar(sensorData.std())                    // 标准化
            .abs()                                              // 取绝对值
            .multiplyScalar(10.0f);                             // 放大10倍
        
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
        IVector v1 = IVector.of(new float[]{1, 2, 3, 4});
        IVector v2 = IVector.of(new float[]{5, 6, 7, 8});
        
        System.out.println("=== 高级操作示例 / Advanced Operations Example ===");
        
        // 距离计算 / Distance calculations
        float euclideanDist = v1.euclideanDistance(v2);
        float manhattanDist = v1.manhattanDistance(v2);
        float cosineSim = v1.cosineSimilarity(v2);
        
        System.out.println("欧几里得距离: " + euclideanDist);
        System.out.println("曼哈顿距离: " + manhattanDist);
        System.out.println("余弦相似度: " + cosineSim);
        
        // 条件操作 / Conditional operations
        boolean[] condition = {true, false, true, false};
        IVector x = IVector.of(new float[]{10, 20, 30, 40});
        IVector y = IVector.of(new float[]{100, 200, 300, 400});
        IVector result = v1.where(condition, x, y);
        
        System.out.println("条件选择结果: " + result);
        
        // 重复和连接 / Repeat and concatenation
        IVector repeated = v1.repeat(2);
        IVector tiled = v1.tile(2);
        
        System.out.println("重复结果: " + repeated);
        System.out.println("平铺结果: " + tiled);
        
        // 数据修改 / Data modification
        IVector clipped = v1.copy().clip(2.0f, 3.0f);
        IVector filled = v1.copy().fill(0.0f);
        
        System.out.println("裁剪结果: " + clipped);
        System.out.println("填充结果: " + filled);
        
        // 累积操作 / Cumulative operations
        IVector cumsum = v1.cumsum();
        IVector cumprod = v1.cumprod();
        
        System.out.println("累积求和: " + cumsum);
        System.out.println("累积乘积: " + cumprod);
        
        // 差分操作 / Difference operations
        IVector diff = v1.diff();
        IVector diff2 = v1.diff(2);
        
        System.out.println("一阶差分: " + diff);
        System.out.println("二阶差分: " + diff2);
        
        // 逻辑运算 / Logical operations
        IVector logicalAnd = v1.logicalAnd(v2);
        IVector logicalOr = v1.logicalOr(v2);
        IVector logicalNot = v1.logicalNot();
        
        System.out.println("逻辑与: " + logicalAnd);
        System.out.println("逻辑或: " + logicalOr);
        System.out.println("逻辑非: " + logicalNot);
        
        // 线性代数扩展 / Extended linear algebra
        IVector normalized = v1.normalize();
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
        IVector randomVec = IVector.random(1000);                    // 1000个[0,1)随机数
        IVector normalVec = IVector.randomNormal(1000, 0.0f, 1.0f); // 1000个标准正态分布随机数
        
        // 生成线性空间向量 / Generate linear space vectors
        IVector linVec = IVector.linspace(0.0f, 10.0f, 100);        // 100个等间距值
        IVector logVec = IVector.logspace(0.0f, 3.0f, 50);          // 50个对数等间距值
        
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
        IVector time = IVector.range(signalLength).multiplyScalar(0.01f); // 时间轴 / Time axis
        
        // 生成复合信号：正弦波 + 噪声 / Generate composite signal: sine wave + noise
        IVector signal = time.multiplyScalar(2 * (float)Math.PI * 10)  // 10Hz正弦波 / 10Hz sine wave
            .sin()
            .add(IVector.random(signalLength).multiplyScalar(0.1f)); // 添加噪声 / Add noise
        
        System.out.println("信号统计:");
        System.out.println("  长度: " + signal.length());
        System.out.println("  均值: " + signal.mean());
        System.out.println("  标准差: " + signal.std());
        System.out.println("  峰值: " + signal.max());
        
        // 信号滤波 / Signal filtering
        System.out.println("\n=== 信号滤波 / Signal Filtering ===");
        
        // 简单的移动平均滤波 / Simple moving average filter
        int windowSize = 20;
        IVector filteredSignal = movingAverageFilter(signal, windowSize);
        
        System.out.println("滤波后信号统计:");
        System.out.println("  均值: " + filteredSignal.mean());
        System.out.println("  标准差: " + filteredSignal.std());
        System.out.println("  峰值: " + filteredSignal.max());
        
        // 计算信噪比 / Calculate signal-to-noise ratio
        float snr = calculateSNR(signal, filteredSignal);
        System.out.println("信噪比改善: " + snr + " dB");
    }
    
    // 移动平均滤波 / Moving average filter
    private static IVector movingAverageFilter(IVector signal, int windowSize) {
        int length = signal.length();
        float[] filtered = new float[length];
        
        for (int i = 0; i < length; i++) {
            int start = Math.max(0, i - windowSize / 2);
            int end = Math.min(length, i + windowSize / 2 + 1);
            
            float sum = 0;
            int count = 0;
            for (int j = start; j < end; j++) {
                sum += signal.get(j);
                count++;
            }
            filtered[i] = sum / count;
        }
        
        return IVector.of(filtered);
    }
    
    // 计算信噪比 / Calculate signal-to-noise ratio
    private static float calculateSNR(IVector original, IVector filtered) {
        IVector noise = original.sub(filtered);
        float signalPower = filtered.squre().mean();
        float noisePower = noise.squre().mean();
        
        return 10 * (float) Math.log10(signalPower / noisePower);
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
        IVector image = generateTestImage(width, height);
        
        System.out.println("图像统计:");
        System.out.println("  尺寸: " + width + " x " + height);
        System.out.println("  像素数: " + totalPixels);
        System.out.println("  最小像素值: " + image.min());
        System.out.println("  最大像素值: " + image.max());
        System.out.println("  平均像素值: " + image.mean());
        
        // 图像增强 / Image enhancement
        System.out.println("\n=== 图像增强 / Image Enhancement ===");
        
        // 对比度增强 / Contrast enhancement
        IVector enhancedImage = enhanceContrast(image);
        
        System.out.println("增强后图像统计:");
        System.out.println("  最小像素值: " + enhancedImage.min());
        System.out.println("  最大像素值: " + enhancedImage.max());
        System.out.println("  平均像素值: " + enhancedImage.mean());
        
        // 边缘检测 / Edge detection
        System.out.println("\n=== 边缘检测 / Edge Detection ===");
        
        IVector edgeImage = detectEdges(image, width, height);
        
        System.out.println("边缘图像统计:");
        System.out.println("  边缘像素比例: " + (edgeImage.greaterThan(0.5f).sum() / totalPixels * 100) + "%");
    }
    
    // 生成测试图像 / Generate test image
    private static IVector generateTestImage(int width, int height) {
        float[] pixels = new float[width * height];
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 创建渐变 / Create gradient
                float gradient = (x + y) / (float) (width + height);
                
                // 添加噪声 / Add noise
                float noise = (float) (Math.random() - 0.5) * 0.1f;
                
                pixels[y * width + x] = Math.max(0, Math.min(1, gradient + noise));
            }
        }
        
        return IVector.of(pixels);
    }
    
    // 对比度增强 / Contrast enhancement
    private static IVector enhanceContrast(IVector image) {
        float min = image.min();
        float max = image.max();
        float range = max - min;
        
        if (range > 0) {
            return image.sub(min).divideByScalar(range);
        }
        return image;
    }
    
    // 简单的边缘检测 / Simple edge detection
    private static IVector detectEdges(IVector image, int width, int height) {
        float[] edges = new float[image.length()];
        
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = y * width + x;
                
                // 计算梯度 / Calculate gradient
                float gx = image.get(idx + 1) - image.get(idx - 1);
                float gy = image.get(idx + width) - image.get(idx - width);
                
                // 梯度幅值 / Gradient magnitude
                edges[idx] = (float) Math.sqrt(gx * gx + gy * gy);
            }
        }
        
        return IVector.of(edges);
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
        IVector largeData = IVector.random(dataSize);
        
        // 测量批量操作性能 / Measure batch operation performance
        long startTime = System.currentTimeMillis();
        
        // 批量操作 / Batch operations
        IVector result = largeData
            .addScalar(100.0f)
            .multiplyScalar(2.0f)
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
        
        float[] manualResult = new float[dataSize];
        for (int i = 0; i < dataSize; i++) {
            float value = largeData.get(i);
            value = value + 100.0f;
            value = value * 2.0f;
            value = value * value;
            value = (float) Math.sqrt(value);
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

本文档展示了 `IVector` 接口的各种使用方法和实际应用场景。通过合理使用向量操作，可以：

1. **提高代码可读性** / **Improve code readability**: 链式操作使代码更清晰
2. **提升性能** / **Enhance performance**: 批量操作比循环操作更快
3. **简化复杂计算** / **Simplify complex calculations**: 内置函数处理常见数学运算
4. **支持大规模数据** / **Support large-scale data**: 高效的内存管理和算法优化

建议在实际使用中：
- 优先使用向量化操作而不是循环
- 合理使用链式操作提高代码可读性
- 注意内存使用，避免创建过多临时对象
- 根据具体需求选择合适的向量创建方法

---

**向量运算示例** - 从基础到高级，掌握向量操作的精髓！

**Vector Operation Examples** - From basic to advanced, master the essence of vector operations!
