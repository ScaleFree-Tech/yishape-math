# 数学工具类示例 (Math Utilities Examples)

## 概述 / Overview

本文档提供了 `RereMathUtil` 类的详细使用示例，展示各种数学工具函数的使用方法。

## 基础数学函数示例 / Basic Mathematical Function Examples

### 类型转换函数 / Type Conversion Functions

```java
import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.IMatrix;

public class TypeConversionExample {
    public static void main(String[] args) {
        System.out.println("=== 类型转换函数示例 / Type Conversion Functions Example ===");
        
        // 1. 数组类型转换 / Array type conversion
        System.out.println("1. 数组类型转换 / Array Type Conversion");
        
        double[] doubleArray = {1.1, 2.2, 3.3, 4.4, 5.5};
        int[] intArray = {1, 2, 3, 4, 5};
        
        // double[] 转 float[] / Convert double[] to float[]
        float[] floatArray = RereMathUtil.doubleToFloat(doubleArray);
        System.out.println("double[] -> float[]: " + Arrays.toString(floatArray));
        
        // int[] 转 float[] / Convert int[] to float[]
        float[] intToFloatArray = RereMathUtil.intToFloat(intArray);
        System.out.println("int[] -> float[]: " + Arrays.toString(intToFloatArray));
        
        // float[] 转 double[] / Convert float[] to double[]
        double[] floatToDoubleArray = RereMathUtil.floatToDouble(floatArray);
        System.out.println("float[] -> double[]: " + Arrays.toString(floatToDoubleArray));
        
        // 2. 向量类型转换 / Vector type conversion
        System.out.println("\n2. 向量类型转换 / Vector Type Conversion");
        
        IVector floatVector = IVector.of(new float[]{1.1f, 2.2f, 3.3f});
        System.out.println("原始float向量: " + floatVector);
        
        // 转换为double数组 / Convert to double array
        double[] doubleVector = RereMathUtil.vectorToDoubleArray(floatVector);
        System.out.println("转换为double数组: " + Arrays.toString(doubleVector));
        
        // 转换为int数组 / Convert to int array
        int[] intVector = RereMathUtil.vectorToIntArray(floatVector);
        System.out.println("转换为int数组: " + Arrays.toString(intVector));
        
        // 3. 矩阵类型转换 / Matrix type conversion
        System.out.println("\n3. 矩阵类型转换 / Matrix Type Conversion");
        
        float[][] floatMatrixData = {{1.1f, 2.2f}, {3.3f, 4.4f}};
        IMatrix floatMatrix = IMatrix.of(floatMatrixData);
        System.out.println("原始float矩阵: " + floatMatrix);
        
        // 转换为double二维数组 / Convert to double 2D array
        double[][] doubleMatrix = RereMathUtil.matrixToDoubleArray(floatMatrix);
        System.out.println("转换为double矩阵:");
        for (double[] row : doubleMatrix) {
            System.out.println("  " + Arrays.toString(row));
        }
        
        // 转换为int二维数组 / Convert to int 2D array
        int[][] intMatrix = RereMathUtil.matrixToIntArray(floatMatrix);
        System.out.println("转换为int矩阵:");
        for (int[] row : intMatrix) {
            System.out.println("  " + Arrays.toString(row));
        }
    }
}
```

### 随机数生成 / Random Number Generation

```java
import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.IMatrix;

public class RandomNumberExample {
    public static void main(String[] args) {
        System.out.println("=== 随机数生成示例 / Random Number Generation Example ===");
        
        // 1. 生成随机向量 / Generate random vectors
        System.out.println("1. 随机向量生成 / Random Vector Generation");
        
        // 生成标准正态分布随机向量 / Generate standard normal random vector
        IVector normalVector = RereMathUtil.randn(10);
        System.out.println("标准正态分布随机向量 (长度10): " + normalVector);
        
        // 生成均匀分布随机向量 / Generate uniform random vector
        IVector uniformVector = RereMathUtil.rand(10);
        System.out.println("均匀分布随机向量 (长度10): " + uniformVector);
        
        // 生成指定范围的随机向量 / Generate random vector in specified range
        IVector rangeVector = RereMathUtil.rand(10, -5.0f, 5.0f);
        System.out.println("范围[-5,5]随机向量: " + rangeVector);
        
        // 2. 生成随机矩阵 / Generate random matrices
        System.out.println("\n2. 随机矩阵生成 / Random Matrix Generation");
        
        // 生成标准正态分布随机矩阵 / Generate standard normal random matrix
        IMatrix normalMatrix = RereMathUtil.randn(3, 4);
        System.out.println("标准正态分布随机矩阵 (3x4): " + normalMatrix);
        
        // 生成均匀分布随机矩阵 / Generate uniform random matrix
        IMatrix uniformMatrix = RereMathUtil.rand(4, 3);
        System.out.println("均匀分布随机矩阵 (4x3): " + uniformMatrix);
        
        // 生成指定范围的随机矩阵 / Generate random matrix in specified range
        IMatrix rangeMatrix = RereMathUtil.rand(3, 3, 0.0f, 10.0f);
        System.out.println("范围[0,10]随机矩阵 (3x3): " + rangeMatrix);
        
        // 3. 随机数种子设置 / Random number seed setting
        System.out.println("\n3. 随机数种子设置 / Random Number Seed Setting");
        
        // 设置种子以获得可重复的结果 / Set seed for reproducible results
        RereMathUtil.setSeed(42L);
        IVector reproducibleVector1 = RereMathUtil.randn(5);
        System.out.println("种子42的随机向量1: " + reproducibleVector1);
        
        IVector reproducibleVector2 = RereMathUtil.randn(5);
        System.out.println("种子42的随机向量2: " + reproducibleVector2);
        
        // 重置种子 / Reset seed
        RereMathUtil.setSeed(42L);
        IVector reproducibleVector3 = RereMathUtil.randn(5);
        System.out.println("重置种子后的随机向量: " + reproducibleVector3);
        System.out.println("向量1和向量3是否相同: " + reproducibleVector1.equals(reproducibleVector3));
    }
}
```

## 数学函数示例 / Mathematical Function Examples

### 基本数学函数 / Basic Mathematical Functions

```java
import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.IMatrix;

public class MathematicalFunctionsExample {
    public static void main(String[] args) {
        System.out.println("=== 数学函数示例 / Mathematical Functions Example ===");
        
        // 1. 三角函数 / Trigonometric functions
        System.out.println("1. 三角函数 / Trigonometric Functions");
        
        float[] angles = {0.0f, (float)Math.PI/6, (float)Math.PI/4, (float)Math.PI/3, (float)Math.PI/2};
        System.out.println("角度 (弧度): " + Arrays.toString(angles));
        
        for (float angle : angles) {
            float sin = RereMathUtil.sin(angle);
            float cos = RereMathUtil.cos(angle);
            float tan = RereMathUtil.tan(angle);
            
            System.out.printf("角度 %.2f: sin=%.4f, cos=%.4f, tan=%.4f%n", 
                            angle, sin, cos, tan);
        }
        
        // 2. 指数和对数函数 / Exponential and logarithmic functions
        System.out.println("\n2. 指数和对数函数 / Exponential and Logarithmic Functions");
        
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        System.out.println("数值: " + Arrays.toString(values));
        
        for (float value : values) {
            float exp = RereMathUtil.exp(value);
            float log = RereMathUtil.log(value);
            float log10 = RereMathUtil.log10(value);
            
            System.out.printf("数值 %.1f: exp=%.4f, ln=%.4f, log10=%.4f%n", 
                            value, exp, log, log10);
        }
        
        // 3. 幂函数和根函数 / Power and root functions
        System.out.println("\n3. 幂函数和根函数 / Power and Root Functions");
        
        float base = 2.0f;
        float[] exponents = {0.5f, 1.0f, 2.0f, 3.0f, 4.0f};
        System.out.println("底数: " + base);
        System.out.println("指数: " + Arrays.toString(exponents));
        
        for (float exp : exponents) {
            float power = RereMathUtil.pow(base, exp);
            float sqrt = RereMathUtil.sqrt(RereMathUtil.pow(base, exp));
            
            System.out.printf("指数 %.1f: %g^%.1f=%.4f, sqrt=%.4f%n", 
                            exp, base, exp, power, sqrt);
        }
        
        // 4. 舍入函数 / Rounding functions
        System.out.println("\n4. 舍入函数 / Rounding Functions");
        
        float[] roundValues = {1.4f, 1.5f, 1.6f, -1.4f, -1.5f, -1.6f};
        System.out.println("数值: " + Arrays.toString(roundValues));
        
        for (float value : roundValues) {
            float floor = RereMathUtil.floor(value);
            float ceil = RereMathUtil.ceil(value);
            float round = RereMathUtil.round(value);
            
            System.out.printf("数值 %.1f: floor=%.0f, ceil=%.0f, round=%.0f%n", 
                            value, floor, ceil, round);
        }
    }
}
```

### 统计函数 / Statistical Functions

```java
import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.IMatrix;

public class StatisticalFunctionsExample {
    public static void main(String[] args) {
        System.out.println("=== 统计函数示例 / Statistical Functions Example ===");
        
        // 1. 基本统计量 / Basic statistics
        System.out.println("1. 基本统计量 / Basic Statistics");
        
        float[] data = {1.2f, 3.4f, 5.6f, 7.8f, 9.0f, 2.1f, 4.3f, 6.5f, 8.7f, 0.9f};
        System.out.println("数据: " + Arrays.toString(data));
        
        float sum = RereMathUtil.sum(data);
        float mean = RereMathUtil.mean(data);
        float variance = RereMathUtil.variance(data);
        float stdDev = RereMathUtil.std(data);
        float min = RereMathUtil.min(data);
        float max = RereMathUtil.max(data);
        
        System.out.println("总和: " + sum);
        System.out.println("均值: " + mean);
        System.out.println("方差: " + variance);
        System.out.println("标准差: " + stdDev);
        System.out.println("最小值: " + min);
        System.out.println("最大值: " + max);
        
        // 2. 向量统计 / Vector statistics
        System.out.println("\n2. 向量统计 / Vector Statistics");
        
        IVector dataVector = IVector.of(data);
        System.out.println("数据向量: " + dataVector);
        
        float vectorSum = dataVector.sum();
        float vectorMean = dataVector.mean();
        float vectorVariance = dataVector.variance();
        float vectorStdDev = dataVector.std();
        
        System.out.println("向量总和: " + vectorSum);
        System.out.println("向量均值: " + vectorMean);
        System.out.println("向量方差: " + vectorVariance);
        System.out.println("向量标准差: " + vectorStdDev);
        
        // 3. 矩阵统计 / Matrix statistics
        System.out.println("\n3. 矩阵统计 / Matrix Statistics");
        
        float[][] matrixData = {
            {1.1f, 2.2f, 3.3f},
            {4.4f, 5.5f, 6.6f},
            {7.7f, 8.8f, 9.9f}
        };
        IMatrix dataMatrix = IMatrix.of(matrixData);
        System.out.println("数据矩阵: " + dataMatrix);
        
        float matrixSum = dataMatrix.sum();
        float matrixMean = dataMatrix.mean();
        float matrixVariance = dataMatrix.variance();
        
        System.out.println("矩阵总和: " + matrixSum);
        System.out.println("矩阵均值: " + matrixMean);
        System.out.println("矩阵方差: " + matrixVariance);
        
        // 4. 相关系数计算 / Correlation coefficient calculation
        System.out.println("\n4. 相关系数计算 / Correlation Coefficient Calculation");
        
        float[] x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] y = {2.0f, 4.0f, 5.0f, 4.0f, 5.0f};
        
        float correlation = RereMathUtil.correlation(x, y);
        System.out.println("X: " + Arrays.toString(x));
        System.out.println("Y: " + Arrays.toString(y));
        System.out.println("相关系数: " + correlation);
        
        // 5. 分位数计算 / Quantile calculation
        System.out.println("\n5. 分位数计算 / Quantile Calculation");
        
        float[] sortedData = Arrays.copyOf(data, data.length);
        Arrays.sort(sortedData);
        System.out.println("排序后数据: " + Arrays.toString(sortedData));
        
        float q25 = RereMathUtil.quantile(sortedData, 0.25f);
        float q50 = RereMathUtil.quantile(sortedData, 0.50f);
        float q75 = RereMathUtil.quantile(sortedData, 0.75f);
        
        System.out.println("25%分位数: " + q25);
        System.out.println("50%分位数 (中位数): " + q50);
        System.out.println("75%分位数: " + q75);
    }
}
```

## 数组操作示例 / Array Operation Examples

### 数组变换和操作 / Array Transformation and Operations

```java
import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.IMatrix;

public class ArrayOperationsExample {
    public static void main(String[] args) {
        System.out.println("=== 数组操作示例 / Array Operations Example ===");
        
        // 1. 数组排序 / Array sorting
        System.out.println("1. 数组排序 / Array Sorting");
        
        float[] unsortedData = {5.2f, 1.8f, 9.3f, 2.7f, 6.1f, 3.9f, 8.4f, 0.5f, 7.6f, 4.2f};
        System.out.println("未排序数据: " + Arrays.toString(unsortedData));
        
        // 升序排序 / Ascending sort
        float[] ascendingData = RereMathUtil.sort(unsortedData, true);
        System.out.println("升序排序: " + Arrays.toString(ascendingData));
        
        // 降序排序 / Descending sort
        float[] descendingData = RereMathUtil.sort(unsortedData, false);
        System.out.println("降序排序: " + Arrays.toString(descendingData));
        
        // 2. 数组查找 / Array searching
        System.out.println("\n2. 数组查找 / Array Searching");
        
        float searchValue = 6.1f;
        int index = RereMathUtil.indexOf(unsortedData, searchValue);
        System.out.println("查找值 " + searchValue + " 的索引: " + index);
        
        boolean contains = RereMathUtil.contains(unsortedData, searchValue);
        System.out.println("数组是否包含 " + searchValue + ": " + contains);
        
        // 3. 数组统计 / Array statistics
        System.out.println("\n3. 数组统计 / Array Statistics");
        
        float[] randomData = RereMathUtil.randn(100).toArray();
        System.out.println("生成100个标准正态分布随机数");
        
        float dataMean = RereMathUtil.mean(randomData);
        float dataStd = RereMathUtil.std(randomData);
        float dataSkew = RereMathUtil.skewness(randomData);
        float dataKurt = RereMathUtil.kurtosis(randomData);
        
        System.out.println("均值: " + dataMean);
        System.out.println("标准差: " + dataStd);
        System.out.println("偏度: " + dataSkew);
        System.out.println("峰度: " + dataKurt);
        
        // 4. 数组变换 / Array transformation
        System.out.println("\n4. 数组变换 / Array Transformation");
        
        float[] originalData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        System.out.println("原始数据: " + Arrays.toString(originalData));
        
        // 标准化 / Standardization
        float[] standardizedData = RereMathUtil.standardize(originalData);
        System.out.println("标准化后: " + Arrays.toString(standardizedData));
        
        // 归一化到[0,1] / Normalize to [0,1]
        float[] normalizedData = RereMathUtil.normalize(originalData);
        System.out.println("归一化后: " + Arrays.toString(normalizedData));
        
        // 5. 数组生成 / Array generation
        System.out.println("\n5. 数组生成 / Array Generation");
        
        // 生成等差数列 / Generate arithmetic sequence
        float[] arithmeticSeq = RereMathUtil.linspace(0.0f, 10.0f, 11);
        System.out.println("等差数列 [0,10]: " + Arrays.toString(arithmeticSeq));
        
        // 生成对数数列 / Generate logarithmic sequence
        float[] logSeq = RereMathUtil.logspace(0.1f, 10.0f, 10);
        System.out.println("对数数列 [0.1,10]: " + Arrays.toString(logSeq));
        
        // 生成网格 / Generate meshgrid
        float[] x = RereMathUtil.linspace(-2.0f, 2.0f, 5);
        float[] y = RereMathUtil.linspace(-2.0f, 2.0f, 5);
        Tuple2<float[][], float[][]> mesh = RereMathUtil.meshgrid(x, y);
        
        System.out.println("X网格:");
        for (float[] row : mesh._1) {
            System.out.println("  " + Arrays.toString(row));
        }
        System.out.println("Y网格:");
        for (float[] row : mesh._2) {
            System.out.println("  " + Arrays.toString(row));
        }
    }
}
```

## 实际应用示例 / Practical Application Examples

### 数据预处理 / Data Preprocessing

```java
import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.IMatrix;

public class DataPreprocessingExample {
    public static void main(String[] args) {
        System.out.println("=== 数据预处理示例 / Data Preprocessing Example ===");
        
        // 1. 模拟真实数据集 / Simulate real dataset
        System.out.println("1. 模拟真实数据集 / Simulate Real Dataset");
        
        // 生成包含噪声的数据 / Generate data with noise
        int samples = 1000;
        float[] trueValues = RereMathUtil.linspace(0.0f, 10.0f, samples);
        float[] noise = RereMathUtil.randn(samples).multiplyByScalar(0.5f).toArray();
        float[] noisyData = new float[samples];
        
        for (int i = 0; i < samples; i++) {
            noisyData[i] = trueValues[i] + noise[i];
        }
        
        System.out.println("生成 " + samples + " 个样本");
        System.out.println("数据范围: [" + RereMathUtil.min(noisyData) + ", " + RereMathUtil.max(noisyData) + "]");
        
        // 2. 数据清洗 / Data cleaning
        System.out.println("\n2. 数据清洗 / Data Cleaning");
        
        // 检测异常值 / Detect outliers
        float mean = RereMathUtil.mean(noisyData);
        float std = RereMathUtil.std(noisyData);
        float threshold = 3.0f; // 3σ原则 / 3-sigma rule
        
        int outlierCount = 0;
        for (float value : noisyData) {
            if (Math.abs(value - mean) > threshold * std) {
                outlierCount++;
            }
        }
        
        System.out.println("检测到的异常值数量: " + outlierCount);
        System.out.println("异常值比例: " + (outlierCount * 100.0f / samples) + "%");
        
        // 3. 数据标准化 / Data standardization
        System.out.println("\n3. 数据标准化 / Data Standardization");
        
        // Z-score标准化 / Z-score standardization
        float[] zScoreData = RereMathUtil.standardize(noisyData);
        float zScoreMean = RereMathUtil.mean(zScoreData);
        float zScoreStd = RereMathUtil.std(zScoreData);
        
        System.out.println("Z-score标准化后:");
        System.out.println("  均值: " + zScoreMean);
        System.out.println("  标准差: " + zScoreStd);
        
        // Min-Max归一化 / Min-Max normalization
        float[] minMaxData = RereMathUtil.normalize(noisyData);
        float minMaxMin = RereMathUtil.min(minMaxData);
        float minMaxMax = RereMathUtil.max(minMaxData);
        
        System.out.println("Min-Max归一化后:");
        System.out.println("  最小值: " + minMaxMin);
        System.out.println("  最大值: " + minMaxMax);
        
        // 4. 特征工程 / Feature engineering
        System.out.println("\n4. 特征工程 / Feature Engineering");
        
        // 创建多项式特征 / Create polynomial features
        float[] x = RereMathUtil.linspace(-2.0f, 2.0f, 100);
        float[] xSquared = new float[x.length];
        float[] xCubed = new float[x.length];
        
        for (int i = 0; i < x.length; i++) {
            xSquared[i] = x[i] * x[i];
            xCubed[i] = x[i] * x[i] * x[i];
        }
        
        // 计算特征间的相关性 / Calculate correlation between features
        float corrX_X2 = RereMathUtil.correlation(x, xSquared);
        float corrX_X3 = RereMathUtil.correlation(x, xCubed);
        float corrX2_X3 = RereMathUtil.correlation(xSquared, xCubed);
        
        System.out.println("特征相关性:");
        System.out.println("  x与x²: " + corrX_X2);
        System.out.println("  x与x³: " + corrX_X3);
        System.out.println("  x²与x³: " + corrX2_X3);
        
        // 5. 数据采样 / Data sampling
        System.out.println("\n5. 数据采样 / Data Sampling");
        
        // 随机采样 / Random sampling
        int sampleSize = 100;
        float[] sampledData = RereMathUtil.sample(noisyData, sampleSize);
        
        System.out.println("随机采样 " + sampleSize + " 个样本");
        System.out.println("采样后数据范围: [" + RereMathUtil.min(sampledData) + ", " + RereMathUtil.max(sampledData) + "]");
        
        // 分层采样 / Stratified sampling (简化版本)
        float[] stratifiedSample = RereMathUtil.stratifiedSample(noisyData, 5, sampleSize);
        System.out.println("分层采样 " + sampleSize + " 个样本");
    }
}
```

## 总结 / Summary

本文档展示了数学工具类的各种使用方法。建议在实际使用中：

1. **合理选择数据类型** / **Choose appropriate data types**
2. **注意数值精度** / **Pay attention to numerical precision**
3. **合理使用随机数** / **Use random numbers reasonably**
4. **注意内存使用** / **Pay attention to memory usage**

---

**数学工具类示例** - 从基础到应用，掌握数学工具的精髓！
