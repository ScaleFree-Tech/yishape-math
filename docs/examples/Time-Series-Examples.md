# 时间序列分析示例 (Time Series Analysis Examples)

## 概述 / Overview

本文档按照从简单到复杂的顺序，系统性地编排了时间序列分析包的详细使用示例。每个级别都包含相应的理论背景、实践示例和进阶指导。

This document systematically organizes detailed usage examples for the time series analysis package in order from simple to complex. Each level includes corresponding theoretical background, practical examples, and advanced guidance.

---

## 第一部分：入门基础 (Level 1 - 基础入门) / Part 1: Beginner Level (Level 1 - Basic Introduction)

### 1.1 环境准备和基本概念 / Environment Setup and Basic Concepts

#### 导入必要的类 / Import Required Classes

```java
import com.reremouse.lab.math.timeseries.*;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
```

#### 创建时间序列数据 / Creating Time Series Data

```java
public class BasicTimeSeriesExample {
    public static void main(String[] args) {
        // 创建示例数据 / Create sample data
        IVector<Double> data = Linalg.randn(100);
        
        // 使用Series工厂类创建时间序列 / Create time series using Series factory class
        TimeSeriesData timeSeries = Series.createTimeSeries(data, "temperature");
        
        // 使用TimeSeriesData静态工厂方法 / Using TimeSeriesData static factory methods
        TimeSeriesData timeSeries2 = TimeSeriesData.of(data, "temperature2");
        
        // 使用构建器模式 / Using builder pattern
        TimeSeriesData timeSeries3 = TimeSeriesData.builder()
            .data(data, "temperature3")
            .samplingRate(1.0)
            .build();
        
        // 创建示例时间序列 / Create sample time series
        TimeSeriesData sampleSeries = TimeSeriesData.sample(100, "sample");
        TimeSeriesData sineWave = TimeSeriesData.sineWave(100, 2.0, "sine");
        
        // 基本属性 / Basic properties
        System.out.println("=== 时间序列基本信息 / Time Series Basic Information ===");
        System.out.println("长度: " + timeSeries.getLength() + " / Length: " + timeSeries.getLength());
        System.out.println("变量数: " + timeSeries.getNumVariables() + " / Variables: " + timeSeries.getNumVariables());
        System.out.println("采样率: " + timeSeries.getSamplingRate() + " / Sampling Rate: " + timeSeries.getSamplingRate());
        System.out.println("是否单变量: " + timeSeries.isUnivariate() + " / Is Univariate: " + timeSeries.isUnivariate());
        System.out.println("是否多变量: " + timeSeries.isMultivariate() + " / Is Multivariate: " + timeSeries.isMultivariate());
        
        // 获取数据 / Get data
        IVector<Double> temperature = timeSeries.getVariable("temperature");
        System.out.println("温度数据前5个值: " + temperature.slice(0, 5) + " / First 5 temperature values: " + temperature.slice(0, 5));
        
        // 获取统计信息 / Get statistics
        IMatrix<Double> stats = timeSeries.getStatistics();
        System.out.println("统计信息矩阵形状: " + stats.getRowNum() + "x" + stats.getColNum());
    }
}
```

### 1.2 基本统计分析 / Basic Statistical Analysis

```java
public class BasicStatisticalAnalysisExample {
    public static void main(String[] args) {
        // 创建带趋势和季节性的示例数据 / Create sample data with trend and seasonality
        IVector<Double> data = generateSampleTimeSeries(100);
        TimeSeriesData timeSeries = Series.createTimeSeries(data, "sales");
        
        // 使用TimeSeriesAnalyzer进行分析 / Using TimeSeriesAnalyzer for analysis
        TimeSeriesAnalyzer analyzer = Series.createTimeSeriesAnalyzer(data, "sales");
        
        // 获取数据统计信息 / Get data statistics
        Map<String, Object> stats = analyzer.getDataStatistics();
        
        System.out.println("=== 数据统计信息 / Data Statistics ===");
        System.out.println("均值: " + stats.get("mean") + " / Mean: " + stats.get("mean"));
        System.out.println("标准差: " + stats.get("std") + " / Standard Deviation: " + stats.get("std"));
        System.out.println("最小值: " + stats.get("min") + " / Minimum: " + stats.get("min"));
        System.out.println("最大值: " + stats.get("max") + " / Maximum: " + stats.get("max"));
        System.out.println("偏度: " + stats.get("skewness") + " / Skewness: " + stats.get("skewness"));
        System.out.println("峰度: " + stats.get("kurtosis") + " / Kurtosis: " + stats.get("kurtosis"));
        System.out.println("是否平稳: " + stats.get("isStationary") + " / Is Stationary: " + stats.get("isStationary"));
        
        // 趋势分析 / Trend analysis
        Map<String, Object> trend = analyzer.getTrendAnalysis();
        System.out.println("\n=== 趋势分析 / Trend Analysis ===");
        System.out.println("趋势斜率: " + trend.get("slope") + " / Trend Slope: " + trend.get("slope"));
        System.out.println("趋势截距: " + trend.get("intercept") + " / Trend Intercept: " + trend.get("intercept"));
        System.out.println("R²值: " + trend.get("rSquared") + " / R² Value: " + trend.get("rSquared"));
        System.out.println("趋势强度: " + trend.get("strength") + " / Trend Strength: " + trend.get("strength"));
        
        // 使用TimeSeriesData的增强功能 / Using enhanced TimeSeriesData features
        System.out.println("\n=== TimeSeriesData增强功能 / TimeSeriesData Enhanced Features ===");
        
        // 标准化 / Normalize
        TimeSeriesData normalized = timeSeries.normalize();
        System.out.println("标准化后均值: " + normalized.getVariable(0).mean());
        System.out.println("标准化后标准差: " + normalized.getVariable(0).std());
        
        // 合并时间序列 / Merge time series
        TimeSeriesData sineSeries = TimeSeriesData.sineWave(100, 2.0, "sine");
        TimeSeriesData merged = timeSeries.merge(sineSeries, "merged");
        System.out.println("合并后变量数: " + merged.getNumVariables());
        
        // 移动窗口操作 / Moving window operation
        TimeSeriesData windowed = timeSeries.movingWindow(10, vector -> vector.mean());
        System.out.println("窗口操作后长度: " + windowed.getLength());
    }
    
    private static IVector<Double> generateSampleTimeSeries(int length) {
        // 生成包含趋势、季节性和噪声的示例数据
        // Generate sample data with trend, seasonality, and noise
        IVector<Double> trend = Linalg.range(length).multiplyScalar(0.01);
        IVector<Double> seasonal = generateSeasonalComponent(length, 12);
        IVector<Double> noise = Linalg.randn(length).multiplyScalar(0.1);
        
        return trend.add(seasonal).add(noise);
    }
    
    private static IVector<Double> generateSeasonalComponent(int length, int period) {
        IVector<Double> seasonal = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            seasonal.set(i, Math.sin(2 * Math.PI * i / period) * 0.5);
        }
        return seasonal;
    }
}
```

### 1.3 简单预测方法 / Simple Forecasting Methods

```java
public class SimpleForecastingExample {
    public static void main(String[] args) {
        // 创建示例数据 / Create sample data
        IVector<Double> data = generateSampleTimeSeries(50);
        TimeSeriesData timeSeries = Series.createTimeSeries(data, "value");
        
        // 使用Series工厂类进行预测 / Using Series factory class for forecasting
        
        // 简单移动平均预测 / Simple moving average forecasting
        System.out.println("=== 简单移动平均预测 / Simple Moving Average Forecasting ===");
        ForecastResult smaResult = Series.simpleMovingAverage(
            timeSeries, "value", 5, 10, 0.95);
        
        System.out.println("预测值: " + smaResult.forecast);
        System.out.println("置信区间下限: " + smaResult.lowerBound);
        System.out.println("置信区间上限: " + smaResult.upperBound);
        System.out.println("均方误差: " + smaResult.mse);
        System.out.println("平均绝对误差: " + smaResult.mae);
        System.out.println("平均绝对百分比误差: " + smaResult.mape + "%");
        
        // 指数平滑预测 / Exponential smoothing forecasting
        System.out.println("\n=== 指数平滑预测 / Exponential Smoothing Forecasting ===");
        ForecastResult esResult = Series.exponentialSmoothing(
            timeSeries, "value", 0.3, 10, 0.95);
        
        System.out.println("预测值: " + esResult.forecast);
        System.out.println("均方误差: " + esResult.mse);
        System.out.println("平均绝对误差: " + esResult.mae);
        
        // 线性回归预测 / Linear regression forecasting
        System.out.println("\n=== 线性回归预测 / Linear Regression Forecasting ===");
        ForecastResult lrResult = Series.linearRegression(
            timeSeries, "value", 10, 0.95);
        
        System.out.println("预测值: " + lrResult.forecast);
        System.out.println("均方误差: " + lrResult.mse);
        System.out.println("平均绝对误差: " + lrResult.mae);
    }
    
    private static IVector<Double> generateSampleTimeSeries(int length) {
        // 生成包含趋势和噪声的示例数据
        // Generate sample data with trend and noise
        IVector<Double> trend = Linalg.range(length).multiplyScalar(0.02);
        IVector<Double> noise = Linalg.randn(length).multiplyScalar(0.1);
        
        return trend.add(noise);
    }
}
```

## 第二部分：中级应用 (Level 2 - 中级应用) / Part 2: Intermediate Level (Level 2 - Intermediate Applications)

### 2.1 时间序列分解 / Time Series Decomposition

```java
public class TimeSeriesDecompositionExample {
    public static void main(String[] args) {
        // 创建包含明显季节性的示例数据 / Create sample data with obvious seasonality
        IVector<Double> data = generateSeasonalTimeSeries(120);
        TimeSeriesData timeSeries = Series.createTimeSeries(data, "sales");
        
        // 经典分解（加法模型）/ Classical decomposition (additive model)
        System.out.println("=== 经典分解（加法模型）/ Classical Decomposition (Additive Model) ===");
        DecompositionResult additiveResult = 
            Series.classicalDecomposition(
                timeSeries, "sales", 12, 
                TimeSeriesDecomposition.DecompositionModel.ADDITIVE);
        
        System.out.println("趋势成分强度: " + additiveResult.trendStrength);
        System.out.println("季节性成分强度: " + additiveResult.seasonalStrength);
        System.out.println("残差成分强度: " + additiveResult.residualStrength);
        System.out.println("分解模型: " + additiveResult.model);
        System.out.println("季节周期: " + additiveResult.period);
        
        // 经典分解（乘法模型）/ Classical decomposition (multiplicative model)
        System.out.println("\n=== 经典分解（乘法模型）/ Classical Decomposition (Multiplicative Model) ===");
        DecompositionResult multiplicativeResult = 
            Series.classicalDecomposition(
                timeSeries, "sales", 12, 
                TimeSeriesDecomposition.DecompositionModel.MULTIPLICATIVE);
        
        System.out.println("趋势成分强度: " + multiplicativeResult.trendStrength);
        System.out.println("季节性成分强度: " + multiplicativeResult.seasonalStrength);
        System.out.println("残差成分强度: " + multiplicativeResult.residualStrength);
        
        // STL分解 / STL decomposition
        System.out.println("\n=== STL分解 / STL Decomposition ===");
        DecompositionResult stlResult = 
            Series.stlDecomposition(timeSeries, "sales", 12, 7, 21);
        
        System.out.println("趋势成分强度: " + stlResult.trendStrength);
        System.out.println("季节性成分强度: " + stlResult.seasonalStrength);
        System.out.println("残差成分强度: " + stlResult.residualStrength);
    }
    
    private static IVector<Double> generateSeasonalTimeSeries(int length) {
        // 生成包含趋势、季节性和噪声的示例数据
        // Generate sample data with trend, seasonality, and noise
        IVector<Double> trend = Linalg.range(length).multiplyScalar(0.02);
        IVector<Double> seasonal = generateSeasonalComponent(length, 12);
        IVector<Double> noise = Linalg.randn(length).multiplyScalar(0.05);
        
        return trend.add(seasonal).add(noise);
    }
    
    private static IVector<Double> generateSeasonalComponent(int length, int period) {
        IVector<Double> seasonal = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            seasonal.set(i, Math.sin(2 * Math.PI * i / period) * 0.5);
        }
        return seasonal;
    }
}
```

### 2.2 时间序列滤波 / Time Series Filtering

```java
public class TimeSeriesFilteringExample {
    public static void main(String[] args) {
        // 创建包含噪声的示例数据 / Create sample data with noise
        IVector<Double> cleanData = generateCleanSignal(100);
        IVector<Double> noisyData = cleanData.add(Linalg.randn(100).multiplyScalar(0.2));
        TimeSeriesData timeSeries = Series.createTimeSeries(noisyData, "signal");
        
        // 移动平均滤波 / Moving average filtering
        System.out.println("=== 移动平均滤波 / Moving Average Filtering ===");
        FilterResult maResult = Series.movingAverage(
            timeSeries, "signal", 5);
        
        System.out.println("滤波类型: " + maResult.filterType);
        System.out.println("信噪比: " + maResult.snr);
        System.out.println("滤波后数据长度: " + maResult.filtered.getLength());
        
        // 指数平滑滤波 / Exponential smoothing filtering
        System.out.println("\n=== 指数平滑滤波 / Exponential Smoothing Filtering ===");
        FilterResult esResult = Series.exponentialSmoothing(
            timeSeries, "signal", 0.3);
        
        System.out.println("滤波类型: " + esResult.filterType);
        System.out.println("信噪比: " + esResult.snr);
        
        // 高斯滤波 / Gaussian filtering
        System.out.println("\n=== 高斯滤波 / Gaussian Filtering ===");
        FilterResult gaussianResult = Series.gaussianFilter(
            timeSeries, "signal", 1.0);
        
        System.out.println("滤波类型: " + gaussianResult.filterType);
        System.out.println("信噪比: " + gaussianResult.snr);
        
        // 中值滤波 / Median filtering
        System.out.println("\n=== 中值滤波 / Median Filtering ===");
        FilterResult medianResult = Series.medianFilter(
            timeSeries, "signal", 5);
        
        System.out.println("滤波类型: " + medianResult.filterType);
        System.out.println("信噪比: " + medianResult.snr);
    }
    
    private static IVector<Double> generateCleanSignal(int length) {
        // 生成清洁信号 / Generate clean signal
        IVector<Double> signal = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            double t = i / 10.0;
            signal.set(i, Math.sin(2 * Math.PI * t) + 0.5 * Math.sin(4 * Math.PI * t));
        }
        return signal;
    }
}
```

### 2.3 ARIMA模型预测 / ARIMA Model Forecasting

```java
public class ARIMAForecastingExample {
    public static void main(String[] args) {
        // 创建示例数据 / Create sample data
        IVector<Double> data = generateARIMASample(100);
        TimeSeriesData timeSeries = Series.createTimeSeries(data, "value");
        
        // ARIMA(1,1,1)模型预测 / ARIMA(1,1,1) model forecasting
        System.out.println("=== ARIMA(1,1,1)模型预测 / ARIMA(1,1,1) Model Forecasting ===");
        ForecastResult arimaResult = Series.arimaForecast(
            timeSeries, "value", 1, 1, 1, 20, 0.95);
        
        System.out.println("预测值: " + arimaResult.forecast);
        System.out.println("置信区间下限: " + arimaResult.lowerBound);
        System.out.println("置信区间上限: " + arimaResult.upperBound);
        System.out.println("均方误差: " + arimaResult.mse);
        System.out.println("平均绝对误差: " + arimaResult.mae);
        System.out.println("平均绝对百分比误差: " + arimaResult.mape + "%");
        
        // 尝试不同的ARIMA参数 / Try different ARIMA parameters
        System.out.println("\n=== 不同ARIMA参数比较 / Different ARIMA Parameters Comparison ===");
        
        // ARIMA(2,1,2) / ARIMA(2,1,2)
        ForecastResult arima221 = Series.arimaForecast(
            timeSeries, "value", 2, 1, 2, 20, 0.95);
        System.out.println("ARIMA(2,1,2) MSE: " + arima221.mse);
        
        // ARIMA(0,1,1) / ARIMA(0,1,1)
        ForecastResult arima011 = Series.arimaForecast(
            timeSeries, "value", 0, 1, 1, 20, 0.95);
        System.out.println("ARIMA(0,1,1) MSE: " + arima011.mse);
        
        // 选择最佳模型 / Select best model
        if (arimaResult.mse < arima221.mse && arimaResult.mse < arima011.mse) {
            System.out.println("ARIMA(1,1,1) 是最佳模型 / ARIMA(1,1,1) is the best model");
        } else if (arima221.mse < arima011.mse) {
            System.out.println("ARIMA(2,1,2) 是最佳模型 / ARIMA(2,1,2) is the best model");
        } else {
            System.out.println("ARIMA(0,1,1) 是最佳模型 / ARIMA(0,1,1) is the best model");
        }
    }
    
    private static IVector<Double> generateARIMASample(int length) {
        // 生成ARIMA(1,1,1)过程的示例数据
        // Generate sample data from ARIMA(1,1,1) process
        IVector<Double> data = Linalg.zeros(length);
        IVector<Double> noise = Linalg.randn(length).multiplyScalar(0.1);
        
        // 初始化 / Initialize
        data.set(0, noise.get(0));
        
        // ARIMA(1,1,1)过程 / ARIMA(1,1,1) process
        for (int i = 1; i < length; i++) {
            double value = 0.7 * data.get(i-1) + noise.get(i) + 0.3 * noise.get(i-1);
            data.set(i, value);
        }
        
        return data;
    }
}
```

## 第三部分：高级应用 (Level 3 - 高级应用) / Part 3: Advanced Level (Level 3 - Advanced Applications)

### 3.1 完整分析流程 / Complete Analysis Workflow

```java
public class CompleteAnalysisWorkflowExample {
    public static void main(String[] args) {
        // 1. 创建复杂的时间序列数据 / Create complex time series data
        IVector<Double> data = generateComplexTimeSeries(200);
        TimeSeriesData timeSeries = Series.createTimeSeries(data, "complex_signal");
        
        System.out.println("=== 完整时间序列分析流程 / Complete Time Series Analysis Workflow ===");
        
        // 2. 数据预处理 / Data preprocessing
        System.out.println("\n--- 步骤1: 数据预处理 / Step 1: Data Preprocessing ---");
        TimeSeriesData preprocessed = preprocessData(timeSeries);
        System.out.println("预处理完成，数据长度: " + preprocessed.getLength() + " / Preprocessing completed, data length: " + preprocessed.getLength());
        
        // 3. 时间序列分解 / Time series decomposition
        System.out.println("\n--- 步骤2: 时间序列分解 / Step 2: Time Series Decomposition ---");
        DecompositionResult decomposition = 
            Series.classicalDecomposition(
                preprocessed, "complex_signal", 12, 
                TimeSeriesDecomposition.DecompositionModel.ADDITIVE);
        
        System.out.println("趋势成分强度: " + decomposition.trendStrength);
        System.out.println("季节性成分强度: " + decomposition.seasonalStrength);
        System.out.println("残差成分强度: " + decomposition.residualStrength);
        
        // 4. 滤波处理 / Filtering
        System.out.println("\n--- 步骤3: 滤波处理 / Step 3: Filtering ---");
        FilterResult filtering = 
            Series.movingAverage(preprocessed, "complex_signal", 3);
        
        System.out.println("滤波类型: " + filtering.filterType);
        System.out.println("信噪比: " + filtering.snr);
        
        // 5. 预测分析 / Forecasting
        System.out.println("\n--- 步骤4: 预测分析 / Step 4: Forecasting ---");
        ForecastResult forecast = 
            Series.autoForecast(filtering.filtered, "complex_signal", 30, 0.95);
        
        System.out.println("预测步数: " + forecast.forecast.length());
        System.out.println("均方误差: " + forecast.mse);
        System.out.println("平均绝对误差: " + forecast.mae);
        System.out.println("平均绝对百分比误差: " + forecast.mape + "%");
        
        // 6. 模型诊断 / Model diagnostics
        System.out.println("\n--- 步骤5: 模型诊断 / Step 5: Model Diagnostics ---");
        TimeSeriesAnalyzer analyzer = Series.createTimeSeriesAnalyzer(data, "complex_signal");
        TimeSeriesAnalyzer.AnalysisResult analysis = analyzer.quickAnalyze();
        
        if (analysis.diagnostics != null) {
            System.out.println("诊断完成 / Diagnostics completed");
        }
        
        // 7. 生成分析报告 / Generate analysis report
        System.out.println("\n--- 步骤6: 生成分析报告 / Step 6: Generate Analysis Report ---");
        String report = analyzer.exportReport("text");
        System.out.println(report);
    }
    
    private static IVector<Double> generateComplexTimeSeries(int length) {
        // 生成包含趋势、季节性、周期性和噪声的复杂时间序列
        // Generate complex time series with trend, seasonality, cyclicity, and noise
        IVector<Double> trend = Linalg.range(length).multiplyScalar(0.01);
        IVector<Double> seasonal = generateSeasonalComponent(length, 12);
        IVector<Double> cyclic = generateCyclicComponent(length, 50);
        IVector<Double> noise = Linalg.randn(length).multiplyScalar(0.1);
        
        return trend.add(seasonal).add(cyclic).add(noise);
    }
    
    private static IVector<Double> generateCyclicComponent(int length, int period) {
        IVector<Double> cyclic = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            cyclic.set(i, 0.3 * Math.sin(2 * Math.PI * i / period));
        }
        return cyclic;
    }
    
    private static TimeSeriesData preprocessData(TimeSeriesData timeSeries) {
        // 简化的数据预处理 / Simplified data preprocessing
        IVector<Double> data = timeSeries.getVariable(0);
        
        // 移除异常值 / Remove outliers
        double mean = data.mean();
        double std = data.std();
        double threshold = 3.0;
        
        List<Double> filtered = new ArrayList<>();
        for (int i = 0; i < data.length(); i++) {
            double value = data.get(i);
            if (Math.abs(value - mean) <= threshold * std) {
                filtered.add(value);
            }
        }
        
        IVector<Double> cleanData = Linalg.vector(filtered.stream().mapToDouble(Double::doubleValue).toArray());
        return TimeSeriesData.of(cleanData, timeSeries.getColumnNames()[0]);
    }
    
    private static IVector<Double> generateSeasonalComponent(int length, int period) {
        IVector<Double> seasonal = Linalg.zeros(length);
        for (int i = 0; i < length; i++) {
            seasonal.set(i, Math.sin(2 * Math.PI * i / period) * 0.5);
        }
        return seasonal;
    }
}
```

### 3.2 多变量时间序列分析 / Multivariate Time Series Analysis

```java
public class MultivariateTimeSeriesExample {
    public static void main(String[] args) {
        // 创建多变量时间序列数据 / Create multivariate time series data
        int length = 100;
        String[] columnNames = {"temperature", "humidity", "pressure"};
        IMatrix<Double> data = Linalg.randn(length, 3);
        
        // 添加相关性 / Add correlation
        for (int i = 1; i < length; i++) {
            data.set(i, 0, data.get(i-1, 0) * 0.8 + data.get(i, 0) * 0.2); // temperature
            data.set(i, 1, data.get(i-1, 1) * 0.6 + data.get(i, 1) * 0.4); // humidity
            data.set(i, 2, data.get(i-1, 2) * 0.7 + data.get(i, 2) * 0.3); // pressure
        }
        
        LocalDateTime[] timestamps = generateTimestamps(length);
        TimeSeriesData multiTimeSeries = Series.createMultivariateTimeSeries(timestamps, data.toDoubleArray(), columnNames);
        
        System.out.println("=== 多变量时间序列分析 / Multivariate Time Series Analysis ===");
        System.out.println("数据形状: " + multiTimeSeries.getLength() + " x " + multiTimeSeries.getNumVariables());
        
        // 分析每个变量 / Analyze each variable
        for (String columnName : columnNames) {
            System.out.println("\n--- 分析变量: " + columnName + " / Analyzing Variable: " + columnName + " ---");
            
            // 单变量分析 / Univariate analysis
            IVector<Double> variable = multiTimeSeries.getVariable(columnName);
            TimeSeriesAnalyzer analyzer = Series.createTimeSeriesAnalyzer(variable, columnName);
            
            // 基本统计 / Basic statistics
            Map<String, Object> stats = analyzer.getDataStatistics();
            System.out.println("均值: " + stats.get("mean"));
            System.out.println("标准差: " + stats.get("std"));
            System.out.println("是否平稳: " + stats.get("isStationary"));
            
            // 趋势分析 / Trend analysis
            Map<String, Object> trend = analyzer.getTrendAnalysis();
            System.out.println("趋势强度: " + trend.get("strength"));
            
            // 季节性分析 / Seasonal analysis
            Map<String, Object> seasonal = analyzer.getSeasonalAnalysis(12);
            System.out.println("季节性强度: " + seasonal.get("strength"));
            System.out.println("是否有季节性: " + seasonal.get("hasSeasonality"));
            
            // 预测 / Forecasting
            ForecastResult forecast = 
                Series.autoForecast(multiTimeSeries, columnName, 10, 0.95);
            System.out.println("预测MSE: " + forecast.mse);
        }
        
        // 变量间相关性分析 / Inter-variable correlation analysis
        System.out.println("\n--- 变量间相关性分析 / Inter-variable Correlation Analysis ---");
        IMatrix<Double> correlationMatrix = calculateCorrelationMatrix(multiTimeSeries);
        System.out.println("相关性矩阵: / Correlation Matrix:");
        for (int i = 0; i < columnNames.length; i++) {
            for (int j = 0; j < columnNames.length; j++) {
                System.out.printf("%.3f ", correlationMatrix.get(i, j));
            }
            System.out.println();
        }
    }
    
    private static IMatrix<Double> calculateCorrelationMatrix(TimeSeriesData timeSeries) {
        int numVars = timeSeries.getNumVariables();
        IMatrix<Double> correlation = Linalg.zeros(numVars, numVars);
        
        for (int i = 0; i < numVars; i++) {
            for (int j = 0; j < numVars; j++) {
                IVector<Double> var1 = timeSeries.getVariable(i);
                IVector<Double> var2 = timeSeries.getVariable(j);
                
                double correlationValue = calculateCorrelation(var1, var2);
                correlation.set(i, j, correlationValue);
            }
        }
        
        return correlation;
    }
    
    private static double calculateCorrelation(IVector<Double> x, IVector<Double> y) {
        double meanX = x.mean();
        double meanY = y.mean();
        
        double numerator = 0.0;
        double sumXSquared = 0.0;
        double sumYSquared = 0.0;
        
        for (int i = 0; i < x.length(); i++) {
            double dx = x.get(i) - meanX;
            double dy = y.get(i) - meanY;
            numerator += dx * dy;
            sumXSquared += dx * dx;
            sumYSquared += dy * dy;
        }
        
        double denominator = Math.sqrt(sumXSquared * sumYSquared);
        return denominator == 0 ? 0 : numerator / denominator;
    }
    
    private static LocalDateTime[] generateTimestamps(int length) {
        LocalDateTime[] timestamps = new LocalDateTime[length];
        LocalDateTime startTime = LocalDateTime.now().minusDays(length);
        for (int i = 0; i < length; i++) {
            timestamps[i] = startTime.plusDays(i);
        }
        return timestamps;
    }
}
```

### 3.3 高级滤波和分解技术 / Advanced Filtering and Decomposition Techniques

```java
public class AdvancedFilteringDecompositionExample {
    public static void main(String[] args) {
        // 创建复杂信号 / Create complex signal
        IVector<Double> data = generateComplexSignal(200);
        TimeSeriesData timeSeries = Series.createTimeSeries(data, "complex_signal");
        
        System.out.println("=== 高级滤波和分解技术 / Advanced Filtering and Decomposition Techniques ===");
        
        // 频域滤波 / Frequency domain filtering
        System.out.println("\n--- 频域滤波 / Frequency Domain Filtering ---");
        
        // 低通滤波 / Low pass filtering
        FilterResult lpResult = Series.lowPassFilter(
            timeSeries, "complex_signal", 0.1, 4);
        System.out.println("低通滤波信噪比: " + lpResult.snr);
        
        // 高通滤波 / High pass filtering
        FilterResult hpResult = Series.highPassFilter(
            timeSeries, "complex_signal", 0.01, 4);
        System.out.println("高通滤波信噪比: " + hpResult.snr);
        
        // 带通滤波 / Band pass filtering
        FilterResult bpResult = Series.bandPassFilter(
            timeSeries, "complex_signal", 0.01, 0.1, 4);
        System.out.println("带通滤波信噪比: " + bpResult.snr);
        
        // 自适应滤波 / Adaptive filtering
        FilterResult adaptiveResult = Series.adaptiveFilter(
            timeSeries, "complex_signal", 0.1);
        System.out.println("自适应滤波信噪比: " + adaptiveResult.snr);
        
        // 高级分解技术 / Advanced decomposition techniques
        System.out.println("\n--- 高级分解技术 / Advanced Decomposition Techniques ---");
        
        // X-13ARIMA-SEATS分解 / X-13ARIMA-SEATS decomposition
        DecompositionResult x13Result = 
            Series.x13Decomposition(timeSeries, "complex_signal", 12);
        System.out.println("X-13ARIMA-SEATS - 趋势强度: " + x13Result.trendStrength);
        System.out.println("X-13ARIMA-SEATS - 季节性强度: " + x13Result.seasonalStrength);
        
        // STL分解 / STL decomposition
        DecompositionResult stlResult = 
            Series.stlDecomposition(timeSeries, "complex_signal", 12, 7, 21);
        System.out.println("STL - 趋势强度: " + stlResult.trendStrength);
        System.out.println("STL - 季节性强度: " + stlResult.seasonalStrength);
        
        // 小波分解 / Wavelet decomposition
        DecompositionResult waveletResult = 
            Series.waveletDecomposition(timeSeries, "complex_signal", "db4", 4);
        System.out.println("小波分解 - 趋势强度: " + waveletResult.trendStrength);
        System.out.println("小波分解 - 季节性强度: " + waveletResult.seasonalStrength);
    }
    
    private static IVector<Double> generateComplexSignal(int length) {
        // 生成包含多个频率成分的复杂信号
        // Generate complex signal with multiple frequency components
        IVector<Double> signal = Linalg.zeros(length);
        
        for (int i = 0; i < length; i++) {
            double t = i / 10.0;
            double value = 0.0;
            
            // 低频成分 / Low frequency component
            value += Math.sin(2 * Math.PI * t / 20);
            
            // 中频成分 / Medium frequency component
            value += 0.5 * Math.sin(2 * Math.PI * t / 5);
            
            // 高频成分 / High frequency component
            value += 0.2 * Math.sin(2 * Math.PI * t / 1);
            
            // 噪声 / Noise
            value += 0.1 * (Math.random() - 0.5);
            
            signal.set(i, value);
        }
        
        return signal;
    }
}
```

## 辅助方法 / Helper Methods

```java
// 生成时间戳的辅助方法 / Helper method for generating timestamps
private static List<LocalDateTime> generateTimestamps(int length) {
    List<LocalDateTime> timestamps = new ArrayList<>();
    LocalDateTime startTime = LocalDateTime.now().minusDays(length);
    for (int i = 0; i < length; i++) {
        timestamps.add(startTime.plusDays(i));
    }
    return timestamps;
}

// 生成季节性成分的辅助方法 / Helper method for generating seasonal component
private static IVector<Double> generateSeasonalComponent(int length, int period) {
    IVector<Double> seasonal = Linalg.zeros(length);
    for (int i = 0; i < length; i++) {
        seasonal.set(i, Math.sin(2 * Math.PI * i / period) * 0.5);
    }
    return seasonal;
}
```

---

**时间序列分析示例** - 从基础到高级的完整学习路径！

**Time Series Analysis Examples** - Complete learning path from basic to advanced!