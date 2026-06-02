# 时间序列分析 (Time Series Analysis)

## 概述 / Overview

`math.timeseries` 包提供了完整的时间序列分析功能，包括数据预处理、模型选择、预测、滤波、分解和可视化等。该模块整合了项目现有的 `linalg`、`stats` 和 `signal` 包的功能，为时间序列分析提供统一、高效的解决方案。

The `math.timeseries` package provides comprehensive time series analysis functionality including data preprocessing, model selection, forecasting, filtering, decomposition, and visualization. This module integrates existing `linalg`, `stats`, and `signal` package functionality to provide a unified and efficient solution for time series analysis.

## 快速上手 / Quick Start

```java
// 创建时间序列（用 TSA.data 工厂 + Linalg）
IVector<Double> temperature = Linalg.randn(100);           // 模拟温度数据
TimeSeriesData ts = TSA.data(temperature, "temperature");
System.out.println("序列长度: " + ts.getLength());            // → 100

// 简单移动平均预测（默认 95% 置信度）
ForecastResult sma = TSA.forecast.movingAverage(temperature, 5, 10);
System.out.println("MA 预测值: " + sma.forecast.get(0));  // 未来第1步预测

// 指数平滑预测 / Exponential smoothing forecast
ForecastResult es = TSA.forecast.expSmooth(temperature, 0.3, 10);

// 线性回归趋势 / Linear regression trend
ForecastResult lr = TSA.forecast.linearTrend(temperature, 10);

// 序列分解（趋势 + 季节 + 残差）
var decomp = TSA.decompose.stl(temperature, 12, 7, 21);
IVector<Double> trend = decomp.trend;      // 趋势成分
IVector<Double> seasonal = decomp.seasonal; // 季节成分

// 时间序列滤波 / Filter noise
FilterResult filtered = TSA.filter.lowPass(temperature, 0.5, 4);  // 截止频率0.5Hz，阶数4
```

## 核心类 / Core Classes

### TSA 类 / TSA Class

统一时间序列分析入口类，通过 `TSA.data()` 创建数据，通过 `TSA.forecast`/`TSA.filter`/`TSA.decompose`/`TSA.cointegrate`/`TSA.plot` 子域调用分析功能。

Unified time series analysis entry point: `TSA.data()` for data creation, subdomain groups for analysis.

### TimeSeriesData 类 / TimeSeriesData Class

时间序列数据容器类，支持单变量和多变量时间序列，提供基本的数据操作和访问方法。

Time series data container class supporting univariate and multivariate time series with basic data operations and access methods.

#### TimeSeriesData 构造方法 / TimeSeriesData Constructor Methods

TimeSeriesData 提供了多种创建时间序列数据的方式：

TimeSeriesData provides multiple ways to create time series data:

1. `of(IVector<Double> data, String name)` - 通过数据向量和名称创建单变量时间序列
   Create univariate time series from data vector and name

2. `of(IVector<Double> data, String name, double frequency)` - 通过数据向量、名称和频率创建单变量时间序列
   Create univariate time series from data vector, name and frequency

3. `of(IVector<Double> data, double samplingRate, String name, LocalDateTime startTime)` - 通过数据向量、采样率、名称和开始时间创建单变量时间序列
   Create univariate time series from data vector, sampling rate, name and start time

4. `of(LocalDateTime[] timestamps, double[] values, String name)` - 通过时间戳数组、值数组和名称创建单变量时间序列
   Create univariate time series from timestamp array, value array and name

5. `of(LocalDateTime[] timestamps, double[][] data, String[] variableNames)` - 通过时间戳数组、数据矩阵和变量名数组创建多变量时间序列
   Create multivariate time series from timestamp array, data matrix and variable name array

#### TimeSeriesData 静态工厂方法 / TimeSeriesData Static Factory Methods

TimeSeriesData 还提供了一些便捷的静态工厂方法：

TimeSeriesData also provides some convenient static factory methods:

1. `sample(int length, String name)` - 创建指定长度的示例时间序列
   Create sample time series with specified length

2. `sineWave(int length, double frequency, String name)` - 创建正弦波时间序列
   Create sine wave time series

3. `of(IVector<Double> data, double samplingRate, String name, LocalDateTime startTime)` - 通过数据向量、采样率、名称和开始时间创建
   Create from data vector, sampling rate, name and start time

#### TimeSeriesData 静态工厂方法汇总 / TimeSeriesData Static Factory Summary

创建时间序列推荐使用 `TSA.data()` 或 `TimeSeriesData.of()` 静态工厂方法：

Prefer `TSA.data()` or `TimeSeriesData.of()` for creating time series:

```
// 从 IVector 创建
TimeSeriesData ts1 = TSA.data(values, "temperature");

// 从 double[] 创建
TimeSeriesData ts2 = TSA.data(new double[]{1.0, 2.0, 3.0}, "price");

// 带时间戳
TimeSeriesData ts3 = TSA.data(values, "price", timestamps);

// 直接使用 TimeSeriesData.of()
TimeSeriesData ts4 = TimeSeriesData.of(values, 1.0, "temperature", LocalDateTime.now());
```

### TimeSeriesAnalyzer 类 / TimeSeriesAnalyzer Class

统一时间序列分析工具类，提供完整的时间序列分析功能，包括数据预处理、模型选择、预测、诊断等。

Unified time series analysis tool class providing comprehensive time series analysis functionality including data preprocessing, model selection, forecasting, and diagnostics.

### TimeSeriesForecasting 类 / TimeSeriesForecasting Class

时间序列预测类，提供多种预测方法，包括ARIMA模型、指数平滑、线性回归预测等。

Time series forecasting class providing various forecasting methods including ARIMA models, exponential smoothing, and linear regression forecasting.

### TimeSeriesFiltering 类 / TimeSeriesFiltering Class

时间序列滤波类，提供多种滤波方法，包括移动平均、指数平滑、卡尔曼滤波、小波滤波等。

Time series filtering class providing various filtering methods including moving average, exponential smoothing, Kalman filtering, and wavelet filtering.

### TimeSeriesDecomposition 类 / TimeSeriesDecomposition Class

时间序列分解类，提供时间序列分解功能，包括趋势、季节性、周期性成分的分离。

Time series decomposition class providing time series decomposition functionality including trend, seasonal, and cyclical component separation.

## 主要功能 / Main Features

### 1. 时间序列数据管理 / Time Series Data Management

#### 1.1 创建时间序列数据 / Creating Time Series Data

```
import com.yishape.lab.math.timeseries.TSA;
import com.yishape.lab.math.timeseries.TimeSeriesData;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

// 使用 TSA.data 工厂创建（类似 Linalg.vector 模式）
IVector<Double> values = Linalg.randn(100);
TimeSeriesData timeSeries1 = TSA.data(values, "temperature");

// 使用 TimeSeriesData 静态工厂方法
TimeSeriesData timeSeries2 = TimeSeriesData.of(values, "temperature2");
TimeSeriesData timeSeries3 = TimeSeriesData.of(values, 1.0, "temperature3", LocalDateTime.now());

// 使用 of() 创建（替代原 builder 模式）
TimeSeriesData timeSeries4 = TimeSeriesData.of(values, "temperature4");

// 创建多变量时间序列 / Create multivariate time series
String[] columnNames = {"temperature", "humidity", "pressure"};
double[][] multiData = Linalg.randn(100, 3).toDoubleArray();
LocalDateTime[] timestamps = generateTimestamps(100);
TimeSeriesData multiTimeSeries = TimeSeriesData.of(timestamps, multiData, columnNames);

// 创建示例时间序列 / Create sample time series
TimeSeriesData sampleSeries = TimeSeriesData.sample(100, "sample");
TimeSeriesData sineWave = TimeSeriesData.sineWave(100, 2.0, "sine");
```

#### 1.2 数据访问和操作 / Data Access and Operations

```
// 获取基本属性 / Get basic properties
int length = timeSeries1.getLength();
int numVariables = timeSeries1.getNumVariables();
double samplingRate = timeSeries1.getSamplingRate();
boolean isUnivariate = timeSeries1.isUnivariate();
boolean isMultivariate = timeSeries1.isMultivariate();

// 获取变量数据 / Get variable data
IVector<Double> temperature = timeSeries1.getVariable(0);
IVector<Double> humidity = timeSeries1.getVariable("humidity");

// 获取变量索引 / Get variable index
int tempIndex = timeSeries1.getVariableIndex("temperature");

// 获取时间范围 / Get time range
LocalDateTime[] timeRange = timeSeries1.getTimeRange();
double timeInterval = timeSeries1.getTimeInterval();

// 切片操作 / Slicing operations
TimeSeriesData sliced = timeSeries1.slice(10, 50);
TimeSeriesData timeRangeSliced = timeSeries1.slice(
    LocalDateTime.now().minusDays(10), 
    LocalDateTime.now().minusDays(5)
);

// 重采样 / Resampling
TimeSeriesData resampled = timeSeries1.resample(0.5); // 降低采样率到0.5Hz

// 添加噪声 / Add noise
TimeSeriesData noisy = timeSeries1.addNoise(0.1);

// 转换为单变量时间序列 / Convert to univariate time series
TimeSeriesData univariate = timeSeries1.toUnivariate("temperature");

// 合并时间序列 / Merge time series
TimeSeriesData merged = timeSeries1.merge(timeSeries2, "merged_series");

// 标准化 / Normalize
TimeSeriesData normalized = timeSeries1.normalize();

// 移动窗口操作 / Moving window operation
TimeSeriesData windowed = timeSeries1.movingWindow(10, vector -> vector.mean());

// 获取统计信息 / Get statistics
IMatrix<Double> stats = timeSeries1.getStatistics();

// 转换为数组格式 / Convert to array format
Tuple2<double[], double[][]> arrays = timeSeries1.toDoubleArrays();
```

#### 1.3 多变量时间序列创建 / Creating Multivariate Time Series

```
// 创建多变量时间序列 / Create multivariate time series
String[] varNames = {"var1", "var2", "var3"};
IMatrix<Double> multiData = Linalg.randn(1000, 3);
LocalDateTime[] timestamps = generateTimestamps(1000);
TimeSeriesData multiTs = new TimeSeriesData(
    Arrays.asList(timestamps), multiData.getColumn(0), varNames[0]);
// 或使用构造函数: new TimeSeriesData(timestampsList, dataMatrix, varNames)

// 简单单变量创建 / Simple univariate creation
TimeSeriesData simpleSeries = TSA.data(new double[]{1.0, 2.0, 3.0, 4.0, 5.0}, "simple_series");
```

### 2. 时间序列分析 / Time Series Analysis

#### 2.1 统一分析器 / Unified Analyzer

```
import com.yishape.lab.math.timeseries.TimeSeriesAnalyzer;

// 创建分析器 / Create analyzer
TimeSeriesAnalyzer analyzer = new TimeSeriesAnalyzer(values, "temperature");

// 快速分析 / Quick analysis
TimeSeriesAnalyzer.AnalysisResult result = analyzer.quickAnalyze();

// 自定义分析配置 / Custom analysis configuration
TimeSeriesAnalyzer.AnalysisConfig config = new TimeSeriesAnalyzer.AnalysisConfig.Builder()
    .setForecastSteps(20)
    .setConfidenceLevel(0.95)
    .setPerformDiagnostics(true)
    .setAutoPreprocess(true)
    .build();

TimeSeriesAnalyzer.AnalysisResult customResult = analyzer.analyze(config);

// 获取分析结果 / Get analysis results
ITimeSeriesModel bestModel = result.bestModel;
ITimeSeriesForecastResult forecast = result.forecast;
ITimeSeriesDiagnostics diagnostics = result.diagnostics;
String summary = result.summary;
```

#### 2.2 数据统计和特征分析 / Data Statistics and Feature Analysis

```
// 获取数据统计信息 / Get data statistics
Map<String, Object> stats = analyzer.getDataStatistics();
System.out.println("数据长度: " + stats.get("length"));
System.out.println("均值: " + stats.get("mean"));
System.out.println("标准差: " + stats.get("std"));
System.out.println("是否平稳: " + stats.get("isStationary"));

// 趋势分析 / Trend analysis
Map<String, Object> trend = analyzer.getTrendAnalysis();
System.out.println("趋势斜率: " + trend.get("slope"));
System.out.println("趋势强度: " + trend.get("strength"));
System.out.println("R²值: " + trend.get("rSquared"));

// 季节性分析 / Seasonal analysis
Map<String, Object> seasonal = analyzer.getSeasonalAnalysis(12); // 12个月周期
System.out.println("季节性强度: " + seasonal.get("strength"));
System.out.println("是否有季节性: " + seasonal.get("hasSeasonality"));
```

### 3. 时间序列预测 / Time Series Forecasting

#### 算法原理 / Algorithm Principles

时间序列预测的核心思想是：**利用数据的时间依赖性，用历史数据推断未来**。

**三种时间序列成分**：

- **趋势（T）**：长期方向（上升、下降、平稳）
- **季节性（S）**：固定周期的波动（如每年夏天销量上升）
- **残差（R）**：无法解释的随机波动

一个加法模型：`Y(t) = T(t) + S(t) + R(t)` 或乘法模型 `Y(t) = T(t) × S(t) × R(t)`。

**ARIMA 的直观理解**：

- **AR（自回归）**：用过去 p 个值预测当前值
- **I（差分）**：让不平稳序列变平稳
- **MA（滑动平均）**：用过去 q 个预测误差调整

Holt-Winters 是 ARIMA 的季节性扩展，加入了：
- 平滑水平 α、平滑趋势 β、平滑季节 γ（均在 0~1 之间，越接近 0 越重视历史）

**选型建议**：

| 数据特征 | 推荐方法 |
|---------|---------|
| 无趋势、无季节（随机波动）| 简单均值或白噪声 |
| 有趋势、无季节 | Holt 线性指数平滑 |
| 有趋势、有季节 | Holt-Winters |
| 有自相关（AR 性质明显）| ARIMA |
| 数据量小、特征复杂 | 机器学习（ML 回归）|

#### 3.1 简单预测方法 / Simple Forecasting Methods

```
import com.yishape.lab.math.timeseries.TSA;

// 简单移动平均预测 / Simple moving average forecasting
TimeSeriesForecasting.ForecastResult smaResult = TSA.forecast.movingAverage(
    timeSeries.getVariable("temperature"), 5, 10);

// 指数平滑预测 / Exponential smoothing forecasting
ForecastResult esResult = TSA.forecast.expSmooth(
    timeSeries.getVariable("temperature"), 0.3, 10);

// 线性回归预测 / Linear regression forecasting
ForecastResult lrResult = TSA.forecast.linearTrend(
    timeSeries.getVariable("temperature"), 10);
```

#### 3.2 高级预测方法 / Advanced Forecasting Methods

```
// ARIMA模型预测 / ARIMA model forecasting
TimeSeriesForecasting.ForecastResult arimaResult = TSA.forecast.arima(
    timeSeries.getVariable("temperature"), 1, 1, 1, 10);

// 季节性预测 / Seasonal forecasting
TimeSeriesForecasting.ForecastResult seasonalResult = TSA.forecast.seasonal(
    timeSeries.getVariable("temperature"), 12, 10);

// Holt-Winters预测 / Holt-Winters forecasting
TimeSeriesForecasting.ForecastResult hwResult = TSA.forecast.holtWinters(
    timeSeries.getVariable(0), 0.3, 0.1, 0.1, 12, 10);  // 参数: data, alpha, beta, gamma, period, steps

// GARCH预测 / GARCH forecasting
TimeSeriesForecasting.ForecastResult garchResult = TSA.forecast.garch(
    timeSeries.getVariable("temperature"), 1, 1, 10);

// 状态空间模型预测 / State space model forecasting
TimeSeriesForecasting.ForecastResult ssResult = TSA.forecast.stateSpace(
    timeSeries.getVariable("temperature"), 0.1, 0.05, 0.2, 10);

// 自动模型选择预测 / Automatic model selection forecasting
TimeSeriesForecasting.ForecastResult autoResult = TSA.forecast.auto(
    timeSeries.getVariable("temperature"), 10);
```

#### 3.3 预测结果分析 / Forecast Result Analysis

```
// 获取预测结果 / Get forecast results
IVector<Double> forecast = result.forecast;
IVector<Double> lowerBound = result.lowerBound;
IVector<Double> upperBound = result.upperBound;

// 获取误差指标 / Get error metrics
double mse = result.mse;
double mae = result.mae;
double mape = result.mape;

// 获取模型信息 / Get model information
String modelType = result.modelType;
double confidenceLevel = result.confidenceLevel;

System.out.println("预测值: " + forecast);
System.out.println("置信区间: [" + lowerBound + ", " + upperBound + "]");
System.out.println("均方误差: " + mse);
System.out.println("平均绝对误差: " + mae);
System.out.println("平均绝对百分比误差: " + mape + "%");
```

### 4. 时间序列滤波 / Time Series Filtering

#### 4.1 基础滤波方法 / Basic Filtering Methods

```
import com.yishape.lab.math.timeseries.TSA;

// 移动平均滤波 / Moving average filtering
FilterResult maResult = TSA.filter.movingAverage(
    timeSeries.getVariable("temperature"), 5);

// 指数平滑滤波 / Exponential smoothing filtering
FilterResult esResult = TSA.filter.expSmooth(
    timeSeries.getVariable("temperature"), 0.3);

// 高斯滤波 / Gaussian filtering
TimeSeriesFiltering.FilterResult gaussianResult = TSA.filter.gaussian(
    timeSeries.getVariable("temperature"), 1.0);

// 中值滤波 / Median filtering
TimeSeriesFiltering.FilterResult medianResult = TSA.filter.median(
    timeSeries.getVariable("temperature"), 5);
```

#### 4.2 频域滤波方法 / Frequency Domain Filtering Methods

```
// 低通滤波 / Low pass filtering
TimeSeriesFiltering.FilterResult lpResult = TSA.filter.lowPass(
    timeSeries.getVariable("temperature"), 0.1, 4);

// 高通滤波 / High pass filtering
TimeSeriesFiltering.FilterResult hpResult = TSA.filter.highPass(
    timeSeries.getVariable("temperature"), 0.01, 4);

// 带通滤波 / Band pass filtering
TimeSeriesFiltering.FilterResult bpResult = TSA.filter.bandPass(
    timeSeries.getVariable("temperature"), 0.01, 0.1, 4);

// 自适应滤波 / Adaptive filtering
TimeSeriesFiltering.FilterResult adaptiveResult = TSA.filter.adaptive(
    timeSeries.getVariable("temperature"), 0.1);
```

#### 4.3 滤波结果分析 / Filter Result Analysis

```
// 获取滤波结果 / Get filtering results
IVector<Double> filtered = result.filtered;
IVector<Double> noise = result.noise;
double snr = result.snr;
String filterType = result.filterType;

System.out.println("滤波类型: " + filterType);
System.out.println("信噪比: " + snr);
System.out.println("滤波后数据长度: " + filtered.length());
System.out.println("噪声数据长度: " + noise.length());
```

### 5. 时间序列分解 / Time Series Decomposition

#### 5.1 经典分解方法 / Classical Decomposition Methods

```
import com.yishape.lab.math.timeseries.TSA;

// 经典分解（加法模型）/ Classical decomposition (additive model)
TimeSeriesDecomposition.DecompositionResult classicResult = 
    TSA.decompose.classical(
        timeSeries.getVariable("temperature"), 12, 
        TimeSeriesDecomposition.DecompositionModel.ADDITIVE);

// 经典分解（乘法模型）/ Classical decomposition (multiplicative model)
TimeSeriesDecomposition.DecompositionResult multiplicativeResult = 
    TSA.decompose.classical(
        timeSeries.getVariable("temperature"), 12, 
        TimeSeriesDecomposition.DecompositionModel.MULTIPLICATIVE);
```

#### 5.2 高级分解方法 / Advanced Decomposition Methods

```
// X-13ARIMA-SEATS分解 / X-13ARIMA-SEATS decomposition
TimeSeriesDecomposition.DecompositionResult x13Result = 
    TSA.decompose.x13(timeSeries.getVariable("temperature"), 12);

// STL分解 / STL decomposition
TimeSeriesDecomposition.DecompositionResult stlResult = 
    TSA.decompose.stl(timeSeries.getVariable("temperature"), 12, 7, 21);

// 小波分解 / Wavelet decomposition
TimeSeriesDecomposition.DecompositionResult waveletResult = 
    TSA.decompose.wavelet(timeSeries.getVariable("temperature"), "db4", 4);
```

#### 5.3 分解结果分析 / Decomposition Result Analysis

```
// 获取分解结果 / Get decomposition results
IVector<Double> trend = result.trend;
IVector<Double> seasonal = result.seasonal;
IVector<Double> residual = result.residual;
IVector<Double> original = result.original;

// 获取成分强度 / Get component strengths
double trendStrength = result.trendStrength;
double seasonalStrength = result.seasonalStrength;
double residualStrength = result.residualStrength;

System.out.println("趋势成分强度: " + trendStrength);
System.out.println("季节性成分强度: " + seasonalStrength);
System.out.println("残差成分强度: " + residualStrength);
System.out.println("分解模型: " + result.model);
System.out.println("季节周期: " + result.period);
```

### 6. Levinson-Durbin 求解器 / Levinson-Durbin Solver

#### 算法原理 / Algorithm Principles

Levinson-Durbin 递归是一种 O(n²) 算法，用于求解 Toeplitz 线性系统和 Yule-Walker 方程。它利用 Toeplitz 矩阵的位移不变性，递归地构建解向量和自回归系数。

The Levinson-Durbin recursion is an O(n²) algorithm for solving Toeplitz linear systems and Yule-Walker equations. It exploits the shift-invariance of Toeplitz matrices to recursively build solution vectors and autoregressive coefficients.

#### 6.1 Yule-Walker 方程 / Yule-Walker Equations

```java
import com.yishape.lab.math.timeseries.solver.LevinsonDurbinSolver;

// 从自相关函数估计 AR(p) 系数 / Estimate AR(p) coefficients from autocorrelation
double[] r = {1.0, 0.8, 0.5, 0.3};  // r[0], r[1], ..., r[p]
int p = 3;                            // AR 阶数

// 快速求解 (仅返回 AR 系数 + 噪声方差)
IVector<Double> arCoeffs = LevinsonDurbinSolver.solveYuleWalker(r, p);

// 详细求解 (含反射系数、AIC 准则)
LevinsonDurbinSolver.YuleWalkerResult result =
    LevinsonDurbinSolver.solveYuleWalkerDetailed(r, p);

double[] ar = result.arCoefficients().toDoubleArray();          // [a1, ..., ap]
double noiseVar = result.noiseVariance();                       // 噪声方差
double[] refl = result.reflectionCoefficients().toDoubleArray(); // 反射系数
double[] aic = result.aic();                                    // 各阶 AIC
```

**反射系数的含义**: 第 k 个反射系数等于 PACF(k)，用于判断 AR 模型阶数（PACF 截尾性质）。

**AIC 准则**: AIC(k) = log(E_k) + 2k/(N-1)，最小值对应的 k 为最优 AR 阶数。

#### 6.2 通用 Toeplitz 系统求解 / General Toeplitz System Solver

```java
// 对称 Toeplitz 系统 T x = b / Symmetric Toeplitz system T x = b
double[] column = {5, 2, 1, 0.5};  // T[i][j] = column[|i-j|]
double[] b = {10, 8, 6, 4};
IVector<Double> x = LevinsonDurbinSolver.solve(column, b);

// 使用 IMatrix 接口 (自动提取第一列) / Using IMatrix interface
IMatrix<Double> T = Linalg.toeplitz(column);
IVector<Double> x2 = LevinsonDurbinSolver.solve(T, Linalg.vector(b));
```

Durbin 递归复杂度 O(n²)，远优于稠密 LU 的 O(n³)。

The Durbin recursion achieves O(n²) complexity, significantly faster than dense LU at O(n³).

#### 6.3 PACF 计算集成 / PACF Computation Integration

`TimeSeriesUtils.calculatePartialAutocorrelation()` 内部使用 Levinson-Durbin 算法计算偏自相关函数：

```java
IVector<Double> pacf = TimeSeriesUtils.calculatePartialAutocorrelation(data, maxLag);
// pacf[k] = 第 k 阶反射系数 (k = 0 固定为 1.0)
```

### 7. 时间序列可视化 / Time Series Visualization

#### 6.1 基础图表 / Basic Charts

```
import com.yishape.lab.math.timeseries.TSA;

// 时间序列线图 / Time series line chart
IPlot plot1 = TSA.plot.plotTimeSeries(timeSeries, "Temperature Over Time");

// 多变量时间序列图 / Multivariate time series plot
IPlot plot2 = TSA.plot.plotMultivariateTimeSeries(multiTimeSeries, "Environmental Data");

// 预测结果图 / Forecast result plot
IPlot plot3 = TSA.plot.plotForecasting(timeSeries, 20, "Temperature Forecast");

// 分解结果图 / Decomposition result plot
IPlot plot4 = TSA.plot.plotSeasonalDecomposition(timeSeries, 12, "Temperature Decomposition");
```

#### 6.2 高级图表 / Advanced Charts

```
// 自相关函数图 / Autocorrelation function plot
IPlot plot5 = TSA.plot.plotAutocorrelation(timeSeries, 20, "ACF Plot");

// 偏自相关函数图 / Partial autocorrelation function plot
IPlot plot6 = TSA.plot.plotPartialAutocorrelation(timeSeries, 20, "PACF Plot");

// 功率谱密度图 / Power spectral density plot
IPlot plot7 = com.yishape.lab.math.signal.Signals.plotPowerSpectralDensity(
    timeSeries.getVariable(0), timeSeries.getSamplingRate(), 256, 0.5, "PSD Plot");

// 小波变换图 / Wavelet transform plot
IPlot plot8 = com.yishape.lab.math.signal.WaveletPlots.plotWaveletCoefficients(
    timeSeries.getVariable(0), "db4", 4, "Wavelet Transform");

// 滤波结果对比图 / Filter result comparison plot
IPlot plot9 = TimeSeriesPlots.plotTimeSeriesComparison(
    Arrays.asList(original, filtered), 
    Arrays.asList("Original", "Filtered"), 
    "Filter Comparison");
```

## 使用示例 / Usage Examples

### 完整分析流程 / Complete Analysis Workflow

```
public class CompleteTimeSeriesAnalysis {
    public static void main(String[] args) {
        // 1. 创建时间序列数据 / Create time series data
        IVector<Double> data = generateSampleData();
        TimeSeriesData timeSeries = TSA.data(data, "temperature");
        
        // 2. 数据预处理 / Data preprocessing
        TimeSeriesData preprocessed = preprocessData(timeSeries);
        
        // 3. 时间序列分解 / Time series decomposition
        IVector<Double> y = preprocessed.getVariable("temperature");
        DecompositionResult decomposition = 
            TSA.decompose.classical(y, 12, 
                TimeSeriesDecomposition.DecompositionModel.ADDITIVE);
        
        // 4. 滤波处理 / Filtering
        FilterResult filtering = TSA.filter.movingAverage(y, 3);
        
        // 5. 预测分析 / Forecasting
        ForecastResult forecast = TSA.forecast.auto(y, 20);
        
        // 6. 结果可视化 / Result visualization
        List<IPlot> dashboard = TSA.plot.createTimeSeriesDashboard(timeSeries, "Complete Analysis");
        
        // 7. 生成分析报告 / Generate analysis report
        String report = generateAnalysisReport(decomposition, filtering, forecast);
        System.out.println(report);
    }
    
    private static IVector<Double> generateSampleData() {
        // 生成包含趋势、季节性和噪声的示例数据
        // Generate sample data with trend, seasonality, and noise
        return Linalg.randn(100).multiplyByScalar(0.1)
            .add(Linalg.range(100).multiplyByScalar(0.01))
            .add(generateSeasonalComponent(100, 12));
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

```
public class MultivariateTimeSeriesAnalysis {
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
        TimeSeriesData multiTimeSeries = TimeSeriesData.of(timestamps, data.toDoubleArray(), columnNames);
        
        System.out.println("=== 多变量时间序列分析 / Multivariate Time Series Analysis ===");
        System.out.println("数据形状: " + multiTimeSeries.getLength() + " x " + multiTimeSeries.getNumVariables());
        
        // 分析每个变量 / Analyze each variable
        for (String columnName : columnNames) {
            System.out.println("\n--- 分析变量: " + columnName + " / Analyzing Variable: " + columnName + " ---");
            
            // 单变量分析 / Univariate analysis
            IVector<Double> variable = multiTimeSeries.getVariable(columnName);
            TimeSeriesAnalyzer analyzer = new TimeSeriesAnalyzer(variable, columnName);
            
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
                TSA.forecast.auto(multiTimeSeries.getVariable(columnName), 10);
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

## 性能特性 / Performance Features

### 计算效率 / Computational Efficiency

- 高效的向量化操作 / Efficient vectorized operations
- 优化的算法实现 / Optimized algorithm implementations
- 内存友好的数据结构 / Memory-friendly data structures
- 并行计算支持 / Parallel computing support

### 数值稳定性 / Numerical Stability

- 稳定的数值算法 / Stable numerical algorithms
- 边界情况处理 / Boundary case handling
- 精度控制 / Precision control
- 异常处理 / Exception handling

### 扩展性 / Extensibility

- 模块化设计 / Modular design
- 接口抽象 / Interface abstraction
- 插件式架构 / Plugin architecture
- 自定义模型支持 / Custom model support

## 注意事项 / Notes

1. **数据质量** / **Data Quality**: 确保输入数据的质量和完整性 / Ensure input data quality and completeness
2. **参数选择** / **Parameter Selection**: 根据数据特性选择合适的参数 / Choose appropriate parameters based on data characteristics
3. **模型验证** / **Model Validation**: 使用交叉验证等方法验证模型性能 / Use cross-validation and other methods to validate model performance
4. **内存管理** / **Memory Management**: 对于大规模数据，注意内存使用 / For large-scale data, pay attention to memory usage
5. **异常处理** / **Exception Handling**: 适当处理可能的异常情况 / Properly handle possible exception cases

## 与Python库功能对照表 / Python Library Functionality Comparison Table

| 功能类别 / Function Category | yishape-math | Python库 / Python Libraries | 说明 / Description |
|---------|-------------|-------------------|------|
| **数据管理 / Data Management** | | | |
| 时间序列数据类 / Time series data class | `TimeSeriesData` | `pandas.Series`, `pandas.DataFrame` | 时间序列数据容器 / Time series data container |
| 数据切片 / Data slicing | `slice()`, `slice(LocalDateTime, LocalDateTime)` | `pandas.Series.loc[]` | 时间序列切片 / Time series slicing |
| 重采样 / Resampling | `resample()` | `pandas.Series.resample()` | 时间序列重采样 / Time series resampling |
| **预测方法 / Forecasting Methods** | | | |
| 简单移动平均 / Simple moving average | `simpleMovingAverage()` | `pandas.Series.rolling().mean()` | 简单移动平均预测 / Simple moving average forecasting |
| 指数平滑 / Exponential smoothing | `exponentialSmoothing()` | `statsmodels.tsa.holtwinters.ExponentialSmoothing` | 指数平滑预测 / Exponential smoothing forecasting |
| ARIMA模型 / ARIMA model | `arimaForecast()` | `statsmodels.tsa.arima.model.ARIMA` | ARIMA模型预测 / ARIMA model forecasting |
| 季节性预测 / Seasonal forecasting | `seasonalForecast()` | `statsmodels.tsa.seasonal.seasonal_decompose` | 季节性预测 / Seasonal forecasting |
| Holt-Winters / Holt-Winters | `holtWintersForecast()` | `statsmodels.tsa.holtwinters.HoltWinters` | Holt-Winters预测 / Holt-Winters forecasting |
| GARCH模型 / GARCH model | `garchForecast()` | `arch.arch_model` | GARCH模型预测 / GARCH model forecasting |
| 状态空间模型 / State space model | `stateSpaceForecast()` | `statsmodels.tsa.statespace` | 状态空间模型预测 / State space model forecasting |
| **滤波方法 / Filtering Methods** | | | |
| 移动平均滤波 / Moving average filter | `movingAverage()` | `scipy.signal.savgol_filter` | 移动平均滤波 / Moving average filtering |
| 指数平滑滤波 / Exponential smoothing filter | `exponentialSmoothing()` | `pandas.Series.ewm()` | 指数平滑滤波 / Exponential smoothing filtering |
| 高斯滤波 / Gaussian filter | `gaussianFilter()` | `scipy.ndimage.gaussian_filter1d` | 高斯滤波 / Gaussian filtering |
| 中值滤波 / Median filter | `medianFilter()` | `scipy.signal.medfilt` | 中值滤波 / Median filtering |
| 低通滤波 / Low pass filter | `lowPassFilter()` | `scipy.signal.butter` | 低通滤波 / Low pass filtering |
| 高通滤波 / High pass filter | `highPassFilter()` | `scipy.signal.butter` | 高通滤波 / High pass filtering |
| 带通滤波 / Band pass filter | `bandPassFilter()` | `scipy.signal.butter` | 带通滤波 / Band pass filtering |
| 自适应滤波 / Adaptive filter | `adaptiveFilter()` | `scipy.signal.lfilter` | 自适应滤波 / Adaptive filtering |
| **分解方法 / Decomposition Methods** | | | |
| 经典分解 / Classical decomposition | `classicalDecomposition()` | `statsmodels.tsa.seasonal.seasonal_decompose` | 经典时间序列分解 / Classical time series decomposition |
| X-13ARIMA-SEATS / X-13ARIMA-SEATS | `x13Decomposition()` | `seasonal.x13` | X-13ARIMA-SEATS分解 / X-13ARIMA-SEATS decomposition |
| STL分解 / STL decomposition | `stlDecomposition()` | `statsmodels.tsa.seasonal.STL` | STL分解 / STL decomposition |
| 小波分解 / Wavelet decomposition | `waveletDecomposition()` | `pywt.wavedec` | 小波分解 / Wavelet decomposition |
| **Levinson-Durbin / Yule-Walker** | | | |
| Yule-Walker 方程 | `LevinsonDurbinSolver.solveYuleWalker()` | `statsmodels.regression.linear_model.yule_walker` | AR 系数估计 / AR coefficient estimation |
| Toeplitz 系统求解 | `LevinsonDurbinSolver.solve()` | `scipy.linalg.solve_toeplitz()` | O(n²) Toeplitz 求解 / O(n²) Toeplitz solver |
| 偏自相关 (PACF) | `TimeSeriesUtils.calculatePartialAutocorrelation()` | `statsmodels.tsa.stattools.pacf()` | Levinson-Durbin 算法 / Levinson-Durbin algorithm |
| **可视化 / Visualization** | | | |
| 时间序列图 / Time series plot | `plotTimeSeries()` | `matplotlib.pyplot.plot` | 时间序列线图 / Time series line chart |
| 预测图 / Forecast plot | `plotForecast()` | `matplotlib.pyplot.plot` | 预测结果图 / Forecast result plot |
| 分解图 / Decomposition plot | `plotDecomposition()` | `statsmodels.graphics.tsaplots.plot_decomposed` | 分解结果图 / Decomposition result plot |
| ACF图 / ACF plot | `plotACF()` | `statsmodels.graphics.tsaplots.plot_acf` | 自相关函数图 / Autocorrelation function plot |
| PACF图 / PACF plot | `plotPACF()` | `statsmodels.graphics.tsaplots.plot_pacf` | 偏自相关函数图 / Partial autocorrelation function plot |
| 功率谱密度图 / PSD plot | `plotPSD()` | `matplotlib.pyplot.psd` | 功率谱密度图 / Power spectral density plot |
| 小波图 / Wavelet plot | `plotWavelet()` | `pywt.cwt` | 小波变换图 / Wavelet transform plot |

## 最佳实践建议 / Best Practices Recommendations

### 数据预处理 / Data Preprocessing

1. **缺失值处理** / **Missing Value Handling**: 检查并处理缺失值 / Check and handle missing values
2. **异常值检测** / **Outlier Detection**: 识别并处理异常值 / Identify and handle outliers
3. **平稳性检验** / **Stationarity Testing**: 确保数据平稳性 / Ensure data stationarity
4. **季节性检验** / **Seasonality Testing**: 检测季节性模式 / Detect seasonal patterns

### 模型选择 / Model Selection

1. **数据特征分析** / **Data Feature Analysis**: 分析数据的基本特征 / Analyze basic data characteristics
2. **模型比较** / **Model Comparison**: 比较不同模型的性能 / Compare performance of different models
3. **交叉验证** / **Cross Validation**: 使用交叉验证评估模型 / Use cross-validation to evaluate models
4. **参数调优** / **Parameter Tuning**: 优化模型参数 / Optimize model parameters

### 结果解释 / Result Interpretation

1. **统计显著性** / **Statistical Significance**: 检查结果的统计显著性 / Check statistical significance of results
2. **置信区间** / **Confidence Intervals**: 理解置信区间的含义 / Understand meaning of confidence intervals
3. **误差分析** / **Error Analysis**: 分析预测误差的来源 / Analyze sources of prediction errors
4. **模型诊断** / **Model Diagnostics**: 进行模型诊断检查 / Perform model diagnostic checks

---

**时间序列分析** - 让时间序列数据处理更简单、更高效！

**Time Series Analysis** - Making time series data processing simpler and more efficient!

## 常见问题 / FAQ

### Q1: 何时用移动平均，何时用指数平滑？

| 场景 | 方法 | 原因 |
|------|------|------|
| 趋势稳定的平滑 | 移动平均 | 简单，计算快 |
| 近期数据更重要 | 指数平滑（EWMA）| 近期数据权重指数衰减 |
| 有明显季节性 | Holt-Winters | 同时捕捉趋势和季节性 |
| 需要自动选择 | ARIMA | AIC/BIC 自动定阶 |

### Q2: ARIMA 阶数 (p, d, q) 怎么选？

- **d（差分阶数）**：让序列平稳所需的差分次数。通常 d=0 或 d=1，d=2 很少用。
- **p（自相关阶数）**：用 ACF（自相关函数）和 PACF（偏自相关函数）判断：PACF 在 p 阶后截尾 → p = 该阶数
- **q（移动平均阶数）**：ACF 在 q 阶后截尾 → q = 该阶数

实际中先用 `d=0` 尝试，若序列仍不平稳逐步加 d；p 和 q 通常不超过 5。

### Q3: 预测值和实际值差距大？

1. **数据不稳定** → 先检验平稳性（ADF 检验），不平稳先差分
2. **季节性未捕捉** → 考虑 Holt-Winters 或季节性 ARIMA（SARIMA）
3. **外推太远** → 时间序列预测只适合短期，长期预测误差急剧增大
4. **异常值影响** → 先去噪或用中值滤波预处理

### Q4: 如何判断模型好坏？

主要指标：
- **MAE / RMSE**：预测误差的平均幅度，越小越好
- **MAPE**：误差相对于实际值的百分比，适合业务理解
- **AIC / BIC**：同时惩罚误差和模型复杂度，越小越好，用于模型选择

建议同时报告多个指标，而不仅仅看 RMSE。

### Q5: 非等间隔时间序列怎么处理？

`TimeSeriesData` 支持非等间隔数据，但：
1. 插值到等间隔后再用标准方法（`TimeSeriesData.interpolate()`）
2. 或使用适合不等间隔的模型（如 Gaussian Process）
3. 趋势分析时注明时间间隔变化点
