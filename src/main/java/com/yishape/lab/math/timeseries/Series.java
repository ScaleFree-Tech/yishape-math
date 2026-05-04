package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.timeseries.model.*;
import com.yishape.lab.math.plot.IPlot;

import java.time.LocalDateTime;
import java.util.List;

// Import enums and classes needed for the Series factory methods
import com.yishape.lab.math.timeseries.CointegrationAnalysis.TrendType;
import com.yishape.lab.math.timeseries.model.TimeSeriesModelFactory.SelectionCriterion;


/**
 * 时间序列分析的静态工厂入口类 / Static Factory Entry Class for Time Series Analysis
 * <p>
 * 提供统一的时间序列分析接口，封装了时间序列预测、分解、过滤、协整分析等核心功能。
 * Provides a unified time序列分析 interface that encapsulates core functions such as 
 * time series forecasting, decomposition, filtering, cointegration analysis, etc.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class Series {
    
    // ========== 时间序列数据处理方法 / Time Series Data Processing Methods ==========
    
    /**
     * 创建时间序列数据对象 / Create Time Series Data Object
     * <p>
     * 使用一维向量和名称创建时间序列数据对象。
     * Create time series data object using one-dimensional vector and name.
     * </p>
     *
     * @param data 时间序列数据向量 / Time series data vector
     * @param name 数据名称 / Data name
     * @return 时间序列数据对象 / Time series data object
     * @see TimeSeriesData#of(IVector, String)
     */
    public static TimeSeriesData createTimeSeries(IVector<Double> data, String name) {
        return TimeSeriesData.of(data, name);
    }
    
    /**
     * 创建时间序列数据对象 / Create Time Series Data Object
     * <p>
     * 使用一维向量、名称和采样频率创建时间序列数据对象。
     * Create time series data object using one-dimensional vector, name, and sampling frequency.
     * </p>
     *
     * @param data 时间序列数据向量 / Time series data vector
     * @param name 数据名称 / Data name
     * @param frequency 采样频率 / Sampling frequency
     * @return 时间序列数据对象 / Time series data object
     * @see TimeSeriesData#of(IVector, String, double)
     */
    public static TimeSeriesData createTimeSeries(IVector<Double> data, String name, double frequency) {
        return TimeSeriesData.of(data, name, frequency);
    }
    
    /**
     * 创建时间序列数据对象 / Create Time Series Data Object
     * <p>
     * 使用一维向量、采样率、名称和起始时间创建时间序列数据对象。
     * Create time series data object using one-dimensional vector, sampling rate, name, and start time.
     * </p>
     *
     * @param data 时间序列数据向量 / Time series data vector
     * @param samplingRate 采样率 / Sampling rate
     * @param name 数据名称 / Data name
     * @param startTime 起始时间 / Start time
     * @return 时间序列数据对象 / Time series data object
     * @see TimeSeriesData#of(IVector, double, String, LocalDateTime)
     */
    public static TimeSeriesData createTimeSeries(IVector<Double> data, double samplingRate, String name, LocalDateTime startTime) {
        return TimeSeriesData.of(data, samplingRate, name, startTime);
    }
    
    /**
     * 创建时间序列数据对象 / Create Time Series Data Object
     * <p>
     * 使用时间戳数组、值数组和名称创建时间序列数据对象。
     * Create time series data object using timestamp array, value array, and name.
     * </p>
     *
     * @param timestamps 时间戳数组 / Timestamp array
     * @param values 值数组 / Value array
     * @param name 数据名称 / Data name
     * @return 时间序列数据对象 / Time series data object
     * @see TimeSeriesData#of(LocalDateTime[], double[], String)
     */
    public static TimeSeriesData createTimeSeries(LocalDateTime[] timestamps, double[] values, String name) {
        return TimeSeriesData.of(timestamps, values, name);
    }
    
    /**
     * 创建多变量时间序列数据对象 / Create Multivariate Time Series Data Object
     * <p>
     * 使用时间戳数组、二维数据数组和变量名数组创建多变量时间序列数据对象。
     * Create multivariate time series data object using timestamp array, two-dimensional data array, and variable names array.
     * </p>
     *
     * @param timestamps 时间戳数组 / Timestamp array
     * @param data 二维数据数组 / Two-dimensional data array
     * @param variableNames 变量名数组 / Variable names array
     * @return 时间序列数据对象 / Time series data object
     * @see TimeSeriesData#of(LocalDateTime[], double[][], String[])
     */
    public static TimeSeriesData createMultivariateTimeSeries(LocalDateTime[] timestamps, double[][] data, String[] variableNames) {
        return TimeSeriesData.of(timestamps, data, variableNames);
    }
    
    /**
     * 创建时间序列数据构建器 / Create Time Series Data Builder
     * <p>
     * 返回一个时间序列数据构建器，用于链式构建时间序列数据对象。
     * Return a time series data builder for chain-style construction of time series data objects.
     * </p>
     *
     * @return 时间序列数据构建器 / Time series data builder
     * @see TimeSeriesData#builder()
     */
    public static TimeSeriesData.Builder createTimeSeriesBuilder() {
        return TimeSeriesData.builder();
    }
    
    // ========== 时间序列预测方法 / Time Series Forecasting Methods ==========

    /**
     * 简单移动平均预测 / Simple Moving Average Forecasting
     * <p>
     * 使用简单移动平均方法进行时间序列预测。
     * Use simple moving average method for time series forecasting.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param windowSize 窗口大小 / Window size
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#simpleMovingAverage(TimeSeriesData, int, int, int, double)
     */
    public static ForecastResult simpleMovingAverage(TimeSeriesData timeSeries, int variableIndex, 
                                                   int windowSize, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.simpleMovingAverage(timeSeries, variableIndex, windowSize, forecastSteps, confidenceLevel);
    }
    
    /**
     * 简单移动平均预测（按列名） / Simple Moving Average Forecasting (by column name)
     * <p>
     * 使用简单移动平均方法进行时间序列预测（按列名指定变量）。
     * Use simple moving average method for time series forecasting (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param windowSize 窗口大小 / Window size
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#simpleMovingAverage(TimeSeriesData, String, int, int, double)
     */
    public static ForecastResult simpleMovingAverage(TimeSeriesData timeSeries, String columnName, 
                                                   int windowSize, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.simpleMovingAverage(timeSeries, columnName, windowSize, forecastSteps, confidenceLevel);
    }
    
    /**
     * 指数平滑预测 / Exponential Smoothing Forecasting
     * <p>
     * 使用指数平滑方法进行时间序列预测。
     * Use exponential smoothing method for time series forecasting.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param alpha 平滑参数 / Smoothing parameter
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#exponentialSmoothing(TimeSeriesData, int, double, int, double)
     */
    public static ForecastResult exponentialSmoothing(TimeSeriesData timeSeries, int variableIndex, 
                                                    double alpha, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.exponentialSmoothing(timeSeries, variableIndex, alpha, forecastSteps, confidenceLevel);
    }
    
    /**
     * 指数平滑预测（按列名） / Exponential Smoothing Forecasting (by column name)
     * <p>
     * 使用指数平滑方法进行时间序列预测（按列名指定变量）。
     * Use exponential smoothing method for time series forecasting (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param alpha 平滑参数 / Smoothing parameter
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#exponentialSmoothing(TimeSeriesData, String, double, int, double)
     */
    public static ForecastResult exponentialSmoothing(TimeSeriesData timeSeries, String columnName, 
                                                    double alpha, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.exponentialSmoothing(timeSeries, columnName, alpha, forecastSteps, confidenceLevel);
    }
    
    /**
     * 线性回归预测 / Linear Regression Forecasting
     * <p>
     * 使用线性回归方法进行时间序列预测。
     * Use linear regression method for time series forecasting.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#linearRegression(TimeSeriesData, int, int, double)
     */
    public static ForecastResult linearRegression(TimeSeriesData timeSeries, int variableIndex, 
                                                int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.linearRegression(timeSeries, variableIndex, forecastSteps, confidenceLevel);
    }
    
    /**
     * 线性回归预测（按列名） / Linear Regression Forecasting (by column name)
     * <p>
     * 使用线性回归方法进行时间序列预测（按列名指定变量）。
     * Use linear regression method for time series forecasting (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#linearRegression(TimeSeriesData, String, int, double)
     */
    public static ForecastResult linearRegression(TimeSeriesData timeSeries, String columnName, 
                                                int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.linearRegression(timeSeries, columnName, forecastSteps, confidenceLevel);
    }
    
    /**
     * ARIMA预测 / ARIMA Forecasting
     * <p>
     * 使用ARIMA模型进行时间序列预测。
     * Use ARIMA model for time series forecasting.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param p AR阶数 / AR order
     * @param d 差分阶数 / Differencing order
     * @param q MA阶数 / MA order
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#arimaForecast(TimeSeriesData, int, int, int, int, int, double)
     */
    public static ForecastResult arimaForecast(TimeSeriesData timeSeries, int variableIndex, 
                                                   int p, int d, int q, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.arimaForecast(timeSeries, variableIndex, p, d, q, forecastSteps, confidenceLevel);
    }
    
    /**
     * ARIMA预测（按列名） / ARIMA Forecasting (by column name)
     * <p>
     * 使用ARIMA模型进行时间序列预测（按列名指定变量）。
     * Use ARIMA model for time series forecasting (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param p AR阶数 / AR order
     * @param d 差分阶数 / Differencing order
     * @param q MA阶数 / MA order
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#arimaForecast(TimeSeriesData, String, int, int, int, int, double)
     */
    public static ForecastResult arimaForecast(TimeSeriesData timeSeries, String columnName, 
                                                   int p, int d, int q, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.arimaForecast(timeSeries, columnName, p, d, q, forecastSteps, confidenceLevel);
    }
    
    /**
     * 季节性预测 / Seasonal Forecasting
     * <p>
     * 考虑季节性因素的时间序列预测。
     * Time series forecasting considering seasonal factors.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param period 季节周期 / Seasonal period
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#seasonalForecast(TimeSeriesData, int, int, int, double)
     */
    public static ForecastResult seasonalForecast(TimeSeriesData timeSeries, int variableIndex, 
                                                     int period, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.seasonalForecast(timeSeries, variableIndex, period, forecastSteps, confidenceLevel);
    }
    
    /**
     * 季节性预测（按列名） / Seasonal Forecasting (by column name)
     * <p>
     * 考虑季节性因素的时间序列预测（按列名指定变量）。
     * Time series forecasting considering seasonal factors (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param period 季节周期 / Seasonal period
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#seasonalForecast(TimeSeriesData, String, int, int, double)
     */
    public static ForecastResult seasonalForecast(TimeSeriesData timeSeries, String columnName, 
                                                     int period, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.seasonalForecast(timeSeries, columnName, period, forecastSteps, confidenceLevel);
    }
    
    /**
     * Holt-Winters预测 / Holt-Winters Forecasting
     * <p>
     * 使用Holt-Winters三参数指数平滑进行预测。
     * Use Holt-Winters triple exponential smoothing for forecasting.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param alpha 水平平滑参数 / Level smoothing parameter
     * @param beta 趋势平滑参数 / Trend smoothing parameter
     * @param gamma 季节性平滑参数 / Seasonal smoothing parameter
     * @param period 季节周期 / Seasonal period
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#holtWintersForecast(TimeSeriesData, int, double, double, double, int, int, double)
     */
    public static ForecastResult holtWintersForecast(TimeSeriesData timeSeries, int variableIndex,
                                                        double alpha, double beta, double gamma, int period,
                                                        int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.holtWintersForecast(timeSeries, variableIndex, alpha, beta, gamma, period, forecastSteps, confidenceLevel);
    }
    
    /**
     * GARCH预测 / GARCH Forecasting
     * <p>
     * 使用GARCH模型进行波动率预测。
     * Use GARCH model for volatility forecasting.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param p ARCH阶数 / ARCH order
     * @param q GARCH阶数 / GARCH order
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#garchForecast(TimeSeriesData, int, int, int, int, double)
     */
    public static ForecastResult garchForecast(TimeSeriesData timeSeries, int variableIndex,
                                                      int p, int q, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.garchForecast(timeSeries, variableIndex, p, q, forecastSteps, confidenceLevel);
    }
    
    /**
     * 状态空间模型预测 / State Space Model Forecasting
     * <p>
     * 使用状态空间模型进行预测。
     * Use state space model for forecasting.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param sigmaEta 水平噪声标准差 / Level noise standard deviation
     * @param sigmaZeta 趋势噪声标准差 / Trend noise standard deviation
     * @param sigmaEpsilon 观测噪声标准差 / Observation noise standard deviation
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#stateSpaceForecast(TimeSeriesData, int, double, double, double, int, double)
     */
    public static ForecastResult stateSpaceForecast(TimeSeriesData timeSeries, int variableIndex,
                                                           double sigmaEta, double sigmaZeta, double sigmaEpsilon,
                                                           int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.stateSpaceForecast(timeSeries, variableIndex, sigmaEta, sigmaZeta, sigmaEpsilon, forecastSteps, confidenceLevel);
    }
    
    /**
     * 自动预测 / Auto Forecasting
     * <p>
     * 自动选择最优的预测模型进行预测。
     * Automatically select optimal forecasting model for prediction.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#autoForecast(TimeSeriesData, int, int, double)
     */
    public static ForecastResult autoForecast(TimeSeriesData timeSeries, int variableIndex,
                                                     int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.autoForecast(timeSeries, variableIndex, forecastSteps, confidenceLevel);
    }
    
    /**
     * 自动预测（按列名） / Auto Forecasting (by column name)
     * <p>
     * 自动选择最优的预测模型进行预测（按列名指定变量）。
     * Automatically select optimal forecasting model for prediction (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     * @see TimeSeriesForecasting#autoForecast(TimeSeriesData, int, int, double)
     */
    public static ForecastResult autoForecast(TimeSeriesData timeSeries, String columnName,
                                                     int forecastSteps, double confidenceLevel) {
        int index = timeSeries.getVariableIndex(columnName);
        return TimeSeriesForecasting.autoForecast(timeSeries, index, forecastSteps, confidenceLevel);
    }
    
    // ========== 时间序列分解方法 / Time Series Decomposition Methods ==========

    /**
     * 经典分解 / Classical Decomposition
     * <p>
     * 使用经典方法分解时间序列为趋势、季节性和残差成分。
     * Use classical method to decompose time series into trend, seasonal, and residual components.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param period 季节周期 / Seasonal period
     * @param type 分解模型 / Decomposition model
     * @return 分解结果 / Decomposition result
     * @see TimeSeriesDecomposition#classicalDecomposition(TimeSeriesData, int, int, TimeSeriesDecomposition.DecompositionModel)
     */
    public static DecompositionResult classicalDecomposition(TimeSeriesData timeSeries, int variableIndex, 
                                                              int period, TimeSeriesDecomposition.DecompositionModel type) {
        return TimeSeriesDecomposition.classicalDecomposition(timeSeries, variableIndex, period, type);
    }
    
    /**
     * 经典分解（按列名） / Classical Decomposition (by column name)
     * <p>
     * 使用经典方法分解时间序列为趋势、季节性和残差成分（按列名指定变量）。
     * Use classical method to decompose time series into trend, seasonal, and residual components (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param period 季节周期 / Seasonal period
     * @param type 分解模型 / Decomposition model
     * @return 分解结果 / Decomposition result
     * @see TimeSeriesDecomposition#classicalDecomposition(TimeSeriesData, String, int, TimeSeriesDecomposition.DecompositionModel)
     */
    public static DecompositionResult classicalDecomposition(TimeSeriesData timeSeries, String columnName, 
                                                              int period, TimeSeriesDecomposition.DecompositionModel type) {
        return TimeSeriesDecomposition.classicalDecomposition(timeSeries, columnName, period, type);
    }
    
    /**
     * X-13分解 / X-13 Decomposition
     * <p>
     * 使用X-13ARIMA-SEATS方法进行时间序列分解。
     * Use X-13ARIMA-SEATS method for time series decomposition.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param period 季节周期 / Seasonal period
     * @return 分解结果 / Decomposition result
     * @see TimeSeriesDecomposition#x13Decomposition(TimeSeriesData, int, int)
     */
    public static DecompositionResult x13Decomposition(TimeSeriesData timeSeries, int variableIndex, int period) {
        return TimeSeriesDecomposition.x13Decomposition(timeSeries, variableIndex, period);
    }
    
    /**
     * X-13分解（按列名） / X-13 Decomposition (by column name)
     * <p>
     * 使用X-13ARIMA-SEATS方法进行时间序列分解（按列名指定变量）。
     * Use X-13ARIMA-SEATS method for time series decomposition (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param period 季节周期 / Seasonal period
     * @return 分解结果 / Decomposition result
     * @see TimeSeriesDecomposition#x13Decomposition(TimeSeriesData, String, int)
     */
    public static DecompositionResult x13Decomposition(TimeSeriesData timeSeries, String columnName, int period) {
        return TimeSeriesDecomposition.x13Decomposition(timeSeries, columnName, period);
    }
    
    /**
     * STL分解 / STL Decomposition
     * <p>
     * 使用STL（Seasonal and Trend decomposition using Loess）方法进行分解。
     * Use STL (Seasonal and Trend decomposition using Loess) method for decomposition.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param period 季节周期 / Seasonal period
     * @param seasonalWindow 季节性窗口 / Seasonal window
     * @param trendWindow 趋势窗口 / Trend window
     * @return 分解结果 / Decomposition result
     * @see TimeSeriesDecomposition#stlDecomposition(TimeSeriesData, int, int, int, int)
     */
    public static DecompositionResult stlDecomposition(TimeSeriesData timeSeries, int variableIndex, 
                                                        int period, int seasonalWindow, int trendWindow) {
        return TimeSeriesDecomposition.stlDecomposition(timeSeries, variableIndex, period, seasonalWindow, trendWindow);
    }
    
    /**
     * STL分解（按列名） / STL Decomposition (by column name)
     * <p>
     * 使用STL方法进行分解（按列名指定变量）。
     * Use STL method for decomposition (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param period 季节周期 / Seasonal period
     * @param seasonalWindow 季节性窗口 / Seasonal window
     * @param trendWindow 趋势窗口 / Trend window
     * @return 分解结果 / Decomposition result
     * @see TimeSeriesDecomposition#stlDecomposition(TimeSeriesData, String, int, int, int)
     */
    public static DecompositionResult stlDecomposition(TimeSeriesData timeSeries, String columnName, 
                                                        int period, int seasonalWindow, int trendWindow) {
        return TimeSeriesDecomposition.stlDecomposition(timeSeries, columnName, period, seasonalWindow, trendWindow);
    }
    
    /**
     * 小波分解 / Wavelet Decomposition
     * <p>
     * 使用小波变换进行时间序列分解。
     * Use wavelet transform for time series decomposition.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     * @return 分解结果 / Decomposition result
     * @see TimeSeriesDecomposition#waveletDecomposition(TimeSeriesData, int, String, int)
     */
    public static DecompositionResult waveletDecomposition(TimeSeriesData timeSeries, int variableIndex, 
                                                            String waveletType, int levels) {
        return TimeSeriesDecomposition.waveletDecomposition(timeSeries, variableIndex, waveletType, levels);
    }
    
    /**
     * 小波分解（按列名） / Wavelet Decomposition (by column name)
     * <p>
     * 使用小波变换进行时间序列分解（按列名指定变量）。
     * Use wavelet transform for time series decomposition (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     * @return 分解结果 / Decomposition result
     * @see TimeSeriesDecomposition#waveletDecomposition(TimeSeriesData, String, String, int)
     */
    public static DecompositionResult waveletDecomposition(TimeSeriesData timeSeries, String columnName, 
                                                            String waveletType, int levels) {
        return TimeSeriesDecomposition.waveletDecomposition(timeSeries, columnName, waveletType, levels);
    }
    
    // ========== 时间序列过滤方法 / Time Series Filtering Methods ==========

    /**
     * 移动平均滤波 / Moving Average Filtering
     * <p>
     * 使用移动平均滤波器平滑时间序列。
     * Use moving average filter to smooth time series.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param windowSize 窗口大小 / Window size
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#movingAverage(TimeSeriesData, int, int)
     */
    public static FilterResult movingAverage(TimeSeriesData timeSeries, int variableIndex, int windowSize) {
        return TimeSeriesFiltering.movingAverage(timeSeries, variableIndex, windowSize);
    }
    
    /**
     * 移动平均滤波（按列名） / Moving Average Filtering (by column name)
     * <p>
     * 使用移动平均滤波器平滑时间序列（按列名指定变量）。
     * Use moving average filter to smooth time series (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param windowSize 窗口大小 / Window size
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#movingAverage(TimeSeriesData, String, int)
     */
    public static FilterResult movingAverage(TimeSeriesData timeSeries, String columnName, int windowSize) {
        return TimeSeriesFiltering.movingAverage(timeSeries, columnName, windowSize);
    }
    
    /**
     * 指数平滑滤波 / Exponential Smoothing Filtering
     * <p>
     * 使用指数平滑滤波器平滑时间序列。
     * Use exponential smoothing filter to smooth time series.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param alpha 平滑参数 / Smoothing parameter
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#exponentialSmoothing(TimeSeriesData, int, double)
     */
    public static FilterResult exponentialSmoothing(TimeSeriesData timeSeries, int variableIndex, double alpha) {
        return TimeSeriesFiltering.exponentialSmoothing(timeSeries, variableIndex, alpha);
    }
    
    /**
     * 指数平滑滤波（按列名） / Exponential Smoothing Filtering (by column name)
     * <p>
     * 使用指数平滑滤波器平滑时间序列（按列名指定变量）。
     * Use exponential smoothing filter to smooth time series (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param alpha 平滑参数 / Smoothing parameter
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#exponentialSmoothing(TimeSeriesData, String, double)
     */
    public static FilterResult exponentialSmoothing(TimeSeriesData timeSeries, String columnName, double alpha) {
        return TimeSeriesFiltering.exponentialSmoothing(timeSeries, columnName, alpha);
    }
    
    /**
     * 高斯滤波 / Gaussian Filtering
     * <p>
     * 使用高斯滤波器平滑时间序列。
     * Use Gaussian filter to smooth time series.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param sigma 高斯标准差 / Gaussian standard deviation
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#gaussianFilter(TimeSeriesData, int, double)
     */
    public static FilterResult gaussianFilter(TimeSeriesData timeSeries, int variableIndex, double sigma) {
        return TimeSeriesFiltering.gaussianFilter(timeSeries, variableIndex, sigma);
    }
    
    /**
     * 高斯滤波（按列名） / Gaussian Filtering (by column name)
     * <p>
     * 使用高斯滤波器平滑时间序列（按列名指定变量）。
     * Use Gaussian filter to smooth time series (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param sigma 高斯标准差 / Gaussian standard deviation
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#gaussianFilter(TimeSeriesData, String, double)
     */
    public static FilterResult gaussianFilter(TimeSeriesData timeSeries, String columnName, double sigma) {
        return TimeSeriesFiltering.gaussianFilter(timeSeries, columnName, sigma);
    }
    
    /**
     * 中值滤波 / Median Filtering
     * <p>
     * 使用中值滤波器平滑时间序列。
     * Use median filter to smooth time series.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param windowSize 窗口大小 / Window size
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#medianFilter(TimeSeriesData, int, int)
     */
    public static FilterResult medianFilter(TimeSeriesData timeSeries, int variableIndex, int windowSize) {
        return TimeSeriesFiltering.medianFilter(timeSeries, variableIndex, windowSize);
    }
    
    /**
     * 中值滤波（按列名） / Median Filtering (by column name)
     * <p>
     * 使用中值滤波器平滑时间序列（按列名指定变量）。
     * Use median filter to smooth time series (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param windowSize 窗口大小 / Window size
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#medianFilter(TimeSeriesData, String, int)
     */
    public static FilterResult medianFilter(TimeSeriesData timeSeries, String columnName, int windowSize) {
        return TimeSeriesFiltering.medianFilter(timeSeries, columnName, windowSize);
    }
    
    /**
     * 低通滤波 / Low-pass Filtering
     * <p>
     * 使用低通滤波器去除高频成分。
     * Use low-pass filter to remove high frequency components.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param order 滤波器阶数 / Filter order
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#lowPassFilter(TimeSeriesData, int, double, int)
     */
    public static FilterResult lowPassFilter(TimeSeriesData timeSeries, int variableIndex, 
                                                 double cutoffFreq, int order) {
        return TimeSeriesFiltering.lowPassFilter(timeSeries, variableIndex, cutoffFreq, order);
    }
    
    /**
     * 低通滤波（按列名） / Low-pass Filtering (by column name)
     * <p>
     * 使用低通滤波器去除高频成分（按列名指定变量）。
     * Use low-pass filter to remove high frequency components (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param order 滤波器阶数 / Filter order
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#lowPassFilter(TimeSeriesData, String, double, int)
     */
    public static FilterResult lowPassFilter(TimeSeriesData timeSeries, String columnName, 
                                                 double cutoffFreq, int order) {
        return TimeSeriesFiltering.lowPassFilter(timeSeries, columnName, cutoffFreq, order);
    }
    
    /**
     * 高通滤波 / High-pass Filtering
     * <p>
     * 使用高通滤波器去除低频成分。
     * Use high-pass filter to remove low frequency components.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param order 滤波器阶数 / Filter order
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#highPassFilter(TimeSeriesData, int, double, int)
     */
    public static FilterResult highPassFilter(TimeSeriesData timeSeries, int variableIndex, 
                                                  double cutoffFreq, int order) {
        return TimeSeriesFiltering.highPassFilter(timeSeries, variableIndex, cutoffFreq, order);
    }
    
    /**
     * 高通滤波（按列名） / High-pass Filtering (by column name)
     * <p>
     * 使用高通滤波器去除低频成分（按列名指定变量）。
     * Use high-pass filter to remove low frequency components (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param order 滤波器阶数 / Filter order
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#highPassFilter(TimeSeriesData, String, double, int)
     */
    public static FilterResult highPassFilter(TimeSeriesData timeSeries, String columnName, 
                                                  double cutoffFreq, int order) {
        return TimeSeriesFiltering.highPassFilter(timeSeries, columnName, cutoffFreq, order);
    }
    
    /**
     * 带通滤波 / Band-pass Filtering
     * <p>
     * 使用带通滤波器保留特定频率范围内的成分。
     * Use band-pass filter to retain components within specific frequency range.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param lowFreq 低频截止 / Low frequency cutoff
     * @param highFreq 高频截止 / High frequency cutoff
     * @param order 滤波器阶数 / Filter order
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#bandPassFilter(TimeSeriesData, int, double, double, int)
     */
    public static FilterResult bandPassFilter(TimeSeriesData timeSeries, int variableIndex, 
                                                   double lowFreq, double highFreq, int order) {
        return TimeSeriesFiltering.bandPassFilter(timeSeries, variableIndex, lowFreq, highFreq, order);
    }
    
    /**
     * 带通滤波（按列名） / Band-pass Filtering (by column name)
     * <p>
     * 使用带通滤波器保留特定频率范围内的成分（按列名指定变量）。
     * Use band-pass filter to retain components within specific frequency range (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param lowFreq 低频截止 / Low frequency cutoff
     * @param highFreq 高频截止 / High frequency cutoff
     * @param order 滤波器阶数 / Filter order
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#bandPassFilter(TimeSeriesData, String, double, double, int)
     */
    public static FilterResult bandPassFilter(TimeSeriesData timeSeries, String columnName, 
                                                   double lowFreq, double highFreq, int order) {
        return TimeSeriesFiltering.bandPassFilter(timeSeries, columnName, lowFreq, highFreq, order);
    }
    
    /**
     * 自适应滤波 / Adaptive Filtering
     * <p>
     * 使用自适应滤波器进行时间序列滤波。
     * Use adaptive filter for time series filtering.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param learningRate 学习率 / Learning rate
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#adaptiveFilter(TimeSeriesData, int, double)
     */
    public static FilterResult adaptiveFilter(TimeSeriesData timeSeries, int variableIndex, double learningRate) {
        return TimeSeriesFiltering.adaptiveFilter(timeSeries, variableIndex, learningRate);
    }
    
    /**
     * 自适应滤波（按列名） / Adaptive Filtering (by column name)
     * <p>
     * 使用自适应滤波器进行时间序列滤波（按列名指定变量）。
     * Use adaptive filter for time series filtering (specify variable by column name).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param learningRate 学习率 / Learning rate
     * @return 滤波结果 / Filter result
     * @see TimeSeriesFiltering#adaptiveFilter(TimeSeriesData, String, double)
     */
    public static FilterResult adaptiveFilter(TimeSeriesData timeSeries, String columnName, double learningRate) {
        return TimeSeriesFiltering.adaptiveFilter(timeSeries, columnName, learningRate);
    }
    
    // ========== 协整分析方法 / Cointegration Analysis Methods ==========

    /**
     * Engle-Granger协整检验 / Engle-Granger Cointegration Test
     * <p>
     * 使用Engle-Granger方法检验两个时间序列之间的协整关系。
     * Use Engle-Granger method to test cointegration relationship between two time series.
     * </p>
     *
     * @param y 第一个时间序列 / First time series
     * @param x 第二个时间序列 / Second time series
     * @param maxLags 最大滞后期数 / Maximum lag order
     * @return 协整检验结果 / Cointegration test result
     * @see CointegrationAnalysis#engleGrangerTest(IVector, IVector, int)
     */
    public static CointegrationAnalysis.EngleGrangerResult engleGrangerTest(IVector<Double> y, IVector<Double> x, int maxLags) {
        return CointegrationAnalysis.engleGrangerTest(y, x, maxLags);
    }
    
    /**
     * Johansen协整检验 / Johansen Cointegration Test
     * <p>
     * 使用Johansen方法检验多变量时间序列之间的协整关系。
     * Use Johansen method to test cointegration relationship among multivariate time series.
     * </p>
     *
     * @param data 多变量时间序列数据 / Multivariate time series data
     * @param maxLags 最大滞后期数 / Maximum lag order
     * @param trendType 趋势类型 / Trend type
     * @return 协整检验结果 / Cointegration test result
     * @see CointegrationAnalysis#johansenTest(IMatrix, int, TrendType)
     */
    public static CointegrationAnalysis.JohansenResult johansenTest(IMatrix<Double> data, int maxLags, CointegrationAnalysis.TrendType trendType) {
        return CointegrationAnalysis.johansenTest(data, maxLags, trendType);
    }
    
    /**
     * 估计协整关系 / Estimate Cointegrating Relationship
     * <p>
     * 估计两个时间序列之间的协整关系。
     * Estimate cointegrating relationship between two time series.
     * </p>
     *
     * @param y 第一个时间序列 / First time series
     * @param x 第二个时间序列 / Second time series
     * @return 协整关系对象 / Cointegrating relationship object
     * @see CointegrationAnalysis#estimateCointegratingRelationship(IVector, IVector)
     */
    public static CointegrationAnalysis.CointegratingRelationship estimateCointegratingRelationship(IVector<Double> y, IVector<Double> x) {
        return CointegrationAnalysis.estimateCointegratingRelationship(y, x);
    }
    
    /**
     * 估计误差修正模型 / Estimate Error Correction Model
     * <p>
     * 估计误差修正模型用于短期调整和长期均衡之间的关系。
     * Estimate error correction model for relationship between short-term adjustment and long-term equilibrium.
     * </p>
     *
     * @param deltaY 第一个序列的一阶差分 / First difference of first series
     * @param deltaX 第二个序列的一阶差分 / First difference of second series
     * @param residuals 协整残差 / Cointegration residuals
     * @param maxLags 最大滞后期数 / Maximum lag order
     * @return 误差修正模型 / Error correction model
     * @see CointegrationAnalysis#estimateECM(IVector, IVector, IVector, int)
     */
    public static CointegrationAnalysis.ErrorCorrectionModel estimateECM(IVector<Double> deltaY, IVector<Double> deltaX, 
                                                          IVector<Double> residuals, int maxLags) {
        return CointegrationAnalysis.estimateECM(deltaY, deltaX, residuals, maxLags);
    }
    
    // ========== 时间序列模型方法 / Time Series Model Methods ==========

    /**
     * 创建ARIMA模型 / Create ARIMA Model
     * <p>
     * 使用指定的ARIMA参数创建模型。
     * Create ARIMA model with specified parameters.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param p AR阶数 / AR order
     * @param d 差分阶数 / Differencing order
     * @param q MA阶数 / MA order
     * @return ARIMA模型 / ARIMA model
     * @see TimeSeriesModelFactory#createARIMAModel(IVector, int, int, int)
     */
    public static ITimeSeriesModel createARIMAModel(IVector<Double> data, int p, int d, int q) {
        return TimeSeriesModelFactory.createARIMAModel(data, p, d, q);
    }
    
    /**
     * 自动选择ARIMA模型 / Auto-select ARIMA Model
     * <p>
     * 使用AIC或BIC准则自动选择最优的ARIMA模型参数。
     * Automatically select optimal ARIMA model parameters using AIC or BIC criteria.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param maxP 最大AR阶数 / Maximum AR order
     * @param maxD 最大差分阶数 / Maximum differencing order
     * @param maxQ 最大MA阶数 / Maximum MA order
     * @param criterion 选择准则 / Selection criterion
     * @return 最优ARIMA模型 / Optimal ARIMA model
     * @see TimeSeriesModelFactory#createARIMAModel(IVector, int, int, int, SelectionCriterion)
     */
    public static ITimeSeriesModel createARIMAModel(IVector<Double> data, int maxP, int maxD, int maxQ, 
                                                   TimeSeriesModelFactory.SelectionCriterion criterion) {
        return TimeSeriesModelFactory.createARIMAModel(data, maxP, maxD, maxQ, criterion);
    }
    
    /**
     * 创建指数平滑模型 / Create Exponential Smoothing Model
     * <p>
     * 创建简单指数平滑模型。
     * Create simple exponential smoothing model.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param alpha 平滑参数 / Smoothing parameter
     * @return 简单指数平滑模型 / Simple exponential smoothing model
     * @see ExponentialSmoothingModels.SimpleExponentialSmoothing#fit(IVector, double)
     */
    public static ExponentialSmoothingModels.SimpleExponentialSmoothing createSimpleExponentialSmoothing(IVector<Double> data, double alpha) {
        return ExponentialSmoothingModels.SimpleExponentialSmoothing.fit(data, alpha);
    }
    
    /**
     * 创建双指数平滑模型 / Create Double Exponential Smoothing Model
     * <p>
     * 创建双指数平滑模型（Holt方法）。
     * Create double exponential smoothing model (Holt's method).
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param alpha 水平平滑参数 / Level smoothing parameter
     * @param beta 趋势平滑参数 / Trend smoothing parameter
     * @return 双指数平滑模型 / Double exponential smoothing model
     * @see ExponentialSmoothingModels.DoubleExponentialSmoothing#fit(IVector, double, double)
     */
    public static ExponentialSmoothingModels.DoubleExponentialSmoothing createDoubleExponentialSmoothing(IVector<Double> data, double alpha, double beta) {
        return ExponentialSmoothingModels.DoubleExponentialSmoothing.fit(data, alpha, beta);
    }
    
    /**
     * 创建Holt-Winters平滑模型 / Create Holt-Winters Smoothing Model
     * <p>
     * 创建Holt-Winters三参数指数平滑模型。
     * Create Holt-Winters triple exponential smoothing model.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param alpha 水平平滑参数 / Level smoothing parameter
     * @param beta 趋势平滑参数 / Trend smoothing parameter
     * @param gamma 季节性平滑参数 / Seasonal smoothing parameter
     * @param period 季节周期 / Seasonal period
     * @return Holt-Winters平滑模型 / Holt-Winters smoothing model
     * @see ExponentialSmoothingModels.HoltWintersSmoothing#fit(IVector, double, double, double, int)
     */
    public static ExponentialSmoothingModels.HoltWintersSmoothing createHoltWintersSmoothing(IVector<Double> data, double alpha, double beta, 
                                                                 double gamma, int period) {
        return ExponentialSmoothingModels.HoltWintersSmoothing.fit(data, alpha, beta, gamma, period);
    }
    
    /**
     * 创建自适应指数平滑模型 / Create Adaptive Exponential Smoothing Model
     * <p>
     * 创建自适应指数平滑模型，平滑参数随时间自适应调整。
     * Create adaptive exponential smoothing model with smoothing parameter adjusting adaptively over time.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param initialAlpha 初始平滑参数 / Initial smoothing parameter
     * @param adaptationRate 自适应率 / Adaptation rate
     * @return 自适应指数平滑模型 / Adaptive exponential smoothing model
     * @see ExponentialSmoothingModels.AdaptiveExponentialSmoothing#fit(IVector, double, double)
     */
    public static ExponentialSmoothingModels.AdaptiveExponentialSmoothing createAdaptiveExponentialSmoothing(IVector<Double> data, 
                                                                      double initialAlpha, double adaptationRate) {
        return ExponentialSmoothingModels.AdaptiveExponentialSmoothing.fit(data, initialAlpha, adaptationRate);
    }
    
    /**
     * 选择最佳指数平滑模型 / Select Best Exponential Smoothing Model
     * <p>
     * 自动搜索并选择最优的指数平滑模型。
     * Automatically search and select optimal exponential smoothing model.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param maxPeriod 最大季节周期 / Maximum seasonal period
     * @return 模型选择结果 / Model selection result
     * @see ExponentialSmoothingModels.ModelSelector#selectBestModel(IVector, int)
     */
    public static ExponentialSmoothingModels.ModelSelectionResult selectBestExponentialSmoothingModel(IVector<Double> data, int maxPeriod) {
        return ExponentialSmoothingModels.ModelSelector.selectBestModel(data, maxPeriod);
    }
    
    /**
     * 创建GARCH模型 / Create GARCH Model
     * <p>
     * 创建GARCH模型用于波动率建模。
     * Create GARCH model for volatility modeling.
     * </p>
     *
     * @param returns 收益率序列 / Returns series
     * @param p ARCH阶数 / ARCH order
     * @param q GARCH阶数 / GARCH order
     * @return GARCH模型 / GARCH model
     * @see GARCHModel#fit(IVector, int, int)
     */
    public static GARCHModel createGARCHModel(IVector<Double> returns, int p, int q) {
        return GARCHModel.fit(returns, p, q);
    }
    
    /**
     * 自动拟合GARCH模型 / Auto-fit GARCH Model
     * <p>
     * 使用AIC或BIC准则自动选择最优的GARCH模型参数。
     * Automatically select optimal GARCH model parameters using AIC or BIC criteria.
     * </p>
     *
     * @param returns 收益率序列 / Returns series
     * @param maxP 最大ARCH阶数 / Maximum ARCH order
     * @param maxQ 最大GARCH阶数 / Maximum GARCH order
     * @param criterion 选择准则 / Selection criterion
     * @return 最优GARCH模型 / Optimal GARCH model
     * @see GARCHModel#autoFit(IVector, int, int, GARCHModel.SelectionCriterion)
     */
    public static GARCHModel autoFitGARCHModel(IVector<Double> returns, int maxP, int maxQ, TimeSeriesModelFactory.SelectionCriterion criterion) {
        // Convert SelectionCriterion to GARCHModel.SelectionCriterion
        GARCHModel.SelectionCriterion garchCriterion = GARCHModel.SelectionCriterion.AIC;
        if (criterion == TimeSeriesModelFactory.SelectionCriterion.BIC) {
            garchCriterion = GARCHModel.SelectionCriterion.BIC;
        }
        return GARCHModel.autoFit(returns, maxP, maxQ, garchCriterion);
    }
    
    /**
     * 创建VAR模型 / Create VAR Model
     * <p>
     * 创建向量自回归模型用于多变量时间序列建模。
     * Create vector autoregression model for multivariate time series modeling.
     * </p>
     *
     * @param data 多变量时间序列数据 / Multivariate time series data
     * @param p VAR阶数 / VAR order
     * @param variableNames 变量名数组 / Variable names array
     * @return VAR模型 / VAR model
     * @see VARModel#fit(IMatrix, int, String[])
     */
    public static VARModel createVARModel(IMatrix<Double> data, int p, String[] variableNames) {
        return VARModel.fit(data, p, variableNames);
    }
    
    /**
     * 自动拟合VAR模型 / Auto-fit VAR Model
     * <p>
     * 使用AIC或BIC准则自动选择最优的VAR模型参数。
     * Automatically select optimal VAR model parameters using AIC or BIC criteria.
     * </p>
     *
     * @param data 多变量时间序列数据 / Multivariate time series data
     * @param maxP 最大VAR阶数 / Maximum VAR order
     * @param criterion 选择准则 / Selection criterion
     * @param variableNames 变量名数组 / Variable names array
     * @return 最优VAR模型 / Optimal VAR model
     * @see VARModel#autoFit(IMatrix, int, VARModel.SelectionCriterion, String[])
     */
    public static VARModel autoFitVARModel(IMatrix<Double> data, int maxP, TimeSeriesModelFactory.SelectionCriterion criterion, String[] variableNames) {
        // Convert SelectionCriterion to VARModel.SelectionCriterion
        VARModel.SelectionCriterion varCriterion = VARModel.SelectionCriterion.AIC;
        if (criterion == TimeSeriesModelFactory.SelectionCriterion.BIC) {
            varCriterion = VARModel.SelectionCriterion.BIC;
        }
        return VARModel.autoFit(data, maxP, varCriterion, variableNames);
    }
    
    // ========== 时间序列诊断方法 / Time Series Diagnostics Methods ==========

    /**
     * 执行ARIMA诊断 / Perform ARIMA Diagnostics
     * <p>
     * 对ARIMA模型的残差进行诊断分析。
     * Perform diagnostic analysis on ARIMA model residuals.
     * </p>
     *
     * @param residuals 残差序列 / Residuals series
     * @param originalData 原始数据 / Original data
     * @param fittedValues 拟合值 / Fitted values
     * @param arCoeffs AR系数 / AR coefficients
     * @param maCoeffs MA系数 / MA coefficients
     * @param sigma2 方差 / Variance
     * @param aic AIC信息准则 / AIC information criterion
     * @param bic BIC信息准则 / BIC information criterion
     * @param logLikelihood 对数似然 / Log likelihood
     * @return ARIMA诊断结果 / ARIMA diagnostics result
     * @see ARIMADiagnostics#ARIMADiagnostics(IVector, IVector, IVector, IVector, IVector, double, double, double, double)
     */
    public static ARIMADiagnostics performARIMADiagnostics(IVector<Double> residuals, IVector<Double> originalData, 
                           IVector<Double> fittedValues, IVector<Double> arCoeffs, 
                           IVector<Double> maCoeffs, double sigma2, double aic, 
                           double bic, double logLikelihood) {
        return new ARIMADiagnostics(residuals, originalData, fittedValues, arCoeffs, maCoeffs, sigma2, aic, bic, logLikelihood);
    }
    
    // ========== 时间序列可视化方法 / Time Series Visualization Methods ==========

    /**
     * 绘制时间序列 / Plot Time Series
     * <p>
     * 绘制时间序列数据的图形。
     * Plot time series data graph.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param title 图表标题 / Chart title
     * @return 绘图结果 / Plot result
     * @see TimeSeriesPlots#plotTimeSeries(TimeSeriesData, String)
     */
    public static IPlot plotTimeSeries(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotTimeSeries(timeSeriesData, title);
    }
    
    /**
     * 绘制趋势分析 / Plot Trend Analysis
     * <p>
     * 绘制时间序列的趋势分析图。
     * Plot trend analysis graph of time series.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param title 图表标题 / Chart title
     * @return 绘图结果 / Plot result
     * @see TimeSeriesPlots#plotTrendAnalysis(TimeSeriesData, String)
     */
    public static IPlot plotTrendAnalysis(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotTrendAnalysis(timeSeriesData, title);
    }
    
    /**
     * 绘制季节性分解 / Plot Seasonal Decomposition
     * <p>
     * 绘制时间序列季节性分解的图形。
     * Plot seasonal decomposition graph of time series.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param period 季节周期 / Seasonal period
     * @param title 图表标题 / Chart title
     * @return 绘图结果 / Plot result
     * @see TimeSeriesPlots#plotSeasonalDecomposition(TimeSeriesData, int, String)
     */
    public static IPlot plotSeasonalDecomposition(TimeSeriesData timeSeriesData, int period, String title) {
        return TimeSeriesPlots.plotSeasonalDecomposition(timeSeriesData, period, title);
    }
    
    /**
     * 绘制自相关图 / Plot Autocorrelation
     * <p>
     * 绘制时间序列的自相关图。
     * Plot autocorrelation graph of time series.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param maxLag 最大滞后期数 / Maximum lag order
     * @param title 图表标题 / Chart title
     * @return 绘图结果 / Plot result
     * @see TimeSeriesPlots#plotAutocorrelation(TimeSeriesData, int, String)
     */
    public static IPlot plotAutocorrelation(TimeSeriesData timeSeriesData, int maxLag, String title) {
        return TimeSeriesPlots.plotAutocorrelation(timeSeriesData, maxLag, title);
    }
    
    /**
     * 绘制偏自相关图 / Plot Partial Autocorrelation
     * <p>
     * 绘制时间序列的偏自相关图。
     * Plot partial autocorrelation graph of time series.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param maxLag 最大滞后期数 / Maximum lag order
     * @param title 图表标题 / Chart title
     * @return 绘图结果 / Plot result
     * @see TimeSeriesPlots#plotPartialAutocorrelation(TimeSeriesData, int, String)
     */
    public static IPlot plotPartialAutocorrelation(TimeSeriesData timeSeriesData, int maxLag, String title) {
        return TimeSeriesPlots.plotPartialAutocorrelation(timeSeriesData, maxLag, title);
    }
    
    /**
     * 绘制预测结果 / Plot Forecasting
     * <p>
     * 绘制时间序列预测结果的图形。
     * Plot time series forecasting result graph.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param forecastSteps 预测步数 / Forecast steps
     * @param title 图表标题 / Chart title
     * @return 绘图结果 / Plot result
     * @see TimeSeriesPlots#plotForecasting(TimeSeriesData, int, String)
     */
    public static IPlot plotForecasting(TimeSeriesData timeSeriesData, int forecastSteps, String title) {
        return TimeSeriesPlots.plotForecasting(timeSeriesData, forecastSteps, title);
    }
    
    /**
     * 绘制时间序列统计信息 / Plot Time Series Statistics
     * <p>
     * 绘制时间序列统计信息的图形。
     * Plot time series statistics graph.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param title 图表标题 / Chart title
     * @return 绘图结果 / Plot result
     * @see TimeSeriesPlots#plotTimeSeriesStatistics(TimeSeriesData, String)
     */
    public static IPlot plotTimeSeriesStatistics(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotTimeSeriesStatistics(timeSeriesData, title);
    }
    
    /**
     * 绘制多变量时间序列 / Plot Multivariate Time Series
     * <p>
     * 绘制多变量时间序列的图形。
     * Plot multivariate time series graph.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param title 图表标题 / Chart title
     * @return 绘图结果 / Plot result
     * @see TimeSeriesPlots#plotMultivariateTimeSeries(TimeSeriesData, String)
     */
    public static IPlot plotMultivariateTimeSeries(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotMultivariateTimeSeries(timeSeriesData, title);
    }
    
    /**
     * 绘制时间序列特征 / Plot Time Series Features
     * <p>
     * 绘制时间序列特征的图形。
     * Plot time series features graph.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param title 图表标题 / Chart title
     * @return 绘图结果 / Plot result
     * @see TimeSeriesPlots#plotTimeSeriesFeatures(TimeSeriesData, String)
     */
    public static IPlot plotTimeSeriesFeatures(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotTimeSeriesFeatures(timeSeriesData, title);
    }
    
    /**
     * 创建时间序列仪表板 / Create Time Series Dashboard
     * <p>
     * 创建一个包含多个时间序列图的仪表板。
     * Create a dashboard containing multiple time series plots.
     * </p>
     *
     * @param timeSeriesData 时间序列数据 / Time series data
     * @param title 仪表板标题 / Dashboard title
     * @return 绘图列表 / Plot list
     * @see TimeSeriesPlots#createTimeSeriesDashboard(TimeSeriesData, String)
     */
    public static List<IPlot> createTimeSeriesDashboard(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.createTimeSeriesDashboard(timeSeriesData, title);
    }
    
    // ========== 时间序列分析器方法 / Time Series Analyzer Methods ==========

    /**
     * 创建时间序列分析器 / Create Time Series Analyzer
     * <p>
     * 创建一个时间序列分析器进行分析。
     * Create a time series analyzer for analysis.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param name 数据名称 / Data name
     * @return 时间序列分析器 / Time series analyzer
     * @see TimeSeriesAnalyzer#TimeSeriesAnalyzer(IVector, String)
     */
    public static TimeSeriesAnalyzer createTimeSeriesAnalyzer(IVector<Double> data, String name) {
        return new TimeSeriesAnalyzer(data, name);
    }
    
    /**
     * 创建时间序列分析器 / Create Time Series Analyzer
     * <p>
     * 创建一个带时间戳的时间序列分析器进行分析。
     * Create a time series analyzer with timestamps for analysis.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param name 数据名称 / Data name
     * @param timestamps 时间戳数组 / Timestamp array
     * @return 时间序列分析器 / Time series analyzer
     * @see TimeSeriesAnalyzer#TimeSeriesAnalyzer(IVector, String, LocalDateTime[])
     */
    public static TimeSeriesAnalyzer createTimeSeriesAnalyzer(IVector<Double> data, String name, LocalDateTime[] timestamps) {
        return new TimeSeriesAnalyzer(data, name, timestamps);
    }
}