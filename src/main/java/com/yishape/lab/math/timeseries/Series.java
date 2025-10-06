package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.timeseries.model.*;
import com.yishape.lab.math.viz.IPlot;

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
     * @see TimeSeriesData#of(IVector, String)
     */
    public static TimeSeriesData createTimeSeries(IVector<Double> data, String name) {
        return TimeSeriesData.of(data, name);
    }
    
    /**
     * 创建时间序列数据对象 / Create Time Series Data Object
     * @see TimeSeriesData#of(IVector, String, double)
     */
    public static TimeSeriesData createTimeSeries(IVector<Double> data, String name, double frequency) {
        return TimeSeriesData.of(data, name, frequency);
    }
    
    /**
     * 创建时间序列数据对象 / Create Time Series Data Object
     * @see TimeSeriesData#of(IVector, double, String, LocalDateTime)
     */
    public static TimeSeriesData createTimeSeries(IVector<Double> data, double samplingRate, String name, LocalDateTime startTime) {
        return TimeSeriesData.of(data, samplingRate, name, startTime);
    }
    
    /**
     * 创建时间序列数据对象 / Create Time Series Data Object
     * @see TimeSeriesData#of(LocalDateTime[], double[], String)
     */
    public static TimeSeriesData createTimeSeries(LocalDateTime[] timestamps, double[] values, String name) {
        return TimeSeriesData.of(timestamps, values, name);
    }
    
    /**
     * 创建多变量时间序列数据对象 / Create Multivariate Time Series Data Object
     * @see TimeSeriesData#of(LocalDateTime[], double[][], String[])
     */
    public static TimeSeriesData createMultivariateTimeSeries(LocalDateTime[] timestamps, double[][] data, String[] variableNames) {
        return TimeSeriesData.of(timestamps, data, variableNames);
    }
    
    /**
     * 创建时间序列数据构建器 / Create Time Series Data Builder
     * @see TimeSeriesData#builder()
     */
    public static TimeSeriesData.Builder createTimeSeriesBuilder() {
        return TimeSeriesData.builder();
    }
    
    // ========== 时间序列预测方法 / Time Series Forecasting Methods ==========
    
    /**
     * 简单移动平均预测 / Simple Moving Average Forecasting
     * @see TimeSeriesForecasting#simpleMovingAverage(TimeSeriesData, int, int, int, double)
     */
    public static ForecastResult simpleMovingAverage(TimeSeriesData timeSeries, int variableIndex, 
                                                   int windowSize, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.simpleMovingAverage(timeSeries, variableIndex, windowSize, forecastSteps, confidenceLevel);
    }
    
    /**
     * 简单移动平均预测（按列名） / Simple Moving Average Forecasting (by column name)
     * @see TimeSeriesForecasting#simpleMovingAverage(TimeSeriesData, String, int, int, double)
     */
    public static ForecastResult simpleMovingAverage(TimeSeriesData timeSeries, String columnName, 
                                                   int windowSize, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.simpleMovingAverage(timeSeries, columnName, windowSize, forecastSteps, confidenceLevel);
    }
    
    /**
     * 指数平滑预测 / Exponential Smoothing Forecasting
     * @see TimeSeriesForecasting#exponentialSmoothing(TimeSeriesData, int, double, int, double)
     */
    public static ForecastResult exponentialSmoothing(TimeSeriesData timeSeries, int variableIndex, 
                                                    double alpha, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.exponentialSmoothing(timeSeries, variableIndex, alpha, forecastSteps, confidenceLevel);
    }
    
    /**
     * 指数平滑预测（按列名） / Exponential Smoothing Forecasting (by column name)
     * @see TimeSeriesForecasting#exponentialSmoothing(TimeSeriesData, String, double, int, double)
     */
    public static ForecastResult exponentialSmoothing(TimeSeriesData timeSeries, String columnName, 
                                                    double alpha, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.exponentialSmoothing(timeSeries, columnName, alpha, forecastSteps, confidenceLevel);
    }
    
    /**
     * 线性回归预测 / Linear Regression Forecasting
     * @see TimeSeriesForecasting#linearRegression(TimeSeriesData, int, int, double)
     */
    public static ForecastResult linearRegression(TimeSeriesData timeSeries, int variableIndex, 
                                                int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.linearRegression(timeSeries, variableIndex, forecastSteps, confidenceLevel);
    }
    
    /**
     * 线性回归预测（按列名） / Linear Regression Forecasting (by column name)
     * @see TimeSeriesForecasting#linearRegression(TimeSeriesData, String, int, double)
     */
    public static ForecastResult linearRegression(TimeSeriesData timeSeries, String columnName, 
                                                int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.linearRegression(timeSeries, columnName, forecastSteps, confidenceLevel);
    }
    
    /**
     * ARIMA预测 / ARIMA Forecasting
     * @see TimeSeriesForecasting#arimaForecast(TimeSeriesData, int, int, int, int, int, double)
     */
    public static ForecastResult arimaForecast(TimeSeriesData timeSeries, int variableIndex, 
                                                   int p, int d, int q, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.arimaForecast(timeSeries, variableIndex, p, d, q, forecastSteps, confidenceLevel);
    }
    
    /**
     * ARIMA预测（按列名） / ARIMA Forecasting (by column name)
     * @see TimeSeriesForecasting#arimaForecast(TimeSeriesData, String, int, int, int, int, double)
     */
    public static ForecastResult arimaForecast(TimeSeriesData timeSeries, String columnName, 
                                                   int p, int d, int q, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.arimaForecast(timeSeries, columnName, p, d, q, forecastSteps, confidenceLevel);
    }
    
    /**
     * 季节性预测 / Seasonal Forecasting
     * @see TimeSeriesForecasting#seasonalForecast(TimeSeriesData, int, int, int, double)
     */
    public static ForecastResult seasonalForecast(TimeSeriesData timeSeries, int variableIndex, 
                                                     int period, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.seasonalForecast(timeSeries, variableIndex, period, forecastSteps, confidenceLevel);
    }
    
    /**
     * 季节性预测（按列名） / Seasonal Forecasting (by column name)
     * @see TimeSeriesForecasting#seasonalForecast(TimeSeriesData, String, int, int, double)
     */
    public static ForecastResult seasonalForecast(TimeSeriesData timeSeries, String columnName, 
                                                     int period, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.seasonalForecast(timeSeries, columnName, period, forecastSteps, confidenceLevel);
    }
    
    /**
     * Holt-Winters预测 / Holt-Winters Forecasting
     * @see TimeSeriesForecasting#holtWintersForecast(TimeSeriesData, int, double, double, double, int, int, double)
     */
    public static ForecastResult holtWintersForecast(TimeSeriesData timeSeries, int variableIndex,
                                                        double alpha, double beta, double gamma, int period,
                                                        int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.holtWintersForecast(timeSeries, variableIndex, alpha, beta, gamma, period, forecastSteps, confidenceLevel);
    }
    
    /**
     * GARCH预测 / GARCH Forecasting
     * @see TimeSeriesForecasting#garchForecast(TimeSeriesData, int, int, int, int, double)
     */
    public static ForecastResult garchForecast(TimeSeriesData timeSeries, int variableIndex,
                                                      int p, int q, int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.garchForecast(timeSeries, variableIndex, p, q, forecastSteps, confidenceLevel);
    }
    
    /**
     * 状态空间模型预测 / State Space Model Forecasting
     * @see TimeSeriesForecasting#stateSpaceForecast(TimeSeriesData, int, double, double, double, int, double)
     */
    public static ForecastResult stateSpaceForecast(TimeSeriesData timeSeries, int variableIndex,
                                                           double sigmaEta, double sigmaZeta, double sigmaEpsilon,
                                                           int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.stateSpaceForecast(timeSeries, variableIndex, sigmaEta, sigmaZeta, sigmaEpsilon, forecastSteps, confidenceLevel);
    }
    
    /**
     * 自动预测 / Auto Forecasting
     * @see TimeSeriesForecasting#autoForecast(TimeSeriesData, int, int, double)
     */
    public static ForecastResult autoForecast(TimeSeriesData timeSeries, int variableIndex,
                                                     int forecastSteps, double confidenceLevel) {
        return TimeSeriesForecasting.autoForecast(timeSeries, variableIndex, forecastSteps, confidenceLevel);
    }
    
    /**
     * 自动预测（按列名） / Auto Forecasting (by column name)
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
     * @see TimeSeriesDecomposition#classicalDecomposition(TimeSeriesData, int, int, TimeSeriesDecomposition.DecompositionModel)
     */
    public static DecompositionResult classicalDecomposition(TimeSeriesData timeSeries, int variableIndex, 
                                                              int period, TimeSeriesDecomposition.DecompositionModel type) {
        return TimeSeriesDecomposition.classicalDecomposition(timeSeries, variableIndex, period, type);
    }
    
    /**
     * 经典分解（按列名） / Classical Decomposition (by column name)
     * @see TimeSeriesDecomposition#classicalDecomposition(TimeSeriesData, String, int, TimeSeriesDecomposition.DecompositionModel)
     */
    public static DecompositionResult classicalDecomposition(TimeSeriesData timeSeries, String columnName, 
                                                              int period, TimeSeriesDecomposition.DecompositionModel type) {
        return TimeSeriesDecomposition.classicalDecomposition(timeSeries, columnName, period, type);
    }
    
    /**
     * X-13分解 / X-13 Decomposition
     * @see TimeSeriesDecomposition#x13Decomposition(TimeSeriesData, int, int)
     */
    public static DecompositionResult x13Decomposition(TimeSeriesData timeSeries, int variableIndex, int period) {
        return TimeSeriesDecomposition.x13Decomposition(timeSeries, variableIndex, period);
    }
    
    /**
     * X-13分解（按列名） / X-13 Decomposition (by column name)
     * @see TimeSeriesDecomposition#x13Decomposition(TimeSeriesData, String, int)
     */
    public static DecompositionResult x13Decomposition(TimeSeriesData timeSeries, String columnName, int period) {
        return TimeSeriesDecomposition.x13Decomposition(timeSeries, columnName, period);
    }
    
    /**
     * STL分解 / STL Decomposition
     * @see TimeSeriesDecomposition#stlDecomposition(TimeSeriesData, int, int, int)
     */
    public static DecompositionResult stlDecomposition(TimeSeriesData timeSeries, int variableIndex, 
                                                        int period, int seasonalWindow, int trendWindow) {
        return TimeSeriesDecomposition.stlDecomposition(timeSeries, variableIndex, period, seasonalWindow, trendWindow);
    }
    
    /**
     * STL分解（按列名） / STL Decomposition (by column name)
     * @see TimeSeriesDecomposition#stlDecomposition(TimeSeriesData, String, int, int, int)
     */
    public static DecompositionResult stlDecomposition(TimeSeriesData timeSeries, String columnName, 
                                                        int period, int seasonalWindow, int trendWindow) {
        return TimeSeriesDecomposition.stlDecomposition(timeSeries, columnName, period, seasonalWindow, trendWindow);
    }
    
    /**
     * 小波分解 / Wavelet Decomposition
     * @see TimeSeriesDecomposition#waveletDecomposition(TimeSeriesData, int, String, int)
     */
    public static DecompositionResult waveletDecomposition(TimeSeriesData timeSeries, int variableIndex, 
                                                            String waveletType, int levels) {
        return TimeSeriesDecomposition.waveletDecomposition(timeSeries, variableIndex, waveletType, levels);
    }
    
    /**
     * 小波分解（按列名） / Wavelet Decomposition (by column name)
     * @see TimeSeriesDecomposition#waveletDecomposition(TimeSeriesData, String, String, int)
     */
    public static DecompositionResult waveletDecomposition(TimeSeriesData timeSeries, String columnName, 
                                                            String waveletType, int levels) {
        return TimeSeriesDecomposition.waveletDecomposition(timeSeries, columnName, waveletType, levels);
    }
    
    // ========== 时间序列过滤方法 / Time Series Filtering Methods ==========
    
    /**
     * 移动平均滤波 / Moving Average Filtering
     * @see TimeSeriesFiltering#movingAverage(TimeSeriesData, int, int)
     */
    public static FilterResult movingAverage(TimeSeriesData timeSeries, int variableIndex, int windowSize) {
        return TimeSeriesFiltering.movingAverage(timeSeries, variableIndex, windowSize);
    }
    
    /**
     * 移动平均滤波（按列名） / Moving Average Filtering (by column name)
     * @see TimeSeriesFiltering#movingAverage(TimeSeriesData, String, int)
     */
    public static FilterResult movingAverage(TimeSeriesData timeSeries, String columnName, int windowSize) {
        return TimeSeriesFiltering.movingAverage(timeSeries, columnName, windowSize);
    }
    
    /**
     * 指数平滑滤波 / Exponential Smoothing Filtering
     * @see TimeSeriesFiltering#exponentialSmoothing(TimeSeriesData, int, double)
     */
    public static FilterResult exponentialSmoothing(TimeSeriesData timeSeries, int variableIndex, double alpha) {
        return TimeSeriesFiltering.exponentialSmoothing(timeSeries, variableIndex, alpha);
    }
    
    /**
     * 指数平滑滤波（按列名） / Exponential Smoothing Filtering (by column name)
     * @see TimeSeriesFiltering#exponentialSmoothing(TimeSeriesData, String, double)
     */
    public static FilterResult exponentialSmoothing(TimeSeriesData timeSeries, String columnName, double alpha) {
        return TimeSeriesFiltering.exponentialSmoothing(timeSeries, columnName, alpha);
    }
    
    /**
     * 高斯滤波 / Gaussian Filtering
     * @see TimeSeriesFiltering#gaussianFilter(TimeSeriesData, int, double)
     */
    public static FilterResult gaussianFilter(TimeSeriesData timeSeries, int variableIndex, double sigma) {
        return TimeSeriesFiltering.gaussianFilter(timeSeries, variableIndex, sigma);
    }
    
    /**
     * 高斯滤波（按列名） / Gaussian Filtering (by column name)
     * @see TimeSeriesFiltering#gaussianFilter(TimeSeriesData, String, double)
     */
    public static FilterResult gaussianFilter(TimeSeriesData timeSeries, String columnName, double sigma) {
        return TimeSeriesFiltering.gaussianFilter(timeSeries, columnName, sigma);
    }
    
    /**
     * 中值滤波 / Median Filtering
     * @see TimeSeriesFiltering#medianFilter(TimeSeriesData, int, int)
     */
    public static FilterResult medianFilter(TimeSeriesData timeSeries, int variableIndex, int windowSize) {
        return TimeSeriesFiltering.medianFilter(timeSeries, variableIndex, windowSize);
    }
    
    /**
     * 中值滤波（按列名） / Median Filtering (by column name)
     * @see TimeSeriesFiltering#medianFilter(TimeSeriesData, String, int)
     */
    public static FilterResult medianFilter(TimeSeriesData timeSeries, String columnName, int windowSize) {
        return TimeSeriesFiltering.medianFilter(timeSeries, columnName, windowSize);
    }
    
    /**
     * 低通滤波 / Low-pass Filtering
     * @see TimeSeriesFiltering#lowPassFilter(TimeSeriesData, int, double, int)
     */
    public static FilterResult lowPassFilter(TimeSeriesData timeSeries, int variableIndex, 
                                                 double cutoffFreq, int order) {
        return TimeSeriesFiltering.lowPassFilter(timeSeries, variableIndex, cutoffFreq, order);
    }
    
    /**
     * 低通滤波（按列名） / Low-pass Filtering (by column name)
     * @see TimeSeriesFiltering#lowPassFilter(TimeSeriesData, String, double, int)
     */
    public static FilterResult lowPassFilter(TimeSeriesData timeSeries, String columnName, 
                                                 double cutoffFreq, int order) {
        return TimeSeriesFiltering.lowPassFilter(timeSeries, columnName, cutoffFreq, order);
    }
    
    /**
     * 高通滤波 / High-pass Filtering
     * @see TimeSeriesFiltering#highPassFilter(TimeSeriesData, int, double, int)
     */
    public static FilterResult highPassFilter(TimeSeriesData timeSeries, int variableIndex, 
                                                  double cutoffFreq, int order) {
        return TimeSeriesFiltering.highPassFilter(timeSeries, variableIndex, cutoffFreq, order);
    }
    
    /**
     * 高通滤波（按列名） / High-pass Filtering (by column name)
     * @see TimeSeriesFiltering#highPassFilter(TimeSeriesData, String, double, int)
     */
    public static FilterResult highPassFilter(TimeSeriesData timeSeries, String columnName, 
                                                  double cutoffFreq, int order) {
        return TimeSeriesFiltering.highPassFilter(timeSeries, columnName, cutoffFreq, order);
    }
    
    /**
     * 带通滤波 / Band-pass Filtering
     * @see TimeSeriesFiltering#bandPassFilter(TimeSeriesData, int, double, double, int)
     */
    public static FilterResult bandPassFilter(TimeSeriesData timeSeries, int variableIndex, 
                                                   double lowFreq, double highFreq, int order) {
        return TimeSeriesFiltering.bandPassFilter(timeSeries, variableIndex, lowFreq, highFreq, order);
    }
    
    /**
     * 带通滤波（按列名） / Band-pass Filtering (by column name)
     * @see TimeSeriesFiltering#bandPassFilter(TimeSeriesData, String, double, double, int)
     */
    public static FilterResult bandPassFilter(TimeSeriesData timeSeries, String columnName, 
                                                   double lowFreq, double highFreq, int order) {
        return TimeSeriesFiltering.bandPassFilter(timeSeries, columnName, lowFreq, highFreq, order);
    }
    
    /**
     * 自适应滤波 / Adaptive Filtering
     * @see TimeSeriesFiltering#adaptiveFilter(TimeSeriesData, int, double)
     */
    public static FilterResult adaptiveFilter(TimeSeriesData timeSeries, int variableIndex, double learningRate) {
        return TimeSeriesFiltering.adaptiveFilter(timeSeries, variableIndex, learningRate);
    }
    
    /**
     * 自适应滤波（按列名） / Adaptive Filtering (by column name)
     * @see TimeSeriesFiltering#adaptiveFilter(TimeSeriesData, String, double)
     */
    public static FilterResult adaptiveFilter(TimeSeriesData timeSeries, String columnName, double learningRate) {
        return TimeSeriesFiltering.adaptiveFilter(timeSeries, columnName, learningRate);
    }
    
    // ========== 协整分析方法 / Cointegration Analysis Methods ==========
    
    /**
     * Engle-Granger协整检验 / Engle-Granger Cointegration Test
     * @see CointegrationAnalysis#engleGrangerTest(IVector, IVector, int)
     */
    public static CointegrationAnalysis.EngleGrangerResult engleGrangerTest(IVector<Double> y, IVector<Double> x, int maxLags) {
        return CointegrationAnalysis.engleGrangerTest(y, x, maxLags);
    }
    
    /**
     * Johansen协整检验 / Johansen Cointegration Test
     * @see CointegrationAnalysis#johansenTest(IMatrix, int, TrendType)
     */
    public static CointegrationAnalysis.JohansenResult johansenTest(IMatrix<Double> data, int maxLags, CointegrationAnalysis.TrendType trendType) {
        return CointegrationAnalysis.johansenTest(data, maxLags, trendType);
    }
    
    /**
     * 估计协整关系 / Estimate Cointegrating Relationship
     * @see CointegrationAnalysis#estimateCointegratingRelationship(IVector, IVector)
     */
    public static CointegrationAnalysis.CointegratingRelationship estimateCointegratingRelationship(IVector<Double> y, IVector<Double> x) {
        return CointegrationAnalysis.estimateCointegratingRelationship(y, x);
    }
    
    /**
     * 估计误差修正模型 / Estimate Error Correction Model
     * @see CointegrationAnalysis#estimateECM(IVector, IVector, IVector, int)
     */
    public static CointegrationAnalysis.ErrorCorrectionModel estimateECM(IVector<Double> deltaY, IVector<Double> deltaX, 
                                                          IVector<Double> residuals, int maxLags) {
        return CointegrationAnalysis.estimateECM(deltaY, deltaX, residuals, maxLags);
    }
    
    // ========== 时间序列模型方法 / Time Series Model Methods ==========
    
    /**
     * 创建ARIMA模型 / Create ARIMA Model
     * @see TimeSeriesModelFactory#createARIMAModel(IVector, int, int, int)
     */
    public static ITimeSeriesModel createARIMAModel(IVector<Double> data, int p, int d, int q) {
        return TimeSeriesModelFactory.createARIMAModel(data, p, d, q);
    }
    
    /**
     * 自动选择ARIMA模型 / Auto-select ARIMA Model
     * @see TimeSeriesModelFactory#createARIMAModel(IVector, int, int, int, SelectionCriterion)
     */
    public static ITimeSeriesModel createARIMAModel(IVector<Double> data, int maxP, int maxD, int maxQ, 
                                                   TimeSeriesModelFactory.SelectionCriterion criterion) {
        return TimeSeriesModelFactory.createARIMAModel(data, maxP, maxD, maxQ, criterion);
    }
    
    /**
     * 创建指数平滑模型 / Create Exponential Smoothing Model
     * @see ExponentialSmoothingModels.SimpleExponentialSmoothing#fit(IVector, double)
     */
    public static ExponentialSmoothingModels.SimpleExponentialSmoothing createSimpleExponentialSmoothing(IVector<Double> data, double alpha) {
        return ExponentialSmoothingModels.SimpleExponentialSmoothing.fit(data, alpha);
    }
    
    /**
     * 创建双指数平滑模型 / Create Double Exponential Smoothing Model
     * @see ExponentialSmoothingModels.DoubleExponentialSmoothing#fit(IVector, double, double)
     */
    public static ExponentialSmoothingModels.DoubleExponentialSmoothing createDoubleExponentialSmoothing(IVector<Double> data, double alpha, double beta) {
        return ExponentialSmoothingModels.DoubleExponentialSmoothing.fit(data, alpha, beta);
    }
    
    /**
     * 创建Holt-Winters平滑模型 / Create Holt-Winters Smoothing Model
     * @see ExponentialSmoothingModels.HoltWintersSmoothing#fit(IVector, double, double, double, int)
     */
    public static ExponentialSmoothingModels.HoltWintersSmoothing createHoltWintersSmoothing(IVector<Double> data, double alpha, double beta, 
                                                                 double gamma, int period) {
        return ExponentialSmoothingModels.HoltWintersSmoothing.fit(data, alpha, beta, gamma, period);
    }
    
    /**
     * 创建自适应指数平滑模型 / Create Adaptive Exponential Smoothing Model
     * @see ExponentialSmoothingModels.AdaptiveExponentialSmoothing#fit(IVector, double, double)
     */
    public static ExponentialSmoothingModels.AdaptiveExponentialSmoothing createAdaptiveExponentialSmoothing(IVector<Double> data, 
                                                                      double initialAlpha, double adaptationRate) {
        return ExponentialSmoothingModels.AdaptiveExponentialSmoothing.fit(data, initialAlpha, adaptationRate);
    }
    
    /**
     * 选择最佳指数平滑模型 / Select Best Exponential Smoothing Model
     * @see ExponentialSmoothingModels.ModelSelector#selectBestModel(IVector, int)
     */
    public static ExponentialSmoothingModels.ModelSelectionResult selectBestExponentialSmoothingModel(IVector<Double> data, int maxPeriod) {
        return ExponentialSmoothingModels.ModelSelector.selectBestModel(data, maxPeriod);
    }
    
    /**
     * 创建GARCH模型 / Create GARCH Model
     * @see GARCHModel#fit(IVector, int, int)
     */
    public static GARCHModel createGARCHModel(IVector<Double> returns, int p, int q) {
        return GARCHModel.fit(returns, p, q);
    }
    
    /**
     * 自动拟合GARCH模型 / Auto-fit GARCH Model
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
     * @see VARModel#fit(IMatrix, int, String[])
     */
    public static VARModel createVARModel(IMatrix<Double> data, int p, String[] variableNames) {
        return VARModel.fit(data, p, variableNames);
    }
    
    /**
     * 自动拟合VAR模型 / Auto-fit VAR Model
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
     * @see TimeSeriesPlots#plotTimeSeries(TimeSeriesData, String)
     */
    public static IPlot plotTimeSeries(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotTimeSeries(timeSeriesData, title);
    }
    
    /**
     * 绘制趋势分析 / Plot Trend Analysis
     * @see TimeSeriesPlots#plotTrendAnalysis(TimeSeriesData, String)
     */
    public static IPlot plotTrendAnalysis(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotTrendAnalysis(timeSeriesData, title);
    }
    
    /**
     * 绘制季节性分解 / Plot Seasonal Decomposition
     * @see TimeSeriesPlots#plotSeasonalDecomposition(TimeSeriesData, int, String)
     */
    public static IPlot plotSeasonalDecomposition(TimeSeriesData timeSeriesData, int period, String title) {
        return TimeSeriesPlots.plotSeasonalDecomposition(timeSeriesData, period, title);
    }
    
    /**
     * 绘制自相关图 / Plot Autocorrelation
     * @see TimeSeriesPlots#plotAutocorrelation(TimeSeriesData, int, String)
     */
    public static IPlot plotAutocorrelation(TimeSeriesData timeSeriesData, int maxLag, String title) {
        return TimeSeriesPlots.plotAutocorrelation(timeSeriesData, maxLag, title);
    }
    
    /**
     * 绘制偏自相关图 / Plot Partial Autocorrelation
     * @see TimeSeriesPlots#plotPartialAutocorrelation(TimeSeriesData, int, String)
     */
    public static IPlot plotPartialAutocorrelation(TimeSeriesData timeSeriesData, int maxLag, String title) {
        return TimeSeriesPlots.plotPartialAutocorrelation(timeSeriesData, maxLag, title);
    }
    
    /**
     * 绘制预测结果 / Plot Forecasting
     * @see TimeSeriesPlots#plotForecasting(TimeSeriesData, int, String)
     */
    public static IPlot plotForecasting(TimeSeriesData timeSeriesData, int forecastSteps, String title) {
        return TimeSeriesPlots.plotForecasting(timeSeriesData, forecastSteps, title);
    }
    
    /**
     * 绘制时间序列统计信息 / Plot Time Series Statistics
     * @see TimeSeriesPlots#plotTimeSeriesStatistics(TimeSeriesData, String)
     */
    public static IPlot plotTimeSeriesStatistics(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotTimeSeriesStatistics(timeSeriesData, title);
    }
    
    /**
     * 绘制多变量时间序列 / Plot Multivariate Time Series
     * @see TimeSeriesPlots#plotMultivariateTimeSeries(TimeSeriesData, String)
     */
    public static IPlot plotMultivariateTimeSeries(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotMultivariateTimeSeries(timeSeriesData, title);
    }
    
    /**
     * 绘制时间序列特征 / Plot Time Series Features
     * @see TimeSeriesPlots#plotTimeSeriesFeatures(TimeSeriesData, String)
     */
    public static IPlot plotTimeSeriesFeatures(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.plotTimeSeriesFeatures(timeSeriesData, title);
    }
    
    /**
     * 创建时间序列仪表板 / Create Time Series Dashboard
     * @see TimeSeriesPlots#createTimeSeriesDashboard(TimeSeriesData, String)
     */
    public static List<IPlot> createTimeSeriesDashboard(TimeSeriesData timeSeriesData, String title) {
        return TimeSeriesPlots.createTimeSeriesDashboard(timeSeriesData, title);
    }
    
    // ========== 时间序列分析器方法 / Time Series Analyzer Methods ==========
    
    /**
     * 创建时间序列分析器 / Create Time Series Analyzer
     * @see TimeSeriesAnalyzer#TimeSeriesAnalyzer(IVector, String)
     */
    public static TimeSeriesAnalyzer createTimeSeriesAnalyzer(IVector<Double> data, String name) {
        return new TimeSeriesAnalyzer(data, name);
    }
    
    /**
     * 创建时间序列分析器 / Create Time Series Analyzer
     * @see TimeSeriesAnalyzer#TimeSeriesAnalyzer(IVector, String, LocalDateTime[])
     */
    public static TimeSeriesAnalyzer createTimeSeriesAnalyzer(IVector<Double> data, String name, LocalDateTime[] timestamps) {
        return new TimeSeriesAnalyzer(data, name, timestamps);
    }
}