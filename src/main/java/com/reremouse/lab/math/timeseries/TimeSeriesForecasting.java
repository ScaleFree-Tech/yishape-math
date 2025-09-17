package com.reremouse.lab.math.timeseries;

import com.reremouse.lab.math.timeseries.model.StateSpaceModel;
import com.reremouse.lab.math.timeseries.model.ITimeSeriesForecastResult;
import com.reremouse.lab.math.timeseries.model.ExponentialSmoothingModels;
import com.reremouse.lab.math.timeseries.model.GARCHModel;
import com.reremouse.lab.math.timeseries.model.UnifiedARIMAModel;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.util.Tuple2;

/**
 * 时间序列预测类 / Time Series Forecasting Class
 * <p>
 * 提供时间序列预测功能，包括ARIMA模型、指数平滑、线性回归预测等。
 * 使用项目现有的stats包和linalg包功能进行预测建模。
 * </p>
 * <p>
 * Provides time series forecasting functionality including ARIMA models, exponential smoothing,
 * linear regression forecasting, etc. Uses existing stats and linalg package functionality for forecasting modeling.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class TimeSeriesForecasting {
    
    /**
     * 预测结果类 / Forecasting Result Class
     */
    public static class ForecastResult {
        public final IVector<Double> forecast;
        public final IVector<Double> lowerBound;
        public final IVector<Double> upperBound;
        public final double mse;
        public final double mae;
        public final double mape;
        public final String modelType;
        public final double confidenceLevel;
        
        public ForecastResult(IVector<Double> forecast, IVector<Double> lowerBound, IVector<Double> upperBound,
                            double mse, double mae, double mape, String modelType, double confidenceLevel) {
            this.forecast = forecast;
            this.lowerBound = lowerBound;
            this.upperBound = upperBound;
            this.mse = mse;
            this.mae = mae;
            this.mape = mape;
            this.modelType = modelType;
            this.confidenceLevel = confidenceLevel;
        }
    }
    
    
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
     */
    public static ForecastResult simpleMovingAverage(TimeSeriesData timeSeries, int variableIndex, 
                                                   int windowSize, int forecastSteps, double confidenceLevel) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        int length = data.length();
        
        // 计算移动平均 / Calculate moving average
        IVector<Double> movingAvg = calculateMovingAverage(data, windowSize);
        
        // 预测 / Forecast
        IVector<Double> forecast = Linalg.zeros(forecastSteps);
        double lastValue = movingAvg.get(length - 1);
        
        for (int i = 0; i < forecastSteps; i++) {
            forecast.set(i, lastValue);
        }
        
        // 计算置信区间 / Calculate confidence intervals
        double std = data.slice(length - windowSize, length).std();
        double zScore = getZScore(confidenceLevel);
        
        IVector<Double> lowerBound = forecast.subScalar(zScore * std);
        IVector<Double> upperBound = forecast.addScalar(zScore * std);
        
        // 计算误差指标 / Calculate error metrics
        double mse = calculateMSE(data, movingAvg);
        double mae = calculateMAE(data, movingAvg);
        double mape = calculateMAPE(data, movingAvg);
        
        return new ForecastResult(forecast, lowerBound, upperBound, mse, mae, mape, "SMA", confidenceLevel);
    }
    
    /**
     * 简单移动平均预测（按列名） / Simple Moving Average Forecasting (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param windowSize 窗口大小 / Window size
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     */
    public static ForecastResult simpleMovingAverage(TimeSeriesData timeSeries, String columnName, 
                                                   int windowSize, int forecastSteps, double confidenceLevel) {
        int index = timeSeries.getVariableIndex(columnName);
        return simpleMovingAverage(timeSeries, index, windowSize, forecastSteps, confidenceLevel);
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
     */
    public static ForecastResult exponentialSmoothing(TimeSeriesData timeSeries, int variableIndex, 
                                                    double alpha, int forecastSteps, double confidenceLevel) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        int length = data.length();
        
        // 指数平滑 / Exponential smoothing
        IVector<Double> smoothed = exponentialSmoothing(data, alpha);
        
        // 预测 / Forecast
        IVector<Double> forecast = Linalg.zeros(forecastSteps);
        double lastValue = smoothed.get(length - 1);
        
        for (int i = 0; i < forecastSteps; i++) {
            forecast.set(i, lastValue);
        }
        
        // 计算置信区间 / Calculate confidence intervals
        double std = calculateForecastStd(data, smoothed, alpha);
        double zScore = getZScore(confidenceLevel);
        
        IVector<Double> lowerBound = forecast.subScalar(zScore * std);
        IVector<Double> upperBound = forecast.addScalar(zScore * std);
        
        // 计算误差指标 / Calculate error metrics
        double mse = calculateMSE(data, smoothed);
        double mae = calculateMAE(data, smoothed);
        double mape = calculateMAPE(data, smoothed);
        
        return new ForecastResult(forecast, lowerBound, upperBound, mse, mae, mape, "ES", confidenceLevel);
    }
    
    /**
     * 指数平滑预测（按列名） / Exponential Smoothing Forecasting (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param alpha 平滑参数 / Smoothing parameter
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     */
    public static ForecastResult exponentialSmoothing(TimeSeriesData timeSeries, String columnName, 
                                                    double alpha, int forecastSteps, double confidenceLevel) {
        int index = timeSeries.getVariableIndex(columnName);
        return exponentialSmoothing(timeSeries, index, alpha, forecastSteps, confidenceLevel);
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
     */
    public static ForecastResult linearRegression(TimeSeriesData timeSeries, int variableIndex, 
                                                int forecastSteps, double confidenceLevel) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        int length = data.length();
        
        // 线性回归 / Linear regression
        Tuple2<Double, Double> regression = linearRegression(data);
        double slope = regression._1;
        double intercept = regression._2;
        
        // 预测 / Forecast
        IVector<Double> forecast = Linalg.zeros(forecastSteps);
        for (int i = 0; i < forecastSteps; i++) {
            double value = slope * (length + i) + intercept;
            forecast.set(i, value);
        }
        
        // 计算置信区间 / Calculate confidence intervals
        double std = calculateRegressionStd(data, slope, intercept);
        double zScore = getZScore(confidenceLevel);
        
        IVector<Double> lowerBound = forecast.subScalar(zScore * std);
        IVector<Double> upperBound = forecast.addScalar(zScore * std);
        
        // 计算误差指标 / Calculate error metrics
        IVector<Double> fitted = Linalg.range(length).multiplyScalar(slope).addScalar(intercept);
        double mse = calculateMSE(data, fitted);
        double mae = calculateMAE(data, fitted);
        double mape = calculateMAPE(data, fitted);
        
        return new ForecastResult(forecast, lowerBound, upperBound, mse, mae, mape, "LR", confidenceLevel);
    }
    
    /**
     * 线性回归预测（按列名） / Linear Regression Forecasting (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     */
    public static ForecastResult linearRegression(TimeSeriesData timeSeries, String columnName, 
                                                int forecastSteps, double confidenceLevel) {
        int index = timeSeries.getVariableIndex(columnName);
        return linearRegression(timeSeries, index, forecastSteps, confidenceLevel);
    }
    
    /**
     * ARIMA模型预测 / ARIMA Model Forecasting
     * <p>
     * 使用统一ARIMA模型进行时间序列预测。
     * Use unified ARIMA model for time series forecasting.
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
     */
    public static ForecastResult arimaForecast(TimeSeriesData timeSeries, int variableIndex, 
                                             int p, int d, int q, int forecastSteps, double confidenceLevel) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        
        // 使用统一ARIMA模型类 / Use unified ARIMA model class
        UnifiedARIMAModel model = UnifiedARIMAModel.fit(data, p, d, q);
        
        // 预测 / Forecast
        ITimeSeriesForecastResult forecastResult = model.forecastWithConfidence(forecastSteps, confidenceLevel);
        IVector<Double> forecast = forecastResult.getForecastVector();
        IVector<Double> lowerBound = forecastResult.getLowerBoundsVector();
        IVector<Double> upperBound = forecastResult.getUpperBoundsVector();
        
        // 计算误差指标 / Calculate error metrics
        double[] errorMetrics = forecastResult.getErrorMetrics();
        double mse = errorMetrics[0];
        double mae = errorMetrics[1];
        double mape = errorMetrics[2];
        
        return new ForecastResult(forecast, lowerBound, upperBound, mse, mae, mape, "ARIMA", confidenceLevel);
    }
    
    /**
     * ARIMA模型预测（按列名） / ARIMA Model Forecasting (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param p AR阶数 / AR order
     * @param d 差分阶数 / Differencing order
     * @param q MA阶数 / MA order
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     */
    public static ForecastResult arimaForecast(TimeSeriesData timeSeries, String columnName, 
                                             int p, int d, int q, int forecastSteps, double confidenceLevel) {
        int index = timeSeries.getVariableIndex(columnName);
        return arimaForecast(timeSeries, index, p, d, q, forecastSteps, confidenceLevel);
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
     */
    public static ForecastResult seasonalForecast(TimeSeriesData timeSeries, int variableIndex, 
                                                int period, int forecastSteps, double confidenceLevel) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        int length = data.length();
        
        // 计算季节性成分 / Calculate seasonal component
        IVector<Double> seasonal = calculateSeasonalComponent(data, period);
        
        // 去季节性 / Deseasonalize
        IVector<Double> deseasonalized = data.sub(seasonal);
        
        // 对去季节性数据预测 / Forecast deseasonalized data
        ForecastResult baseForecast = linearRegression(
            new TimeSeriesData(timeSeries.getTimestamps(), deseasonalized, "deseasonalized"),
            0, forecastSteps, confidenceLevel);
        
        // 添加季节性成分 / Add seasonal component
        IVector<Double> forecast = baseForecast.forecast.add(extendSeasonal(seasonal, period, forecastSteps));
        IVector<Double> lowerBound = baseForecast.lowerBound.add(extendSeasonal(seasonal, period, forecastSteps));
        IVector<Double> upperBound = baseForecast.upperBound.add(extendSeasonal(seasonal, period, forecastSteps));
        
        return new ForecastResult(forecast, lowerBound, upperBound, 
                                baseForecast.mse, baseForecast.mae, baseForecast.mape, "Seasonal", confidenceLevel);
    }
    
    /**
     * 季节性预测（按列名） / Seasonal Forecasting (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param period 季节周期 / Seasonal period
     * @param forecastSteps 预测步数 / Forecast steps
     * @param confidenceLevel 置信水平 / Confidence level
     * @return 预测结果 / Forecast result
     */
    public static ForecastResult seasonalForecast(TimeSeriesData timeSeries, String columnName, 
                                                int period, int forecastSteps, double confidenceLevel) {
        int index = timeSeries.getVariableIndex(columnName);
        return seasonalForecast(timeSeries, index, period, forecastSteps, confidenceLevel);
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
     */
    public static ForecastResult holtWintersForecast(TimeSeriesData timeSeries, int variableIndex,
                                                   double alpha, double beta, double gamma, int period,
                                                   int forecastSteps, double confidenceLevel) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        
        // 使用Holt-Winters模型 / Use Holt-Winters model
        ExponentialSmoothingModels.HoltWintersSmoothing model = 
            ExponentialSmoothingModels.HoltWintersSmoothing.fit(data, alpha, beta, gamma, period);
        
        // 预测 / Forecast
        IVector<Double> forecast = model.forecast(forecastSteps);
        
        // 计算置信区间 / Calculate confidence intervals
        double std = Math.sqrt(model.getMse());
        double zScore = getZScore(confidenceLevel);
        
        IVector<Double> lowerBound = forecast.subScalar(zScore * std);
        IVector<Double> upperBound = forecast.addScalar(zScore * std);
        
        return new ForecastResult(forecast, lowerBound, upperBound, model.getMse(), 
                                model.getMae(), 0.0, "HoltWinters", confidenceLevel);
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
     */
    public static ForecastResult garchForecast(TimeSeriesData timeSeries, int variableIndex,
                                             int p, int q, int forecastSteps, double confidenceLevel) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        
        // 使用GARCH模型 / Use GARCH model
        GARCHModel model = GARCHModel.fit(data, p, q);
        
        // 预测 / Forecast
        GARCHModel.GARCHForecastResult garchResult = model.forecastReturns(forecastSteps, confidenceLevel);
        
        return new ForecastResult(garchResult.meanForecast, garchResult.lowerBound, garchResult.upperBound,
                                model.getVariance().get(model.getVariance().length() - 1), 0.0, 0.0, "GARCH", confidenceLevel);
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
     */
    public static ForecastResult stateSpaceForecast(TimeSeriesData timeSeries, int variableIndex,
                                                  double sigmaEta, double sigmaZeta, double sigmaEpsilon,
                                                  int forecastSteps, double confidenceLevel) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        
        // 创建状态空间模型 / Create state space model
        StateSpaceModel model = StateSpaceModel.createLocalLinearTrend(sigmaEta, sigmaZeta, sigmaEpsilon);
        
        // 运行Kalman滤波 / Run Kalman filter
        StateSpaceModel.KalmanFilterResult filterResult = model.runKalmanFilter(data);
        
        // 预测 / Forecast
        StateSpaceModel.StateSpaceForecastResult forecastResult = model.forecast(forecastSteps);
        
        // 提取预测值 / Extract forecast values
        IVector<Double> forecast = Linalg.zeros(forecastSteps);
        IVector<Double> lowerBound = Linalg.zeros(forecastSteps);
        IVector<Double> upperBound = Linalg.zeros(forecastSteps);
        
        for (int i = 0; i < forecastSteps; i++) {
            forecast.set(i, forecastResult.forecastObservations.get(i).get(0));
            double std = forecastResult.forecastStd.get(i).get(0);
            double zScore = getZScore(confidenceLevel);
            lowerBound.set(i, forecast.get(i) - zScore * std);
            upperBound.set(i, forecast.get(i) + zScore * std);
        }
        
        return new ForecastResult(forecast, lowerBound, upperBound, 0.0, 0.0, 0.0, "StateSpace", confidenceLevel);
    }
    
    /**
     * 自动模型选择预测 / Automatic Model Selection Forecasting
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
     */
    public static ForecastResult autoForecast(TimeSeriesData timeSeries, int variableIndex,
                                            int forecastSteps, double confidenceLevel) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        
        // 尝试不同的模型 / Try different models
        ForecastResult bestResult = null;
        double bestMSE = Double.POSITIVE_INFINITY;
        
        // 1. 简单指数平滑 / Simple exponential smoothing
        try {
            ExponentialSmoothingModels.SimpleExponentialSmoothing sesModel = 
                ExponentialSmoothingModels.SimpleExponentialSmoothing.fit(data, 0.3);
            IVector<Double> sesForecast = sesModel.forecast(forecastSteps);
            double std = Math.sqrt(sesModel.getMse());
            double zScore = getZScore(confidenceLevel);
            IVector<Double> lowerBound = sesForecast.subScalar(zScore * std);
            IVector<Double> upperBound = sesForecast.addScalar(zScore * std);
            
            ForecastResult sesResult = new ForecastResult(sesForecast, lowerBound, upperBound,
                                                        sesModel.getMse(), sesModel.getMae(), 0.0, "SES", confidenceLevel);
            if (sesModel.getMse() < bestMSE) {
                bestMSE = sesModel.getMse();
                bestResult = sesResult;
            }
        } catch (Exception e) {
            // 跳过 / Skip
        }
        
        // 2. ARIMA模型 / ARIMA model
        try {
            ForecastResult arimaResult = arimaForecast(timeSeries, variableIndex, 1, 1, 1, forecastSteps, confidenceLevel);
            if (arimaResult.mse < bestMSE) {
                bestMSE = arimaResult.mse;
                bestResult = arimaResult;
            }
        } catch (Exception e) {
            // 跳过 / Skip
        }
        
        // 3. 线性回归 / Linear regression
        try {
            ForecastResult lrResult = linearRegression(timeSeries, variableIndex, forecastSteps, confidenceLevel);
            if (lrResult.mse < bestMSE) {
                bestMSE = lrResult.mse;
                bestResult = lrResult;
            }
        } catch (Exception e) {
            // 跳过 / Skip
        }
        
        if (bestResult == null) {
            throw new RuntimeException("无法找到合适的预测模型");
        }
        
        return bestResult;
    }
    
    // ========== 私有辅助方法 / Private Helper Methods ==========
    
    /**
     * 计算移动平均 / Calculate moving average
     */
    private static IVector<Double> calculateMovingAverage(IVector<Double> data, int windowSize) {
        int length = data.length();
        IVector<Double> movingAvg = Linalg.zeros(length);
        
        for (int i = 0; i < length; i++) {
            int start = Math.max(0, i - windowSize + 1);
            int end = i + 1;
            
            IVector<Double> window = data.slice(start, end);
            movingAvg.set(i, window.mean());
        }
        
        return movingAvg;
    }
    
    /**
     * 指数平滑实现 / Exponential smoothing implementation
     */
    private static IVector<Double> exponentialSmoothing(IVector<Double> data, double alpha) {
        return TimeSeriesUtils.exponentialSmoothing(data, alpha);
    }
    
    /**
     * 线性回归 / Linear regression
     */
    private static Tuple2<Double, Double> linearRegression(IVector<Double> data) {
        int length = data.length();
        IVector<Double> x = Linalg.range(length);
        
        double sumX = x.sum();
        double sumY = data.sum();
        double sumXY = x.multiply(data).sum();
        double sumXX = x.multiply(x).sum();
        
        double slope = (length * sumXY - sumX * sumY) / (length * sumXX - sumX * sumX);
        double intercept = (sumY - slope * sumX) / length;
        
        return new Tuple2<>(slope, intercept);
    }
    
    /**
     * 差分 / Differencing
     */
    private static IVector<Double> difference(IVector<Double> data) {
        return TimeSeriesUtils.difference(data);
    }
    
    /**
     * 逆差分 / Inverse differencing
     */
    private static IVector<Double> inverseDifference(IVector<Double> diff, IVector<Double> original) {
        return TimeSeriesUtils.inverseDifference(original, diff, 1);
    }
    
    
    /**
     * 计算季节性成分 / Calculate seasonal component
     */
    private static IVector<Double> calculateSeasonalComponent(IVector<Double> data, int period) {
        return TimeSeriesUtils.calculateSeasonalComponent(data, period);
    }
    
    /**
     * 扩展季节性成分 / Extend seasonal component
     */
    private static IVector<Double> extendSeasonal(IVector<Double> seasonal, int period, int steps) {
        IVector<Double> extended = Linalg.zeros(steps);
        
        for (int i = 0; i < steps; i++) {
            int index = i % period;
            extended.set(i, seasonal.get(index));
        }
        
        return extended;
    }
    
    /**
     * 计算预测标准差 / Calculate forecast standard deviation
     */
    private static double calculateForecastStd(IVector<Double> data, IVector<Double> fitted, double alpha) {
        IVector<Double> residuals = data.sub(fitted);
        return residuals.std() * Math.sqrt(1 + alpha * alpha);
    }
    
    /**
     * 计算回归标准差 / Calculate regression standard deviation
     */
    private static double calculateRegressionStd(IVector<Double> data, double slope, double intercept) {
        int length = data.length();
        IVector<Double> fitted = Linalg.range(length).multiplyScalar(slope).addScalar(intercept);
        IVector<Double> residuals = data.sub(fitted);
        
        return residuals.std();
    }
    
    /**
     * 获取Z分数 / Get Z-score
     */
    private static double getZScore(double confidenceLevel) {
        if (confidenceLevel == 0.95) return 1.96;
        if (confidenceLevel == 0.90) return 1.645;
        if (confidenceLevel == 0.99) return 2.576;
        return 1.96; // 默认95%置信水平 / Default 95% confidence level
    }
    
    /**
     * 计算均方误差 / Calculate Mean Squared Error
     */
    private static double calculateMSE(IVector<Double> actual, IVector<Double> predicted) {
        IVector<Double> errors = actual.sub(predicted);
        return errors.multiply(errors).mean();
    }
    
    /**
     * 计算平均绝对误差 / Calculate Mean Absolute Error
     */
    private static double calculateMAE(IVector<Double> actual, IVector<Double> predicted) {
        IVector<Double> errors = actual.sub(predicted).apply(Math::abs);
        return errors.mean();
    }
    
    /**
     * 计算平均绝对百分比误差 / Calculate Mean Absolute Percentage Error
     */
    private static double calculateMAPE(IVector<Double> actual, IVector<Double> predicted) {
        IVector<Double> errors = actual.sub(predicted).divide(actual).apply(Math::abs);
        return errors.mean() * 100;
    }
}
