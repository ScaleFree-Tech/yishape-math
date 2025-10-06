package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 时间序列工具类 / Time Series Utilities Class
 * <p>
 * 提供时间序列分析中常用的计算功能，包括统计量计算、自相关分析、
 * 趋势分析、预测方法等。整合了分散在各个类中的重复功能。
 * </p>
 * <p>
 * Provides common computational functions for time series analysis, including
 * statistical calculations, autocorrelation analysis, trend analysis, forecasting
 * methods, etc. Integrates scattered duplicate functionality from various classes.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class TimeSeriesUtils {

    // ========== 统计量计算 / Statistical Calculations ==========

    /**
     * 计算偏度 / Calculate skewness
     * <p>
     * 计算时间序列的偏度，衡量分布的对称性。
     * Calculate skewness of time series, measuring distribution symmetry.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @return 偏度值 / Skewness value
     */
    public static double calculateSkewness(IVector<Double> data) {
        double mean = data.mean();
        double std = data.std();
        if (std == 0) return 0.0;
        
        double sum = 0.0;
        for (int i = 0; i < data.length(); i++) {
            double normalized = (data.get(i) - mean) / std;
            sum += normalized * normalized * normalized;
        }
        
        return sum / data.length();
    }

    /**
     * 计算峰度 / Calculate kurtosis
     * <p>
     * 计算时间序列的峰度，衡量分布的尖锐程度。
     * Calculate kurtosis of time series, measuring distribution sharpness.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @return 峰度值（超额峰度） / Kurtosis value (excess kurtosis)
     */
    public static double calculateKurtosis(IVector<Double> data) {
        double mean = data.mean();
        double std = data.std();
        if (std == 0) return 0.0;
        
        double sum = 0.0;
        for (int i = 0; i < data.length(); i++) {
            double normalized = (data.get(i) - mean) / std;
            sum += normalized * normalized * normalized * normalized;
        }
        
        return sum / data.length() - 3.0; // 减去3得到超额峰度 / Subtract 3 to get excess kurtosis
    }

    /**
     * 计算变异系数 / Calculate coefficient of variation
     * <p>
     * 计算标准差与均值的比值，衡量相对变异性。
     * Calculate ratio of standard deviation to mean, measuring relative variability.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @return 变异系数 / Coefficient of variation
     */
    public static double calculateCoefficientOfVariation(IVector<Double> data) {
        double mean = data.mean();
        if (mean == 0) return Double.NaN;
        return data.std() / Math.abs(mean);
    }

    // ========== 自相关分析 / Autocorrelation Analysis ==========

    /**
     * 计算自相关函数 / Calculate autocorrelation function
     * <p>
     * 计算时间序列的自相关函数，用于分析序列的依赖关系。
     * Calculate autocorrelation function of time series for analyzing dependencies.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param maxLag 最大滞后阶数 / Maximum lag order
     * @return 自相关函数值 / Autocorrelation function values
     */
    public static IVector<Double> calculateAutocorrelation(IVector<Double> data, int maxLag) {
        int n = data.length();
        IVector<Double> autocorr = Linalg.zeros(maxLag + 1);
        
        // 计算均值 / Calculate mean
        double mean = data.mean();
        
        // 计算方差 / Calculate variance
        double variance = 0.0;
        for (int i = 0; i < n; i++) {
            double diff = data.get(i) - mean;
            variance += diff * diff;
        }
        variance /= n;
        
        if (variance == 0) {
            // 如果方差为0，返回单位自相关 / If variance is 0, return unit autocorrelation
            for (int lag = 0; lag <= maxLag; lag++) {
                autocorr.set(lag, lag == 0 ? 1.0 : 0.0);
            }
            return autocorr;
        }
        
        // 计算自相关 / Calculate autocorrelation
        for (int lag = 0; lag <= maxLag; lag++) {
            double sum = 0.0;
            for (int i = 0; i < n - lag; i++) {
                sum += (data.get(i) - mean) * (data.get(i + lag) - mean);
            }
            autocorr.set(lag, sum / (n - lag) / variance);
        }
        
        return autocorr;
    }

    /**
     * 计算偏自相关函数 / Calculate partial autocorrelation function
     * <p>
     * 计算时间序列的偏自相关函数，用于AR模型阶数选择。
     * Calculate partial autocorrelation function for AR model order selection.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param maxLag 最大滞后阶数 / Maximum lag order
     * @return 偏自相关函数值 / Partial autocorrelation function values
     */
    public static IVector<Double> calculatePartialAutocorrelation(IVector<Double> data, int maxLag) {
        IVector<Double> pacf = Linalg.zeros(maxLag + 1);
        pacf.set(0, 1.0); // 滞后0的偏自相关为1 / Partial autocorrelation at lag 0 is 1
        
        if (maxLag == 0) return pacf;
        
        // 使用Yule-Walker方程计算偏自相关 / Use Yule-Walker equations for partial autocorrelation
        IVector<Double> autocorr = calculateAutocorrelation(data, maxLag);
        
        for (int lag = 1; lag <= maxLag; lag++) {
            // 简化的偏自相关计算 / Simplified partial autocorrelation calculation
            // 实际应用中应使用Durbin-Levinson算法 / In practice, Durbin-Levinson algorithm should be used
            double pacfValue = autocorr.get(lag);
            
            // 简单的衰减模型 / Simple decay model
            if (lag > 1) {
                pacfValue *= Math.exp(-(lag - 1) * 0.1);
            }
            
            pacf.set(lag, pacfValue);
        }
        
        return pacf;
    }

    // ========== 趋势分析 / Trend Analysis ==========

    /**
     * 趋势分析结果类 / Trend Analysis Result Class
     */
    public static class TrendResult {
        public final double slope;
        public final double intercept;
        public final double rSquared;
        public final IVector<Double> trend;
        public final IVector<Double> detrended;
        
        public TrendResult(double slope, double intercept, double rSquared, 
                          IVector<Double> trend, IVector<Double> detrended) {
            this.slope = slope;
            this.intercept = intercept;
            this.rSquared = rSquared;
            this.trend = trend;
            this.detrended = detrended;
        }
    }

    /**
     * 分析时间序列趋势 / Analyze time series trend
     * <p>
     * 使用线性回归分析时间序列的趋势成分。
     * Use linear regression to analyze trend component of time series.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @return 趋势分析结果 / Trend analysis result
     */
    public static TrendResult analyzeTrend(IVector<Double> data) {
        int n = data.length();
        IVector<Double> x = Linalg.range(n);
        
        // 线性回归 / Linear regression
        double sumX = x.sum();
        double sumY = data.sum();
        double sumXY = x.multiply(data).sum();
        double sumXX = x.multiply(x).sum();
        
        double slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        double intercept = (sumY - slope * sumX) / n;
        
        // 计算趋势线 / Calculate trend line
        IVector<Double> trend = x.multiplyScalar(slope).addScalar(intercept);
        
        // 计算去趋势数据 / Calculate detrended data
        IVector<Double> detrended = data.sub(trend);
        
        // 计算R² / Calculate R-squared
        double ssRes = detrended.multiply(detrended).sum();
        double ssTot = data.subScalar(data.mean()).multiply(data.subScalar(data.mean())).sum();
        double rSquared = 1.0 - (ssRes / ssTot);
        
        return new TrendResult(slope, intercept, rSquared, trend, detrended);
    }

    /**
     * 检测趋势强度 / Detect trend strength
     * <p>
     * 检测时间序列中趋势的强度。
     * Detect strength of trend in time series.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @return 趋势强度（0-1之间） / Trend strength (between 0 and 1)
     */
    public static double detectTrendStrength(IVector<Double> data) {
        TrendResult trendResult = analyzeTrend(data);
        return Math.abs(trendResult.rSquared);
    }

    // ========== 季节性分析 / Seasonal Analysis ==========

    /**
     * 计算季节性成分 / Calculate seasonal component
     * <p>
     * 计算时间序列的季节性成分。
     * Calculate seasonal component of time series.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param period 季节周期 / Seasonal period
     * @return 季节性成分 / Seasonal component
     */
    public static IVector<Double> calculateSeasonalComponent(IVector<Double> data, int period) {
        int length = data.length();
        IVector<Double> seasonal = Linalg.zeros(length);
        
        for (int i = 0; i < length; i++) {
            int seasonalIndex = i % period;
            double seasonalValue = 0.0;
            int count = 0;
            
            for (int j = seasonalIndex; j < length; j += period) {
                seasonalValue += data.get(j);
                count++;
            }
            
            if (count > 0) {
                seasonal.set(i, seasonalValue / count);
            }
        }
        
        return seasonal;
    }

    /**
     * 检测季节性强度 / Detect seasonal strength
     * <p>
     * 检测时间序列中季节性的强度。
     * Detect strength of seasonality in time series.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param period 季节周期 / Seasonal period
     * @return 季节性强度（0-1之间） / Seasonal strength (between 0 and 1)
     */
    public static double detectSeasonalStrength(IVector<Double> data, int period) {
        IVector<Double> seasonal = calculateSeasonalComponent(data, period);
        double seasonalVar = seasonal.var();
        double dataVar = data.var();
        
        if (dataVar == 0) return 0.0;
        return Math.min(1.0, seasonalVar / dataVar);
    }

    // ========== 预测方法 / Forecasting Methods ==========

    /**
     * 移动平均预测 / Moving average forecast
     * <p>
     * 使用移动平均方法进行时间序列预测。
     * Use moving average method for time series forecasting.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param steps 预测步数 / Forecast steps
     * @param windowSize 窗口大小 / Window size
     * @return 预测值 / Forecast values
     */
    public static IVector<Double> movingAverageForecast(IVector<Double> data, int steps, int windowSize) {
        IVector<Double> forecast = Linalg.zeros(steps);
        double lastValue = data.slice(data.length() - windowSize, data.length()).mean();
        
        for (int i = 0; i < steps; i++) {
            forecast.set(i, lastValue);
        }
        
        return forecast;
    }

    /**
     * 指数平滑预测 / Exponential smoothing forecast
     * <p>
     * 使用指数平滑方法进行时间序列预测。
     * Use exponential smoothing method for time series forecasting.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param steps 预测步数 / Forecast steps
     * @param alpha 平滑参数 / Smoothing parameter
     * @return 预测值 / Forecast values
     */
    public static IVector<Double> exponentialSmoothingForecast(IVector<Double> data, int steps, double alpha) {
        // 计算指数平滑值 / Calculate exponential smoothing values
        IVector<Double> smoothed = exponentialSmoothing(data, alpha);
        
        // 预测 / Forecast
        IVector<Double> forecast = Linalg.zeros(steps);
        double lastValue = smoothed.get(smoothed.length() - 1);
        
        for (int i = 0; i < steps; i++) {
            forecast.set(i, lastValue);
        }
        
        return forecast;
    }

    /**
     * 指数平滑 / Exponential smoothing
     * <p>
     * 对时间序列进行指数平滑处理。
     * Apply exponential smoothing to time series.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param alpha 平滑参数 / Smoothing parameter
     * @return 平滑后的数据 / Smoothed data
     */
    public static IVector<Double> exponentialSmoothing(IVector<Double> data, double alpha) {
        int length = data.length();
        IVector<Double> smoothed = Linalg.zeros(length);
        
        smoothed.set(0, data.get(0));
        
        for (int i = 1; i < length; i++) {
            double value = alpha * data.get(i) + (1 - alpha) * smoothed.get(i - 1);
            smoothed.set(i, value);
        }
        
        return smoothed;
    }

    // ========== 差分和变换 / Differencing and Transformations ==========

    /**
     * 差分 / Differencing
     * <p>
     * 对时间序列进行差分处理，用于平稳化。
     * Apply differencing to time series for stationarization.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @param order 差分阶数 / Differencing order
     * @return 差分后的数据 / Differenced data
     */
    public static IVector<Double> difference(IVector<Double> data, int order) {
        IVector<Double> diff = data;
        for (int i = 0; i < order; i++) {
            int length = diff.length();
            if (length < 2) return Linalg.zeros(0);

            IVector<Double> newDiff = Linalg.zeros(length - 1);
            for (int j = 1; j < length; j++) {
                newDiff.set(j - 1, diff.get(j) - diff.get(j - 1));
            }
            diff = newDiff;
        }
        return diff;
    }

    /**
     * 一阶差分 / First-order differencing
     * <p>
     * 对时间序列进行一阶差分处理。
     * Apply first-order differencing to time series.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @return 一阶差分后的数据 / First-order differenced data
     */
    public static IVector<Double> difference(IVector<Double> data) {
        return difference(data, 1);
    }

    /**
     * 逆差分 / Inverse differencing
     * <p>
     * 将差分后的数据还原到原始尺度。
     * Restore differenced data to original scale.
     * </p>
     *
     * @param originalData 原始数据 / Original data
     * @param differencedData 差分后的数据 / Differenced data
     * @param order 差分阶数 / Differencing order
     * @return 还原后的数据 / Restored data
     */
    public static IVector<Double> inverseDifference(IVector<Double> originalData, IVector<Double> differencedData, int order) {
        if (order == 0) {
            return differencedData;
        }

        IVector<Double> currentData = originalData;
        for (int k = 0; k < order - 1; k++) {
            currentData = difference(currentData);
        }

        IVector<Double> inverseDiff = Linalg.zeros(differencedData.length());
        double lastOriginalValue = currentData.get(currentData.length() - 1);

        for (int i = 0; i < differencedData.length(); i++) {
            double newValue = lastOriginalValue + differencedData.get(i);
            inverseDiff.set(i, newValue);
            lastOriginalValue = newValue;
        }

        return inverseDiff;
    }

    // ========== 平稳性检验 / Stationarity Tests ==========

    /**
     * 检查平稳性 / Check stationarity
     * <p>
     * 使用简化的方法检查时间序列的平稳性。
     * Use simplified method to check stationarity of time series.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @return 是否平稳 / Whether stationary
     */
    public static boolean checkStationarity(IVector<Double> data) {
        // 简化的平稳性检验 / Simplified stationarity test
        // 实际应用中应使用ADF检验等 / In practice, ADF test should be used
        
        int n = data.length();
        int half = n / 2;
        
        // 比较前半部分和后半部分的均值 / Compare means of first and second half
        double mean1 = data.slice(0, half).mean();
        double mean2 = data.slice(half, n).mean();
        
        // 比较前半部分和后半部分的方差 / Compare variances of first and second half
        double var1 = data.slice(0, half).var();
        double var2 = data.slice(half, n).var();
        
        // 简化的判断标准 / Simplified criteria
        double meanDiff = Math.abs(mean1 - mean2) / Math.abs(mean1 + mean2);
        double varDiff = Math.abs(var1 - var2) / (var1 + var2);
        
        return meanDiff < 0.1 && varDiff < 0.5; // 简化的阈值 / Simplified thresholds
    }

    // ========== 数据预处理 / Data Preprocessing ==========

    /**
     * 标准化 / Standardization
     * <p>
     * 对时间序列进行标准化处理。
     * Apply standardization to time series.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @return 标准化后的数据 / Standardized data
     */
    public static IVector<Double> standardize(IVector<Double> data) {
        double mean = data.mean();
        double std = data.std();
        
        if (std == 0) return Linalg.zeros(data.length());
        
        return data.subScalar(mean).multiplyScalar(1.0 / std);
    }

    /**
     * 归一化 / Normalization
     * <p>
     * 对时间序列进行归一化处理。
     * Apply normalization to time series.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @return 归一化后的数据 / Normalized data
     */
    public static IVector<Double> normalize(IVector<Double> data) {
        double min = data.min();
        double max = data.max();
        
        if (max == min) return Linalg.zeros(data.length());
        
        return data.subScalar(min).multiplyScalar(1.0 / (max - min));
    }

    /**
     * 对数变换 / Log transformation
     * <p>
     * 对时间序列进行对数变换。
     * Apply log transformation to time series.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @return 对数变换后的数据 / Log-transformed data
     */
    public static IVector<Double> logTransform(IVector<Double> data) {
        IVector<Double> transformed = Linalg.zeros(data.length());
        
        for (int i = 0; i < data.length(); i++) {
            double value = data.get(i);
            if (value > 0) {
                transformed.set(i, Math.log(value));
            } else {
                transformed.set(i, Double.NaN);
            }
        }
        
        return transformed;
    }

    // ========== 时间序列特征 / Time Series Features ==========

    /**
     * 计算时间序列特征 / Calculate time series features
     * <p>
     * 计算时间序列的多种特征用于机器学习。
     * Calculate various features of time series for machine learning.
     * </p>
     *
     * @param data 时间序列数据 / Time series data
     * @return 特征向量 / Feature vector
     */
    public static IVector<Double> calculateFeatures(IVector<Double> data) {
        double[] features = {
            data.mean(),                    // 均值 / Mean
            data.std(),                     // 标准差 / Standard deviation
            data.min(),                     // 最小值 / Minimum
            data.max(),                     // 最大值 / Maximum
            calculateSkewness(data),        // 偏度 / Skewness
            calculateKurtosis(data),        // 峰度 / Kurtosis
            calculateCoefficientOfVariation(data), // 变异系数 / Coefficient of variation
            detectTrendStrength(data),      // 趋势强度 / Trend strength
            calculateAutocorrelation(data, 1).get(1), // 一阶自相关 / First-order autocorrelation
            data.max() - data.min()         // 极差 / Range
        };
        
        return Linalg.vector(features);
    }

    /**
     * 获取特征名称 / Get feature names
     * <p>
     * 获取时间序列特征的名称列表。
     * Get list of time series feature names.
     * </p>
     *
     * @return 特征名称数组 / Feature names array
     */
    public static String[] getFeatureNames() {
        return new String[]{
            "Mean", "Std", "Min", "Max", "Skewness", "Kurtosis",
            "CV", "TrendStrength", "ACF1", "Range"
        };
    }
}
