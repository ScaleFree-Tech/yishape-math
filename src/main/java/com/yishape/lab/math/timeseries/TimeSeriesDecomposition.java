package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;

/**
 * 时间序列分解类 / Time Series Decomposition Class
 * <p>
 * 提供时间序列分解功能，包括趋势、季节性、周期性成分的分离。 支持加法模型和乘法模型，使用项目现有的signal包和linalg包功能。
 * </p>
 * <p>
 * Provides time series decomposition functionality including trend, seasonal,
 * and cyclical component separation. Supports additive and multiplicative
 * models, uses existing signal and linalg package functionality.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class TimeSeriesDecomposition {

    /**
     * 分解模型类型枚举 / Decomposition Model Type Enum
     */
    public enum DecompositionModel {
        ADDITIVE, // 加法模型 / Additive model
        MULTIPLICATIVE // 乘法模型 / Multiplicative model
    }

    /**
     * 经典分解 / Classical Decomposition
     * <p>
     * 使用经典方法分解时间序列为趋势、季节性和残差成分。 Use classical method to decompose time series
     * into trend, seasonal, and residual components.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param period 季节周期 / Seasonal period
     * @param model 分解模型 / Decomposition model
     * @return 分解结果 / Decomposition result
     */
    public static DecompositionResult classicalDecomposition(TimeSeriesData timeSeries, int variableIndex,
            int period, DecompositionModel model) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        int length = data.length();

        if (period >= length) {
            throw new IllegalArgumentException("季节周期不能大于等于数据长度");
        }

        // 步骤1：计算趋势成分 / Step 1: Calculate trend component
        IVector<Double> trend = calculateTrend(data, period);

        // 步骤2：去趋势 / Step 2: Detrend
        IVector<Double> detrended = detrend(data, trend, model);

        // 步骤3：计算季节性成分 / Step 3: Calculate seasonal component
        IVector<Double> seasonal = calculateSeasonal(detrended, period);

        // 步骤4：计算残差成分 / Step 4: Calculate residual component
        IVector<Double> residual = calculateResidual(data, trend, seasonal, model);

        // 计算各成分强度 / Calculate component strengths
        double trendStrength = calculateComponentStrength(trend, data);
        double seasonalStrength = calculateComponentStrength(seasonal, data);
        double residualStrength = calculateComponentStrength(residual, data);

        return new DecompositionResult(trend, seasonal, residual, data, model, period,
                trendStrength, seasonalStrength, residualStrength);
    }

    /**
     * 经典分解（按列名） / Classical Decomposition (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param period 季节周期 / Seasonal period
     * @param model 分解模型 / Decomposition model
     * @return 分解结果 / Decomposition result
     */
    public static DecompositionResult classicalDecomposition(TimeSeriesData timeSeries, String columnName,
            int period, DecompositionModel model) {
        int index = timeSeries.getVariableIndex(columnName);
        return classicalDecomposition(timeSeries, index, period, model);
    }

    /**
     * X-13ARIMA-SEATS分解 / X-13ARIMA-SEATS Decomposition
     * <p>
     * 使用X-13ARIMA-SEATS方法进行时间序列分解（简化版本）。 Use X-13ARIMA-SEATS method for time
     * series decomposition (simplified version).
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param period 季节周期 / Seasonal period
     * @return 分解结果 / Decomposition result
     */
    public static DecompositionResult x13Decomposition(TimeSeriesData timeSeries, int variableIndex, int period) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        int length = data.length();

        if (period >= length) {
            throw new IllegalArgumentException("季节周期不能大于等于数据长度");
        }

        // 步骤1：预白化 / Step 1: Pre-whitening
        IVector<Double> prewhitened = prewhiten(data);

        // 步骤2：趋势估计 / Step 2: Trend estimation
        IVector<Double> trend = estimateTrend(prewhitened, period);

        // 步骤3：季节性估计 / Step 3: Seasonal estimation
        IVector<Double> seasonal = estimateSeasonal(prewhitened, trend, period);

        // 步骤4：残差计算 / Step 4: Residual calculation
        IVector<Double> residual = data.sub(trend).sub(seasonal);

        // 计算各成分强度 / Calculate component strengths
        double trendStrength = calculateComponentStrength(trend, data);
        double seasonalStrength = calculateComponentStrength(seasonal, data);
        double residualStrength = calculateComponentStrength(residual, data);

        return new DecompositionResult(trend, seasonal, residual, data, DecompositionModel.ADDITIVE, period,
                trendStrength, seasonalStrength, residualStrength);
    }

    /**
     * X-13ARIMA-SEATS分解（按列名） / X-13ARIMA-SEATS Decomposition (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param period 季节周期 / Seasonal period
     * @return 分解结果 / Decomposition result
     */
    public static DecompositionResult x13Decomposition(TimeSeriesData timeSeries, String columnName, int period) {
        int index = timeSeries.getVariableIndex(columnName);
        return x13Decomposition(timeSeries, index, period);
    }

    /**
     * STL分解 / STL Decomposition
     * <p>
     * 使用STL（Seasonal and Trend decomposition using Loess）方法进行分解。 Use STL
     * (Seasonal and Trend decomposition using Loess) method for decomposition.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param period 季节周期 / Seasonal period
     * @param seasonalWindow 季节性窗口 / Seasonal window
     * @param trendWindow 趋势窗口 / Trend window
     * @return 分解结果 / Decomposition result
     */
    public static DecompositionResult stlDecomposition(TimeSeriesData timeSeries, int variableIndex,
            int period, int seasonalWindow, int trendWindow) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        int length = data.length();

        if (period >= length) {
            throw new IllegalArgumentException("季节周期不能大于等于数据长度");
        }

        // 初始化 / Initialize
        IVector<Double> trend = Linalg.zeros(length);
        IVector<Double> seasonal = Linalg.zeros(length);
        IVector<Double> residual = data.copy();

        // STL迭代 / STL iteration
        for (int iter = 0; iter < 10; iter++) {
            // 步骤1：去趋势 / Step 1: Detrend
            IVector<Double> detrended = residual.sub(trend);

            // 步骤2：季节性平滑 / Step 2: Seasonal smoothing
            seasonal = smoothSeasonal(detrended, period, seasonalWindow);

            // 步骤3：去季节性 / Step 3: Deseasonalize
            IVector<Double> deseasonalized = residual.sub(seasonal);

            // 步骤4：趋势平滑 / Step 4: Trend smoothing
            trend = smoothTrend(deseasonalized, trendWindow);

            // 步骤5：更新残差 / Step 5: Update residual
            residual = data.sub(trend).sub(seasonal);
        }

        // 计算各成分强度 / Calculate component strengths
        double trendStrength = calculateComponentStrength(trend, data);
        double seasonalStrength = calculateComponentStrength(seasonal, data);
        double residualStrength = calculateComponentStrength(residual, data);

        return new DecompositionResult(trend, seasonal, residual, data, DecompositionModel.ADDITIVE, period,
                trendStrength, seasonalStrength, residualStrength);
    }

    /**
     * STL分解（按列名） / STL Decomposition (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param period 季节周期 / Seasonal period
     * @param seasonalWindow 季节性窗口 / Seasonal window
     * @param trendWindow 趋势窗口 / Trend window
     * @return 分解结果 / Decomposition result
     */
    public static DecompositionResult stlDecomposition(TimeSeriesData timeSeries, String columnName,
            int period, int seasonalWindow, int trendWindow) {
        int index = timeSeries.getVariableIndex(columnName);
        return stlDecomposition(timeSeries, index, period, seasonalWindow, trendWindow);
    }

    /**
     * 小波分解 / Wavelet Decomposition
     * <p>
     * 使用小波变换进行时间序列分解。 Use wavelet transform for time series decomposition.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param wavelet 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     * @return 分解结果 / Decomposition result
     */
    public static DecompositionResult waveletDecomposition(TimeSeriesData timeSeries, int variableIndex,
            String wavelet, int levels) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);

        // 小波分解 / Wavelet decomposition
        IMatrix<Double> coeffs = waveletDecompose(data, wavelet, levels);

        // 重构趋势成分 / Reconstruct trend component
        IVector<Double> trend = reconstructTrend(coeffs, levels);

        // 重构季节性成分 / Reconstruct seasonal component
        IVector<Double> seasonal = reconstructSeasonal(coeffs, levels);

        // 计算残差成分 / Calculate residual component
        IVector<Double> residual = data.sub(trend).sub(seasonal);

        // 计算各成分强度 / Calculate component strengths
        double trendStrength = calculateComponentStrength(trend, data);
        double seasonalStrength = calculateComponentStrength(seasonal, data);
        double residualStrength = calculateComponentStrength(residual, data);

        return new DecompositionResult(trend, seasonal, residual, data, DecompositionModel.ADDITIVE, 0,
                trendStrength, seasonalStrength, residualStrength);
    }

    /**
     * 小波分解（按列名） / Wavelet Decomposition (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param wavelet 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     * @return 分解结果 / Decomposition result
     */
    public static DecompositionResult waveletDecomposition(TimeSeriesData timeSeries, String columnName,
            String wavelet, int levels) {
        int index = timeSeries.getVariableIndex(columnName);
        return waveletDecomposition(timeSeries, index, wavelet, levels);
    }

    // ========== 私有辅助方法 / Private Helper Methods ==========
    /**
     * 计算趋势成分 / Calculate trend component
     * <p>
     * 使用移动平均法计算时间序列的趋势成分。
     * Calculate trend component of time series using moving average method.
     * </p>
     *
     * @param data 输入数据 / Input data
     * @param period 季节周期（用于确定窗口大小）/ Seasonal period (used to determine window size)
     * @return 趋势成分序列 / Trend component series
     */
    private static IVector<Double> calculateTrend(IVector<Double> data, int period) {
        int length = data.length();
        IVector<Double> trend = Linalg.zeros(length);

        // 使用移动平均计算趋势 / Use moving average to calculate trend
        int windowSize = Math.max(period, 3);

        for (int i = 0; i < length; i++) {
            int start = Math.max(0, i - windowSize / 2);
            int end = Math.min(length, i + windowSize / 2 + 1);

            IVector<Double> window = data.slice(start, end);
            trend.set(i, window.mean());
        }

        return trend;
    }

    /**
     * 去趋势 / Detrend
     * <p>
     * 根据分解模型去除时间序列的趋势成分。
     * Remove trend component from time series according to decomposition model.
     * </p>
     *
     * @param data 输入数据 / Input data
     * @param trend 趋势成分 / Trend component
     * @param model 分解模型（加法或乘法）/ Decomposition model (additive or multiplicative)
     * @return 去趋势后的序列 / Detrended series
     */
    private static IVector<Double> detrend(IVector<Double> data, IVector<Double> trend, DecompositionModel model) {
        if (model == DecompositionModel.ADDITIVE) {
            return data.sub(trend);
        } else {
            return data.divide(trend);
        }
    }

    /**
     * 计算季节性成分 / Calculate seasonal component
     * <p>
     * 计算去趋势后序列的季节性成分，并进行中心化处理。
     * Calculate seasonal component of detrended series and apply centering.
     * </p>
     *
     * @param detrended 去趋势后的序列 / Detrended series
     * @param period 季节周期 / Seasonal period
     * @return 季节性成分序列 / Seasonal component series
     */
    private static IVector<Double> calculateSeasonal(IVector<Double> detrended, int period) {
        int length = detrended.length();
        IVector<Double> seasonal = Linalg.zeros(length);

        // 计算每个位置的平均值 / Calculate average for each position
        for (int i = 0; i < length; i++) {
            int seasonalIndex = i % period;
            double seasonalValue = 0.0;
            int count = 0;

            for (int j = seasonalIndex; j < length; j += period) {
                seasonalValue += detrended.get(j);
                count++;
            }

            if (count > 0) {
                seasonal.set(i, seasonalValue / count);
            }
        }

        // 中心化 / Center
        double mean = seasonal.mean();
        seasonal = seasonal.subScalar(mean);

        return seasonal;
    }

    /**
     * 计算残差成分 / Calculate residual component
     * <p>
     * 根据分解模型计算时间序列的残差成分。
     * Calculate residual component of time series according to decomposition model.
     * </p>
     *
     * @param data 原始数据 / Original data
     * @param trend 趋势成分 / Trend component
     * @param seasonal 季节性成分 / Seasonal component
     * @param model 分解模型（加法或乘法）/ Decomposition model (additive or multiplicative)
     * @return 残差成分序列 / Residual component series
     */
    private static IVector<Double> calculateResidual(IVector<Double> data, IVector<Double> trend,
            IVector<Double> seasonal, DecompositionModel model) {
        if (model == DecompositionModel.ADDITIVE) {
            return data.sub(trend).sub(seasonal);
        } else {
            return data.divide(trend).divide(seasonal);
        }
    }

    /**
     * 计算成分强度 / Calculate component strength
     * <p>
     * 计算各成分（趋势、季节性、残差）的强度，值越大说明成分越明显。
     * Calculate strength of each component (trend, seasonal, residual), higher value indicates more prominent component.
     * </p>
     *
     * @param component 成分序列 / Component series
     * @param original 原始序列 / Original series
     * @return 成分强度（0到1之间）/ Component strength (between 0 and 1)
     */
    private static double calculateComponentStrength(IVector<Double> component, IVector<Double> original) {
        double componentVar = component.var();
        double originalVar = original.var();

        if (originalVar == 0) {
            return 0.0;
        }

        return Math.max(0, 1 - componentVar / originalVar);
    }

    /**
     * 预白化 / Pre-whitening
     * <p>
     * 对时间序列进行预白化处理（标准化）。
     * Apply pre-whitening (standardization) to time series.
     * </p>
     *
     * @param data 输入数据 / Input data
     * @return 预白化后的序列 / Pre-whitened series
     */
    private static IVector<Double> prewhiten(IVector<Double> data) {
        // 简化的预白化实现 / Simplified pre-whitening implementation
        double mean = data.mean();
        double std = data.std();

        if (std == 0) {
            return data;
        }

        return data.subScalar(mean).divideByScalar(std);
    }

    /**
     * 估计趋势 / Estimate trend
     * <p>
     * 使用移动平均法估计时间序列的趋势。
     * Estimate trend of time series using moving average method.
     * </p>
     *
     * @param data 输入数据 / Input data
     * @param period 季节周期 / Seasonal period
     * @return 趋势估计序列 / Estimated trend series
     */
    private static IVector<Double> estimateTrend(IVector<Double> data, int period) {
        return calculateTrend(data, period);
    }

    /**
     * 估计季节性 / Estimate seasonal
     * <p>
     * 估计时间序列的季节性成分。
     * Estimate seasonal component of time series.
     * </p>
     *
     * @param data 输入数据 / Input data
     * @param trend 趋势成分 / Trend component
     * @param period 季节周期 / Seasonal period
     * @return 季节性估计序列 / Estimated seasonal series
     */
    private static IVector<Double> estimateSeasonal(IVector<Double> data, IVector<Double> trend, int period) {
        IVector<Double> detrended = data.sub(trend);
        return calculateSeasonal(detrended, period);
    }

    /**
     * 季节性平滑 / Seasonal smoothing
     * <p>
     * 对季节性成分进行平滑处理。
     * Apply smoothing to seasonal component.
     * </p>
     *
     * @param data 输入数据 / Input data
     * @param period 季节周期 / Seasonal period
     * @param window 平滑窗口 / Smoothing window
     * @return 平滑后的季节性序列 / Smoothed seasonal series
     */
    private static IVector<Double> smoothSeasonal(IVector<Double> data, int period, int window) {
        int length = data.length();
        IVector<Double> smoothed = Linalg.zeros(length);

        for (int i = 0; i < length; i++) {
            int seasonalIndex = i % period;
            double value = 0.0;
            int count = 0;

            for (int j = seasonalIndex; j < length; j += period) {
                int start = Math.max(0, j - window / 2);
                int end = Math.min(length, j + window / 2 + 1);

                IVector<Double> ww = data.slice(start, end);
                value += ww.mean();
                count++;
            }

            if (count > 0) {
                smoothed.set(i, value / count);
            }
        }

        return smoothed;
    }

    /**
     * 趋势平滑 / Trend smoothing
     * <p>
     * 对趋势成分进行平滑处理。
     * Apply smoothing to trend component.
     * </p>
     *
     * @param data 输入数据 / Input data
     * @param window 平滑窗口 / Smoothing window
     * @return 平滑后的趋势序列 / Smoothed trend series
     */
    private static IVector<Double> smoothTrend(IVector<Double> data, int window) {
        int length = data.length();
        IVector<Double> smoothed = Linalg.zeros(length);

        for (int i = 0; i < length; i++) {
            int start = Math.max(0, i - window / 2);
            int end = Math.min(length, i + window / 2 + 1);

            IVector<Double> ww = data.slice(start, end);
            smoothed.set(i, ww.mean());
        }

        return smoothed;
    }

    /**
     * 小波分解 / Wavelet decomposition
     * <p>
     * 使用小波变换对时间序列进行分解。
     * Decompose time series using wavelet transform.
     * </p>
     *
     * @param data 输入数据 / Input data
     * @param wavelet 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     * @return 小波系数矩阵（每行代表一层）/ Wavelet coefficients matrix (each row represents one level)
     */
    private static IMatrix<Double> waveletDecompose(IVector<Double> data, String wavelet, int levels) {
        // 简化的小波分解实现 / Simplified wavelet decomposition implementation
        int length = data.length();
        IMatrix<Double> coeffs = Linalg.zeros(levels + 1, length);

        // 设置近似系数 / Set approximation coefficients
        coeffs.setRow(0, data);

        // 计算细节系数 / Calculate detail coefficients
        for (int level = 1; level <= levels; level++) {
            int step = 1 << level;
            IVector<Double> detail = Linalg.zeros(length);

            for (int i = 0; i < length - step; i += step) {
                double value = (data.get(i + step) - data.get(i)) / 2.0;
                detail.set(i, value);
            }

            coeffs.setRow(level, detail);
        }

        return coeffs;
    }

    /**
     * 重构趋势成分 / Reconstruct trend component
     * <p>
     * 从小波分解系数中重构趋势成分（近似系数）。
     * Reconstruct trend component (approximation coefficients) from wavelet decomposition coefficients.
     * </p>
     *
     * @param coeffs 小波系数矩阵 / Wavelet coefficients matrix
     * @param levels 分解层数 / Decomposition levels
     * @return 趋势成分序列 / Trend component series
     */
    private static IVector<Double> reconstructTrend(IMatrix<Double> coeffs, int levels) {
        return coeffs.getRow(0);
    }

    /**
     * 重构季节性成分 / Reconstruct seasonal component
     * <p>
     * 从小波分解系数中重构季节性成分（高频成分之和）。
     * Reconstruct seasonal component (sum of high frequency components) from wavelet decomposition coefficients.
     * </p>
     *
     * @param coeffs 小波系数矩阵 / Wavelet coefficients matrix
     * @param levels 分解层数 / Decomposition levels
     * @return 季节性成分序列 / Seasonal component series
     */
    private static IVector<Double> reconstructSeasonal(IMatrix<Double> coeffs, int levels) {
        int length = coeffs.getColNum();
        IVector<Double> seasonal = Linalg.zeros(length);

        // 重构高频成分作为季节性 / Reconstruct high frequency components as seasonal
        for (int level = 1; level <= levels; level++) {
            IVector<Double> detail = coeffs.getRow(level);
            seasonal = seasonal.add(detail);
        }

        return seasonal;
    }
}
