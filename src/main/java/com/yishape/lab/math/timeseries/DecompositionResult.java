package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;

/**
 * 分解结果类 / Decomposition Result Class
 * <p>
 * 存储时间序列分解的结果，包括趋势分量、季节分量和残差分量。
 * 用于TimeSeriesDecomposition类返回分解后的各个成分。
 * </p>
 * <p>
 * Stores the results of time series decomposition, including trend component,
 * seasonal component, and residual component. Used by TimeSeriesDecomposition
 * class to return decomposed components.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class DecompositionResult {

    /** 趋势分量 / Trend component */
    public final IVector<Double> trend;
    /** 季节分量 / Seasonal component */
    public final IVector<Double> seasonal;
    /** 残差分量 / Residual component */
    public final IVector<Double> residual;
    /** 原始序列 / Original series */
    public final IVector<Double> original;
    /** 分解模型类型 / Decomposition model type */
    public final TimeSeriesDecomposition.DecompositionModel model;
    /** 季节周期长度 / Seasonal period length */
    public final int period;
    /** 趋势强度 / Trend strength */
    public final double trendStrength;
    /** 季节强度 / Seasonal strength */
    public final double seasonalStrength;
    /** 残差强度 / Residual strength */
    public final double residualStrength;

    /**
     * 构造函数 / Constructor
     *
     * @param trend 趋势分量 / Trend component
     * @param seasonal 季节分量 / Seasonal component
     * @param residual 残差分量 / Residual component
     * @param original 原始序列 / Original series
     * @param model 分解模型类型 / Decomposition model type
     * @param period 季节周期长度 / Seasonal period length
     * @param trendStrength 趋势强度 / Trend strength
     * @param seasonalStrength 季节强度 / Seasonal strength
     * @param residualStrength 残差强度 / Residual strength
     */
    public DecompositionResult(IVector<Double> trend, IVector<Double> seasonal, IVector<Double> residual,
            IVector<Double> original, TimeSeriesDecomposition.DecompositionModel model, int period,
            double trendStrength, double seasonalStrength, double residualStrength) {
        this.trend = trend;
        this.seasonal = seasonal;
        this.residual = residual;
        this.original = original;
        this.model = model;
        this.period = period;
        this.trendStrength = trendStrength;
        this.seasonalStrength = seasonalStrength;
        this.residualStrength = residualStrength;
    }
}
