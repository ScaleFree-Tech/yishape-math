package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;

/**
 * 预测结果类 / Forecasting Result Class
 * <p>
 * 存储时间序列预测的结果，包括预测值、置信区间和误差指标。
 * 用于TimeSeriesForecasting类返回预测分析的完整结果。
 * </p>
 * <p>
 * Stores the results of time series forecasting, including forecast values,
 * confidence intervals, and error metrics. Used by TimeSeriesForecasting class
 * to return complete results of forecast analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ForecastResult {

    /** 预测值 / Forecast values */
    public final IVector<Double> forecast;
    /** 预测下界 / Forecast lower bound */
    public final IVector<Double> lowerBound;
    /** 预测上界 / Forecast upper bound */
    public final IVector<Double> upperBound;
    /** 均方误差 / Mean squared error */
    public final double mse;
    /** 平均绝对误差 / Mean absolute error */
    public final double mae;
    /** 平均绝对百分比误差 / Mean absolute percentage error */
    public final double mape;
    /** 模型类型 / Model type */
    public final String modelType;
    /** 置信水平 / Confidence level */
    public final double confidenceLevel;

    /**
     * 构造函数 / Constructor
     *
     * @param forecast 预测值 / Forecast values
     * @param lowerBound 预测下界 / Forecast lower bound
     * @param upperBound 预测上界 / Forecast upper bound
     * @param mse 均方误差 / Mean squared error
     * @param mae 平均绝对误差 / Mean absolute error
     * @param mape 平均绝对百分比误差 / Mean absolute percentage error
     * @param modelType 模型类型 / Model type
     * @param confidenceLevel 置信水平 / Confidence level
     */
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
