/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.reremouse.lab.math.timeseries;

import com.reremouse.lab.math.linalg.IVector;

/**
 *
 * @author lteb2
 */
/**
 * 预测结果类 / Forecasting Result Class
 */
public class ForecastResult {

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
