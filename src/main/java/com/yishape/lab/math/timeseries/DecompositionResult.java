/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;

/**
 *
 * @author lteb2
 */
/**
 * 分解结果类 / Decomposition Result Class
 */
public class DecompositionResult {

    public final IVector<Double> trend;
    public final IVector<Double> seasonal;
    public final IVector<Double> residual;
    public final IVector<Double> original;
    public final TimeSeriesDecomposition.DecompositionModel model;
    public final int period;
    public final double trendStrength;
    public final double seasonalStrength;
    public final double residualStrength;

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
