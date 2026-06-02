package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;

/**
 * 时间序列滤波门面 / Time series filtering facade.
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class FilterWrapper {

    private static final double DEFAULT_SR = 1.0;

    // --- movingAverage ---

    public FilterResult movingAverage(IVector<Double> data, int windowSize) {
        return TimeSeriesFiltering.movingAverage(data, windowSize);
    }

    // --- expSmooth ---

    public FilterResult expSmooth(IVector<Double> data, double alpha) {
        return TimeSeriesFiltering.exponentialSmoothing(data, alpha);
    }

    // --- gaussian ---

    public FilterResult gaussian(IVector<Double> data, double sigma) {
        return TimeSeriesFiltering.gaussianFilter(data, sigma);
    }

    // --- median ---

    public FilterResult median(IVector<Double> data, int windowSize) {
        return TimeSeriesFiltering.medianFilter(data, windowSize);
    }

    // --- lowPass ---

    public FilterResult lowPass(IVector<Double> data, double cutoffFreq, int order) {
        return TimeSeriesFiltering.lowPassFilter(data, cutoffFreq, DEFAULT_SR, order);
    }

    public FilterResult lowPass(IVector<Double> data, double cutoffFreq, double samplingRate, int order) {
        return TimeSeriesFiltering.lowPassFilter(data, cutoffFreq, samplingRate, order);
    }

    // --- highPass ---

    public FilterResult highPass(IVector<Double> data, double cutoffFreq, int order) {
        return TimeSeriesFiltering.highPassFilter(data, cutoffFreq, DEFAULT_SR, order);
    }

    public FilterResult highPass(IVector<Double> data, double cutoffFreq, double samplingRate, int order) {
        return TimeSeriesFiltering.highPassFilter(data, cutoffFreq, samplingRate, order);
    }

    // --- bandPass ---

    public FilterResult bandPass(IVector<Double> data, double lowFreq, double highFreq, int order) {
        return TimeSeriesFiltering.bandPassFilter(data, lowFreq, highFreq, DEFAULT_SR, order);
    }

    public FilterResult bandPass(IVector<Double> data, double lowFreq, double highFreq,
            double samplingRate, int order) {
        return TimeSeriesFiltering.bandPassFilter(data, lowFreq, highFreq, samplingRate, order);
    }

    // --- adaptive ---

    public FilterResult adaptive(IVector<Double> data, double learningRate) {
        return TimeSeriesFiltering.adaptiveFilter(data, learningRate);
    }
}
