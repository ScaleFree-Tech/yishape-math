package com.reremouse.lab.math.timeseries;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.signal.Signals;
import com.reremouse.lab.math.signal.core.SignalUtilities;
import com.reremouse.lab.util.Tuple2;

/**
 * 时间序列滤波类 / Time Series Filtering Class
 * <p>
 * 提供时间序列滤波功能，包括移动平均、指数平滑、卡尔曼滤波、小波滤波等。
 * 使用项目现有的signal包和linalg包功能进行滤波处理。
 * </p>
 * <p>
 * Provides time series filtering functionality including moving average, exponential smoothing,
 * Kalman filtering, wavelet filtering, etc. Uses existing signal and linalg package functionality
 * for filtering processing.
 * </p>
 *
 * @author lterb2
 * @version 1.0
 * @since 1.0
 */
public class TimeSeriesFiltering {
    
    /**
     * 滤波结果类 / Filtering Result Class
     */
    public static class FilterResult {
        public final TimeSeriesData filtered;
        public final TimeSeriesData noise;
        public final double snr;
        public final String filterType;
        
        public FilterResult(TimeSeriesData filtered, TimeSeriesData noise, double snr, String filterType) {
            this.filtered = filtered;
            this.noise = noise;
            this.snr = snr;
            this.filterType = filterType;
        }
    }
    
    /**
     * 移动平均滤波 / Moving Average Filter
     * <p>
     * 对时间序列进行移动平均滤波，用于平滑噪声。
     * Apply moving average filter to time series for noise smoothing.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param windowSize 窗口大小 / Window size
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult movingAverage(TimeSeriesData timeSeries, int variableIndex, int windowSize) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        IVector<Double> filtered = Signals.movingAverage(data, windowSize);
        
        // 计算噪声 / Calculate noise
        IVector<Double> noise = data.sub(filtered);
        
        // 计算信噪比 / Calculate SNR
        double signalPower = filtered.multiply(filtered).sum();
        double noisePower = noise.multiply(noise).sum();
        double snr = signalPower / (noisePower + 1e-10);
        
        // 创建滤波后的时间序列 / Create filtered time series
        IMatrix<Double> filteredData = timeSeries.getData().copy();
        filteredData.setColumn(variableIndex, filtered);
        TimeSeriesData filteredTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), filteredData, timeSeries.getColumnNames());
        
        // 创建噪声时间序列 / Create noise time series
        IMatrix<Double> noiseData = Linalg.zeros(timeSeries.getLength(), timeSeries.getNumVariables());
        noiseData.setColumn(variableIndex, noise);
        TimeSeriesData noiseTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), noiseData, timeSeries.getColumnNames());
        
        return new FilterResult(filteredTimeSeries, noiseTimeSeries, snr, "MovingAverage");
    }
    
    /**
     * 移动平均滤波（按列名） / Moving Average Filter (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param windowSize 窗口大小 / Window size
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult movingAverage(TimeSeriesData timeSeries, String columnName, int windowSize) {
        int index = timeSeries.getVariableIndex(columnName);
        return movingAverage(timeSeries, index, windowSize);
    }
    
    /**
     * 指数平滑滤波 / Exponential Smoothing Filter
     * <p>
     * 对时间序列进行指数平滑滤波，适用于趋势数据。
     * Apply exponential smoothing filter to time series, suitable for trend data.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param alpha 平滑参数 (0-1) / Smoothing parameter (0-1)
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult exponentialSmoothing(TimeSeriesData timeSeries, int variableIndex, double alpha) {
        if (alpha < 0 || alpha > 1) {
            throw new IllegalArgumentException("平滑参数必须在0-1之间");
        }
        
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        IVector<Double> filtered = exponentialSmoothing(data, alpha);
        
        // 计算噪声 / Calculate noise
        IVector<Double> noise = data.sub(filtered);
        
        // 计算信噪比 / Calculate SNR
        double signalPower = filtered.multiply(filtered).sum();
        double noisePower = noise.multiply(noise).sum();
        double snr = signalPower / (noisePower + 1e-10);
        
        // 创建滤波后的时间序列 / Create filtered time series
        IMatrix<Double> filteredData = timeSeries.getData().copy();
        filteredData.setColumn(variableIndex, filtered);
        TimeSeriesData filteredTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), filteredData, timeSeries.getColumnNames());
        
        // 创建噪声时间序列 / Create noise time series
        IMatrix<Double> noiseData = Linalg.zeros(timeSeries.getLength(), timeSeries.getNumVariables());
        noiseData.setColumn(variableIndex, noise);
        TimeSeriesData noiseTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), noiseData, timeSeries.getColumnNames());
        
        return new FilterResult(filteredTimeSeries, noiseTimeSeries, snr, "ExponentialSmoothing");
    }
    
    /**
     * 指数平滑滤波（按列名） / Exponential Smoothing Filter (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param alpha 平滑参数 / Smoothing parameter
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult exponentialSmoothing(TimeSeriesData timeSeries, String columnName, double alpha) {
        int index = timeSeries.getVariableIndex(columnName);
        return exponentialSmoothing(timeSeries, index, alpha);
    }
    
    /**
     * 高斯滤波 / Gaussian Filter
     * <p>
     * 对时间序列进行高斯滤波，提供平滑的滤波效果。
     * Apply Gaussian filter to time series for smooth filtering effect.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param sigma 高斯核标准差 / Gaussian kernel standard deviation
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult gaussianFilter(TimeSeriesData timeSeries, int variableIndex, double sigma) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        IVector<Double> filtered = Signals.gaussianFilter(data, sigma);
        
        // 计算噪声 / Calculate noise
        IVector<Double> noise = data.sub(filtered);
        
        // 计算信噪比 / Calculate SNR
        double signalPower = filtered.multiply(filtered).sum();
        double noisePower = noise.multiply(noise).sum();
        double snr = signalPower / (noisePower + 1e-10);
        
        // 创建滤波后的时间序列 / Create filtered time series
        IMatrix<Double> filteredData = timeSeries.getData().copy();
        filteredData.setColumn(variableIndex, filtered);
        TimeSeriesData filteredTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), filteredData, timeSeries.getColumnNames());
        
        // 创建噪声时间序列 / Create noise time series
        IMatrix<Double> noiseData = Linalg.zeros(timeSeries.getLength(), timeSeries.getNumVariables());
        noiseData.setColumn(variableIndex, noise);
        TimeSeriesData noiseTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), noiseData, timeSeries.getColumnNames());
        
        return new FilterResult(filteredTimeSeries, noiseTimeSeries, snr, "Gaussian");
    }
    
    /**
     * 高斯滤波（按列名） / Gaussian Filter (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param sigma 高斯核标准差 / Gaussian kernel standard deviation
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult gaussianFilter(TimeSeriesData timeSeries, String columnName, double sigma) {
        int index = timeSeries.getVariableIndex(columnName);
        return gaussianFilter(timeSeries, index, sigma);
    }
    
    /**
     * 中值滤波 / Median Filter
     * <p>
     * 对时间序列进行中值滤波，有效去除脉冲噪声。
     * Apply median filter to time series for effective impulse noise removal.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param windowSize 窗口大小 / Window size
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult medianFilter(TimeSeriesData timeSeries, int variableIndex, int windowSize) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        IVector<Double> filtered = Signals.medianFilter(data, windowSize);
        
        // 计算噪声 / Calculate noise
        IVector<Double> noise = data.sub(filtered);
        
        // 计算信噪比 / Calculate SNR
        double signalPower = filtered.multiply(filtered).sum();
        double noisePower = noise.multiply(noise).sum();
        double snr = signalPower / (noisePower + 1e-10);
        
        // 创建滤波后的时间序列 / Create filtered time series
        IMatrix<Double> filteredData = timeSeries.getData().copy();
        filteredData.setColumn(variableIndex, filtered);
        TimeSeriesData filteredTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), filteredData, timeSeries.getColumnNames());
        
        // 创建噪声时间序列 / Create noise time series
        IMatrix<Double> noiseData = Linalg.zeros(timeSeries.getLength(), timeSeries.getNumVariables());
        noiseData.setColumn(variableIndex, noise);
        TimeSeriesData noiseTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), noiseData, timeSeries.getColumnNames());
        
        return new FilterResult(filteredTimeSeries, noiseTimeSeries, snr, "Median");
    }
    
    /**
     * 中值滤波（按列名） / Median Filter (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param windowSize 窗口大小 / Window size
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult medianFilter(TimeSeriesData timeSeries, String columnName, int windowSize) {
        int index = timeSeries.getVariableIndex(columnName);
        return medianFilter(timeSeries, index, windowSize);
    }
    
    /**
     * 低通滤波 / Low Pass Filter
     * <p>
     * 对时间序列进行低通滤波，去除高频噪声。
     * Apply low pass filter to time series for high frequency noise removal.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param order 滤波器阶数 / Filter order
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult lowPassFilter(TimeSeriesData timeSeries, int variableIndex, 
                                           double cutoffFreq, int order) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        double samplingRate = timeSeries.getSamplingRate();
        
        IVector<Double> filtered = Signals.butterworthLowPass(data, cutoffFreq, samplingRate, order);
        
        // 计算噪声 / Calculate noise
        IVector<Double> noise = data.sub(filtered);
        
        // 计算信噪比 / Calculate SNR
        double signalPower = filtered.multiply(filtered).sum();
        double noisePower = noise.multiply(noise).sum();
        double snr = signalPower / (noisePower + 1e-10);
        
        // 创建滤波后的时间序列 / Create filtered time series
        IMatrix<Double> filteredData = timeSeries.getData().copy();
        filteredData.setColumn(variableIndex, filtered);
        TimeSeriesData filteredTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), filteredData, timeSeries.getColumnNames());
        
        // 创建噪声时间序列 / Create noise time series
        IMatrix<Double> noiseData = Linalg.zeros(timeSeries.getLength(), timeSeries.getNumVariables());
        noiseData.setColumn(variableIndex, noise);
        TimeSeriesData noiseTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), noiseData, timeSeries.getColumnNames());
        
        return new FilterResult(filteredTimeSeries, noiseTimeSeries, snr, "LowPass");
    }
    
    /**
     * 低通滤波（按列名） / Low Pass Filter (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param order 滤波器阶数 / Filter order
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult lowPassFilter(TimeSeriesData timeSeries, String columnName, 
                                           double cutoffFreq, int order) {
        int index = timeSeries.getVariableIndex(columnName);
        return lowPassFilter(timeSeries, index, cutoffFreq, order);
    }
    
    /**
     * 高通滤波 / High Pass Filter
     * <p>
     * 对时间序列进行高通滤波，去除低频趋势。
     * Apply high pass filter to time series for low frequency trend removal.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param order 滤波器阶数 / Filter order
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult highPassFilter(TimeSeriesData timeSeries, int variableIndex, 
                                            double cutoffFreq, int order) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        double samplingRate = timeSeries.getSamplingRate();
        
        IVector<Double> filtered = Signals.butterworthHighPass(data, cutoffFreq, samplingRate, order);
        
        // 计算噪声 / Calculate noise
        IVector<Double> noise = data.sub(filtered);
        
        // 计算信噪比 / Calculate SNR
        double signalPower = filtered.multiply(filtered).sum();
        double noisePower = noise.multiply(noise).sum();
        double snr = signalPower / (noisePower + 1e-10);
        
        // 创建滤波后的时间序列 / Create filtered time series
        IMatrix<Double> filteredData = timeSeries.getData().copy();
        filteredData.setColumn(variableIndex, filtered);
        TimeSeriesData filteredTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), filteredData, timeSeries.getColumnNames());
        
        // 创建噪声时间序列 / Create noise time series
        IMatrix<Double> noiseData = Linalg.zeros(timeSeries.getLength(), timeSeries.getNumVariables());
        noiseData.setColumn(variableIndex, noise);
        TimeSeriesData noiseTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), noiseData, timeSeries.getColumnNames());
        
        return new FilterResult(filteredTimeSeries, noiseTimeSeries, snr, "HighPass");
    }
    
    /**
     * 高通滤波（按列名） / High Pass Filter (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param order 滤波器阶数 / Filter order
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult highPassFilter(TimeSeriesData timeSeries, String columnName, 
                                            double cutoffFreq, int order) {
        int index = timeSeries.getVariableIndex(columnName);
        return highPassFilter(timeSeries, index, cutoffFreq, order);
    }
    
    /**
     * 带通滤波 / Band Pass Filter
     * <p>
     * 对时间序列进行带通滤波，保留特定频率范围。
     * Apply band pass filter to time series for specific frequency range retention.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param lowFreq 低频截止 / Low frequency cutoff
     * @param highFreq 高频截止 / High frequency cutoff
     * @param order 滤波器阶数 / Filter order
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult bandPassFilter(TimeSeriesData timeSeries, int variableIndex, 
                                            double lowFreq, double highFreq, int order) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        double samplingRate = timeSeries.getSamplingRate();
        IVector<Double> filtered = Signals.bandPass(data, lowFreq, highFreq, samplingRate, order);
        
        // 计算噪声 / Calculate noise
        IVector<Double> noise = data.sub(filtered);
        
        // 计算信噪比 / Calculate SNR
        double signalPower = filtered.multiply(filtered).sum();
        double noisePower = noise.multiply(noise).sum();
        double snr = signalPower / (noisePower + 1e-10);
        
        // 创建滤波后的时间序列 / Create filtered time series
        IMatrix<Double> filteredData = timeSeries.getData().copy();
        filteredData.setColumn(variableIndex, filtered);
        TimeSeriesData filteredTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), filteredData, timeSeries.getColumnNames());
        
        // 创建噪声时间序列 / Create noise time series
        IMatrix<Double> noiseData = Linalg.zeros(timeSeries.getLength(), timeSeries.getNumVariables());
        noiseData.setColumn(variableIndex, noise);
        TimeSeriesData noiseTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), noiseData, timeSeries.getColumnNames());
        
        return new FilterResult(filteredTimeSeries, noiseTimeSeries, snr, "BandPass");
    }
    
    /**
     * 带通滤波（按列名） / Band Pass Filter (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param lowFreq 低频截止 / Low frequency cutoff
     * @param highFreq 高频截止 / High frequency cutoff
     * @param order 滤波器阶数 / Filter order
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult bandPassFilter(TimeSeriesData timeSeries, String columnName, 
                                            double lowFreq, double highFreq, int order) {
        int index = timeSeries.getVariableIndex(columnName);
        return bandPassFilter(timeSeries, index, lowFreq, highFreq, order);
    }
    
    /**
     * 自适应滤波 / Adaptive Filter
     * <p>
     * 对时间序列进行自适应滤波，根据信号特性自动调整滤波参数。
     * Apply adaptive filter to time series, automatically adjusting filter parameters based on signal characteristics.
     * </p>
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param variableIndex 变量索引 / Variable index
     * @param learningRate 学习率 / Learning rate
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult adaptiveFilter(TimeSeriesData timeSeries, int variableIndex, double learningRate) {
        IVector<Double> data = timeSeries.getVariable(variableIndex);
        IVector<Double> filtered = adaptiveFilter(data, learningRate);
        
        // 计算噪声 / Calculate noise
        IVector<Double> noise = data.sub(filtered);
        
        // 计算信噪比 / Calculate SNR
        double signalPower = filtered.multiply(filtered).sum();
        double noisePower = noise.multiply(noise).sum();
        double snr = signalPower / (noisePower + 1e-10);
        
        // 创建滤波后的时间序列 / Create filtered time series
        IMatrix<Double> filteredData = timeSeries.getData().copy();
        filteredData.setColumn(variableIndex, filtered);
        TimeSeriesData filteredTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), filteredData, timeSeries.getColumnNames());
        
        // 创建噪声时间序列 / Create noise time series
        IMatrix<Double> noiseData = Linalg.zeros(timeSeries.getLength(), timeSeries.getNumVariables());
        noiseData.setColumn(variableIndex, noise);
        TimeSeriesData noiseTimeSeries = new TimeSeriesData(
            timeSeries.getTimestamps(), noiseData, timeSeries.getColumnNames());
        
        return new FilterResult(filteredTimeSeries, noiseTimeSeries, snr, "Adaptive");
    }
    
    /**
     * 自适应滤波（按列名） / Adaptive Filter (by column name)
     *
     * @param timeSeries 输入时间序列 / Input time series
     * @param columnName 列名 / Column name
     * @param learningRate 学习率 / Learning rate
     * @return 滤波结果 / Filtering result
     */
    public static FilterResult adaptiveFilter(TimeSeriesData timeSeries, String columnName, double learningRate) {
        int index = timeSeries.getVariableIndex(columnName);
        return adaptiveFilter(timeSeries, index, learningRate);
    }
    
    // ========== 私有辅助方法 / Private Helper Methods ==========
    
    /**
     * 指数平滑实现 / Exponential smoothing implementation
     */
    private static IVector<Double> exponentialSmoothing(IVector<Double> data, double alpha) {
        int length = data.length();
        IVector<Double> smoothed = Linalg.zeros(length);
        
        // 初始化 / Initialize
        smoothed.set(0, data.get(0));
        
        // 指数平滑 / Exponential smoothing
        for (int i = 1; i < length; i++) {
            double value = alpha * data.get(i) + (1 - alpha) * smoothed.get(i - 1);
            smoothed.set(i, value);
        }
        
        return smoothed;
    }
    
    /**
     * 自适应滤波实现 / Adaptive filter implementation
     */
    private static IVector<Double> adaptiveFilter(IVector<Double> data, double learningRate) {
        int length = data.length();
        IVector<Double> filtered = Linalg.zeros(length);
        
        // 初始化 / Initialize
        filtered.set(0, data.get(0));
        
        // 自适应滤波 / Adaptive filtering
        for (int i = 1; i < length; i++) {
            double error = data.get(i) - filtered.get(i - 1);
            double value = filtered.get(i - 1) + learningRate * error;
            filtered.set(i, value);
        }
        
        return filtered;
    }
}

