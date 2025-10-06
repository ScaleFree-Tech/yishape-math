package com.yishape.lab.math.signal.core;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.Signals;

/**
 * 信号工具类 / Signal Utilities Class
 * <p>
 * 提供各种信号处理工具函数，包括窗函数、重采样、信号拼接、信号检测等。
 * 使用IVector和IMatrix接口进行向量和矩阵操作，确保与现有代码库的兼容性。
 * </p>
 * <p>
 * Provides various signal processing utility functions including window functions, resampling,
 * signal concatenation, signal detection, etc. Uses IVector and IMatrix interfaces for vector and
 * matrix operations to ensure compatibility with existing codebase.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class SignalUtilities {

    /**
     * 窗函数类型枚举 / Window Function Type Enum
     */
    public enum WindowType {
        RECTANGULAR,    // 矩形窗 / Rectangular window
        HANNING,        // 汉宁窗 / Hanning window
        HAMMING,        // 汉明窗 / Hamming window
        BLACKMAN,       // 布莱克曼窗 / Blackman window
        KAISER,         // 凯泽窗 / Kaiser window
        BARTLETT,       // 巴特利特窗 / Bartlett window
        GAUSSIAN        // 高斯窗 / Gaussian window
    }

    /**
     * 生成窗函数 / Generate Window Function
     * <p>
     * 生成指定类型和大小的窗函数。
     * Generate window function of specified type and size.
     * </p>
     *
     * @param size 窗函数大小 / Window size
     * @param type 窗函数类型 / Window type
     * @param param 窗函数参数（如凯泽窗的β值） / Window parameter (e.g., β value for Kaiser window)
     * @return 窗函数向量 / Window function vector
     */
    public static IVector<Double> window(int size, WindowType type, double param) {
        switch (type) {
            case RECTANGULAR:
                return rectangularWindow(size);
            case HANNING:
                return hanningWindow(size);
            case HAMMING:
                return hammingWindow(size);
            case BLACKMAN:
                return blackmanWindow(size);
            case KAISER:
                return kaiserWindow(size, param);
            case BARTLETT:
                return bartlettWindow(size);
            case GAUSSIAN:
                return gaussianWindow(size, param);
            default:
                throw new IllegalArgumentException("不支持的窗函数类型");
        }
    }

    /**
     * 生成窗函数（使用默认参数） / Generate Window Function (with default parameters)
     */
    public static IVector<Double> window(int size, WindowType type) {
        return window(size, type, 0);
    }

    /**
     * 矩形窗 / Rectangular Window
     */
    private static IVector<Double> rectangularWindow(int size) {
        return Linalg.ones(size);
    }

    /**
     * 汉宁窗 / Hanning Window
     */
    private static IVector<Double> hanningWindow(int size) {
        IVector<Double> window = Linalg.zeros(size);
        for (int i = 0; i < size; i++) {
            double value = 0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1)));
            window.set(i, value);
        }
        return window;
    }

    /**
     * 汉明窗 / Hamming Window
     */
    private static IVector<Double> hammingWindow(int size) {
        IVector<Double> window = Linalg.zeros(size);
        for (int i = 0; i < size; i++) {
            double value = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (size - 1));
            window.set(i, value);
        }
        return window;
    }

    /**
     * 布莱克曼窗 / Blackman Window
     */
    private static IVector<Double> blackmanWindow(int size) {
        IVector<Double> window = Linalg.zeros(size);
        for (int i = 0; i < size; i++) {
            double value = 0.42 - 0.5 * Math.cos(2 * Math.PI * i / (size - 1)) + 
                          0.08 * Math.cos(4 * Math.PI * i / (size - 1));
            window.set(i, value);
        }
        return window;
    }

    /**
     * 凯泽窗 / Kaiser Window
     */
    private static IVector<Double> kaiserWindow(int size, double beta) {
        IVector<Double> window = Linalg.zeros(size);
        double i0Beta = besselI0(beta);
        
        for (int i = 0; i < size; i++) {
            double x = 2.0 * i / (size - 1) - 1.0;
            double arg = beta * Math.sqrt(1 - x * x);
            double value = besselI0(arg) / i0Beta;
            window.set(i, value);
        }
        return window;
    }

    /**
     * 巴特利特窗 / Bartlett Window
     */
    private static IVector<Double> bartlettWindow(int size) {
        IVector<Double> window = Linalg.zeros(size);
        for (int i = 0; i < size; i++) {
            double value = 1.0 - Math.abs(2.0 * i / (size - 1) - 1.0);
            window.set(i, value);
        }
        return window;
    }

    /**
     * 高斯窗 / Gaussian Window
     */
    private static IVector<Double> gaussianWindow(int size, double sigma) {
        if (sigma <= 0) sigma = 0.4;
        
        IVector<Double> window = Linalg.zeros(size);
        int center = size / 2;
        
        for (int i = 0; i < size; i++) {
            double x = (i - center) / (sigma * (size - 1) / 2);
            double value = Math.exp(-0.5 * x * x);
            window.set(i, value);
        }
        return window;
    }

    /**
     * 信号重采样 / Signal Resampling
     * <p>
     * 使用线性插值对信号进行重采样。
     * Resample signal using linear interpolation.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param newLength 新的信号长度 / New signal length
     * @return 重采样后的信号 / Resampled signal
     */
    public static IVector<Double> resample(IVector<Double> signal, int newLength) {
        if (newLength <= 0) {
            throw new IllegalArgumentException("新长度必须大于0");
        }
        
        int oldLength = signal.length();
        if (oldLength == newLength) {
            return signal.copy();
        }
        
        IVector<Double> resampled = Linalg.zeros(newLength);
        
        for (int i = 0; i < newLength; i++) {
            double pos = (double) i * (oldLength - 1) / (newLength - 1);
            int index = (int) pos;
            double fraction = pos - index;
            
            if (index >= oldLength - 1) {
                resampled.set(i, signal.get(oldLength - 1));
            } else {
                double value = signal.get(index) * (1 - fraction) + signal.get(index + 1) * fraction;
                resampled.set(i, value);
            }
        }
        
        return resampled;
    }

    /**
     * 信号拼接 / Signal Concatenation
     * <p>
     * 将多个信号向量拼接成一个信号。
     * Concatenate multiple signal vectors into one signal.
     * </p>
     *
     * @param signals 信号向量数组 / Signal vector array
     * @return 拼接后的信号 / Concatenated signal
     */
    @SafeVarargs
    public static IVector<Double> concatenate(IVector<Double>... signals) {
        if (signals.length == 0) {
            return Linalg.zeros(0);
        }
        
        int totalLength = 0;
        for (IVector<Double> signal : signals) {
            totalLength += signal.length();
        }
        
        IVector<Double> concatenated = Linalg.zeros(totalLength);
        int index = 0;
        
        for (IVector<Double> signal : signals) {
            for (int i = 0; i < signal.length(); i++) {
                concatenated.set(index++, signal.get(i));
            }
        }
        
        return concatenated;
    }

    /**
     * 信号分割 / Signal Segmentation
     * <p>
     * 将信号分割成指定长度的段。
     * Segment signal into segments of specified length.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param segmentLength 段长度 / Segment length
     * @param overlap 重叠长度 / Overlap length
     * @return 信号段数组 / Signal segment array
     */
    public static IVector<Double>[] segment(IVector<Double> signal, int segmentLength, int overlap) {
        if (segmentLength <= 0 || overlap < 0 || overlap >= segmentLength) {
            throw new IllegalArgumentException("段长度必须大于0，重叠长度必须小于段长度");
        }
        
        int signalLength = signal.length();
        int hopSize = segmentLength - overlap;
        int numSegments = (signalLength - overlap) / hopSize;
        
        @SuppressWarnings("unchecked")
        IVector<Double>[] segments = new IVector[numSegments];
        
        for (int i = 0; i < numSegments; i++) {
            int start = i * hopSize;
            int end = Math.min(start + segmentLength, signalLength);
            segments[i] = signal.slice(start, end);
        }
        
        return segments;
    }

    /**
     * 信号检测 - 峰值检测 / Signal Detection - Peak Detection
     * <p>
     * 检测信号中的峰值点。
     * Detect peak points in signal.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param threshold 阈值 / Threshold
     * @param minDistance 最小峰值距离 / Minimum peak distance
     * @return 峰值位置数组 / Peak position array
     */
    public static int[] detectPeaks(IVector<Double> signal, double threshold, int minDistance) {
        if (minDistance <= 0) {
            throw new IllegalArgumentException("最小距离必须大于0");
        }
        
        java.util.List<Integer> peaks = new java.util.ArrayList<>();
        
        for (int i = 1; i < signal.length() - 1; i++) {
            double current = signal.get(i);
            double prev = signal.get(i - 1);
            double next = signal.get(i + 1);
            
            // 检查是否为峰值 / Check if it's a peak
            if (current > prev && current > next && current > threshold) {
                // 检查与最近峰值的距离 / Check distance from nearest peak
                boolean validPeak = true;
                for (int peak : peaks) {
                    if (Math.abs(i - peak) < minDistance) {
                        validPeak = false;
                        break;
                    }
                }
                
                if (validPeak) {
                    peaks.add(i);
                }
            }
        }
        
        return peaks.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * 信号检测 - 过零检测 / Signal Detection - Zero Crossing Detection
     * <p>
     * 检测信号中的过零点。
     * Detect zero crossings in signal.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return 过零位置数组 / Zero crossing position array
     */
    public static int[] detectZeroCrossings(IVector<Double> signal) {
        java.util.List<Integer> crossings = new java.util.ArrayList<>();
        
        for (int i = 1; i < signal.length(); i++) {
            double current = signal.get(i);
            double prev = signal.get(i - 1);
            
            // 检查是否过零 / Check if zero crossing
            if ((current > 0 && prev <= 0) || (current < 0 && prev >= 0)) {
                crossings.add(i);
            }
        }
        
        return crossings.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * 信号检测 - 突变检测 / Signal Detection - Change Point Detection
     * <p>
     * 检测信号中的突变点。
     * Detect change points in signal.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param windowSize 窗口大小 / Window size
     * @param threshold 阈值 / Threshold
     * @return 突变位置数组 / Change point position array
     */
    public static int[] detectChangePoints(IVector<Double> signal, int windowSize, double threshold) {
        if (windowSize <= 0 || windowSize >= signal.length()) {
            throw new IllegalArgumentException("窗口大小无效");
        }
        
        java.util.List<Integer> changePoints = new java.util.ArrayList<>();
        
        for (int i = windowSize; i < signal.length() - windowSize; i++) {
            // 计算前后窗口的统计量 / Calculate statistics of front and back windows
            IVector<Double> frontWindow = signal.slice(i - windowSize, i);
            IVector<Double> backWindow = signal.slice(i, i + windowSize);
            
            double frontMean = frontWindow.mean();
            double backMean = backWindow.mean();
            double frontVar = frontWindow.var();
            double backVar = backWindow.var();
            
            // 计算变化量 / Calculate change amount
            double meanChange = Math.abs(backMean - frontMean);
            double varChange = Math.abs(backVar - frontVar);
            
            if (meanChange > threshold || varChange > threshold) {
                changePoints.add(i);
            }
        }
        
        return changePoints.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * 信号归一化 / Signal Normalization
     * <p>
     * 将信号归一化到指定范围。
     * Normalize signal to specified range.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param min 目标最小值 / Target minimum value
     * @param max 目标最大值 / Target maximum value
     * @return 归一化后的信号 / Normalized signal
     */
    public static IVector<Double> normalize(IVector<Double> signal, double min, double max) {
        if (min >= max) {
            throw new IllegalArgumentException("最小值必须小于最大值");
        }
        
        double signalMin = signal.min();
        double signalMax = signal.max();
        double signalRange = signalMax - signalMin;
        
        if (signalRange == 0) {
            return Linalg.zeros(signal.length()).addScalar((min + max) / 2);
        }
        
        double scale = (max - min) / signalRange;
        return signal.subScalar(signalMin).multiplyScalar(scale).addScalar(min);
    }

    /**
     * 信号去趋势 / Signal Detrending
     * <p>
     * 去除信号中的线性趋势。
     * Remove linear trend from signal.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return 去趋势后的信号 / Detrended signal
     */
    public static IVector<Double> detrend(IVector<Double> signal) {
        int n = signal.length();
        IVector<Double> x = Linalg.range(n);
        
        // 计算线性回归 / Calculate linear regression
        double xMean = x.mean();
        double yMean = signal.mean();
        
        double numerator = 0;
        double denominator = 0;
        
        for (int i = 0; i < n; i++) {
            double xDiff = x.get(i) - xMean;
            double yDiff = signal.get(i) - yMean;
            numerator += xDiff * yDiff;
            denominator += xDiff * xDiff;
        }
        
        double slope = denominator == 0 ? 0 : numerator / denominator;
        double intercept = yMean - slope * xMean;
        
        // 计算趋势线 / Calculate trend line
        IVector<Double> trend = x.multiplyScalar(slope).addScalar(intercept);
        
        // 去除趋势 / Remove trend
        return signal.sub(trend);
    }

    /**
     * 信号平滑 / Signal Smoothing
     * <p>
     * 使用移动平均对信号进行平滑。
     * Smooth signal using moving average.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param windowSize 窗口大小 / Window size
     * @return 平滑后的信号 / Smoothed signal
     */
    public static IVector<Double> smooth(IVector<Double> signal, int windowSize) {
        return Signals.movingAverage(signal, windowSize);
    }

    // ========== 辅助方法 / Helper Methods ==========

    /**
     * 计算修正贝塞尔函数I0 / Calculate Modified Bessel Function I0
     */
    private static double besselI0(double x) {
        double result = 1.0;
        double term = 1.0;
        
        for (int i = 1; i <= 20; i++) {
            term *= (x / 2) * (x / 2) / (i * i);
            result += term;
        }
        
        return result;
    }
}
