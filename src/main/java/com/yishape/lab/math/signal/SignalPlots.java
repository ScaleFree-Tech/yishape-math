package com.yishape.lab.math.signal;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereFFT;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.util.Tuple2;

import java.util.ArrayList;
import java.util.List;

/**
 * 信号可视化器类 / Signal Visualizer Class
 * <p>
 * 提供信号数据的可视化功能，包括波形图、频谱图、功率谱密度、自相关图、小波分析图等。
 * 使用项目现有的viz包功能进行信号可视化。
 * </p>
 * <p>
 * Provides signal data visualization functionality including waveform plots, spectrum plots, 
 * power spectral density, autocorrelation plots, wavelet analysis plots, etc.
 * Uses existing viz package functionality for signal visualization.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class SignalPlots {
    
    /**
     * 绘制信号波形图 / Plot signal waveform
     * <p>
     * 显示信号的时域波形。
     * Display time-domain waveform of signal.
     * </p>
     *
     * @param signal 信号数据 / Signal data
     * @param samplingRate 采样率 / Sampling rate
     * @param title 图表标题 / Plot title
     * @return 波形图对象 / Waveform plot object
     */
    public static IPlot plotWaveform(IVector<Double> signal, double samplingRate, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("时间 (秒) / Time (s)");
        plot.ylabel("幅度 / Amplitude");
        
        // 创建时间轴 / Create time axis
        double[] timeArray = new double[signal.length()];
        for (int i = 0; i < signal.length(); i++) {
            timeArray[i] = i / samplingRate;
        }
        IVector<Double> time = Linalg.vector(timeArray);
        
        plot.line(time, signal);
        
        return plot;
    }
    
    /**
     * 绘制信号频谱图 / Plot signal spectrum
     * <p>
     * 显示信号的频域表示。
     * Display frequency-domain representation of signal.
     * </p>
     *
     * @param signal 信号数据 / Signal data
     * @param samplingRate 采样率 / Sampling rate
     * @param title 图表标题 / Plot title
     * @return 频谱图对象 / Spectrum plot object
     */
    public static IPlot plotSpectrum(IVector<Double> signal, double samplingRate, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("频率 (Hz) / Frequency (Hz)");
        plot.ylabel("幅度 / Magnitude");
        
        // 计算FFT / Calculate FFT
        Complex[] signalComplex = new Complex[signal.length()];
        for (int i = 0; i < signal.length(); i++) {
            signalComplex[i] = new Complex(signal.get(i), 0);
        }
        // 零填充确保长度为2的幂
        Complex[] paddedSignal = RereFFT.zeroPadToPowerOfTwo(signalComplex);
        Complex[] fftResult = RereFFT.fft(paddedSignal);
        int n = fftResult.length;
        int halfN = n / 2;
        
        // 计算频率轴 / Calculate frequency axis
        double[] freqArray = new double[halfN];
        double[] magnitudeArray = new double[halfN];
        
        for (int i = 0; i < halfN; i++) {
            freqArray[i] = i * samplingRate / n;
            magnitudeArray[i] = Math.sqrt(fftResult[i].real * fftResult[i].real + 
                                        fftResult[i].imag * fftResult[i].imag);
        }
        
        IVector<Double> frequencies = Linalg.vector(freqArray);
        IVector<Double> magnitudes = Linalg.vector(magnitudeArray);
        
        plot.line(frequencies, magnitudes);
        
        return plot;
    }
    
    /**
     * 绘制功率谱密度图 / Plot Power Spectral Density
     * <p>
     * 显示信号的功率谱密度。
     * Display power spectral density of signal.
     * </p>
     *
     * @param signal 信号数据 / Signal data
     * @param samplingRate 采样率 / Sampling rate
     * @param windowSize 窗函数大小 / Window size
     * @param overlap 重叠比例 / Overlap ratio
     * @param title 图表标题 / Plot title
     * @return 功率谱密度图对象 / PSD plot object
     */
    public static IPlot plotPowerSpectralDensity(IVector<Double> signal, double samplingRate, 
                                               int windowSize, double overlap, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("频率 (Hz) / Frequency (Hz)");
        plot.ylabel("功率谱密度 (dB/Hz) / PSD (dB/Hz)");
        
        // 计算功率谱密度 / Calculate PSD
        Tuple2<IVector<Double>, IVector<Double>> psdResult = 
            Signals.powerSpectralDensity(signal, windowSize, overlap, samplingRate);
        
        IVector<Double> frequencies = psdResult.getFirst();
        IVector<Double> psd = psdResult.getSecond();
        
        // 转换为dB / Convert to dB
        IVector<Double> psdDb = psd.apply(x -> 10 * Math.log10(Math.max(x, 1e-10)));
        
        plot.line(frequencies, psdDb);
        
        return plot;
    }
    
    /**
     * 绘制自相关图 / Plot autocorrelation
     * <p>
     * 显示信号的自相关函数。
     * Display autocorrelation function of signal.
     * </p>
     *
     * @param signal 信号数据 / Signal data
     * @param maxLag 最大滞后 / Maximum lag
     * @param title 图表标题 / Plot title
     * @return 自相关图对象 / Autocorrelation plot object
     */
    public static IPlot plotAutocorrelation(IVector<Double> signal, int maxLag, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("滞后 / Lag");
        plot.ylabel("自相关系数 / Autocorrelation");
        
        // 计算自相关 / Calculate autocorrelation
        IVector<Double> autocorr = Signals.autocorrelation(signal);
        
        // 创建滞后轴 / Create lag axis
        double[] lagArray = new double[maxLag + 1];
        for (int i = 0; i <= maxLag; i++) {
            lagArray[i] = i;
        }
        IVector<Double> lags = Linalg.vector(lagArray);
        
        plot.line(lags, autocorr);
        // 添加零线 / Add zero line
        // Note: axhline method not available in current IPlot interface
        
        return plot;
    }
    
    /**
     * 绘制互相关图 / Plot cross-correlation
     * <p>
     * 显示两个信号的互相关函数。
     * Display cross-correlation function of two signals.
     * </p>
     *
     * @param signal1 第一个信号 / First signal
     * @param signal2 第二个信号 / Second signal
     * @param maxLag 最大滞后 / Maximum lag
     * @param title 图表标题 / Plot title
     * @return 互相关图对象 / Cross-correlation plot object
     */
    public static IPlot plotCrossCorrelation(IVector<Double> signal1, IVector<Double> signal2, 
                                           int maxLag, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("滞后 / Lag");
        plot.ylabel("互相关系数 / Cross-correlation");
        
        // 计算互相关 / Calculate cross-correlation
        IVector<Double> crosscorr = Signals.crossCorrelation(signal1, signal2, maxLag);
        
        // 创建滞后轴 / Create lag axis
        double[] lagArray = new double[2 * maxLag + 1];
        for (int i = 0; i < 2 * maxLag + 1; i++) {
            lagArray[i] = i - maxLag;
        }
        IVector<Double> lags = Linalg.vector(lagArray);
        
        plot.line(lags, crosscorr);
        // 添加零线 / Add zero line
        // Note: axhline method not available in current IPlot interface
        
        return plot;
    }
    
    /**
     * 绘制信号统计图 / Plot signal statistics
     * <p>
     * 显示信号的基本统计信息。
     * Display basic statistics of signal.
     * </p>
     *
     * @param signal 信号数据 / Signal data
     * @param title 图表标题 / Plot title
     * @return 统计图对象 / Statistics plot object
     */
    public static IPlot plotSignalStatistics(IVector<Double> signal, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("统计量 / Statistics");
        plot.ylabel("数值 / Value");
        
        // 计算统计量 / Calculate statistics
        double mean = signal.meanValue();
        double std = signal.stdValue();
        double min = signal.minValue();
        double max = signal.maxValue();
        double rms = Math.sqrt(signal.apply(x -> x * x).meanValue());
        double energy = signal.apply(x -> x * x).sumValue();
        
        String[] statNames = {"均值 / Mean", "标准差 / Std", "最小值 / Min", 
                            "最大值 / Max", "RMS", "能量 / Energy"};
        double[] statValues = {mean, std, min, max, rms, energy};
        
        IVector<Double> values = Linalg.vector(statValues);
        List<String> labels = new ArrayList<>();
        for (String name : statNames) {
            labels.add(name);
        }
        
        plot.bar(labels,values);
        
        return plot;
    }
    
    /**
     * 绘制信号比较图 / Plot signal comparison
     * <p>
     * 比较两个信号的波形。
     * Compare waveforms of two signals.
     * </p>
     *
     * @param signal1 第一个信号 / First signal
     * @param signal2 第二个信号 / Second signal
     * @param samplingRate 采样率 / Sampling rate
     * @param title 图表标题 / Plot title
     * @return 比较图对象 / Comparison plot object
     */
    public static IPlot plotSignalComparison(IVector<Double> signal1, IVector<Double> signal2, 
                                           double samplingRate, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("时间 (秒) / Time (s)");
        plot.ylabel("幅度 / Amplitude");
        
        // 创建时间轴 / Create time axis
        int maxLength = Math.max(signal1.length(), signal2.length());
        double[] timeArray = new double[maxLength];
        for (int i = 0; i < maxLength; i++) {
            timeArray[i] = i / samplingRate;
        }
        IVector<Double> time = Linalg.vector(timeArray);
        
        // 绘制两个信号 / Plot both signals
        List<String> labels1 = new ArrayList<>();
        labels1.add("信号1 / Signal 1");
        List<String> labels2 = new ArrayList<>();
        labels2.add("信号2 / Signal 2");
        plot.line(time, signal1, labels1);
        plot.line(time, signal2, labels2);
        
        return plot;
    }
    
    /**
     * 绘制信号质量评估图 / Plot signal quality assessment
     * <p>
     * 显示信号的质量指标。
     * Display signal quality metrics.
     * </p>
     *
     * @param signal 信号数据 / Signal data
     * @param samplingRate 采样率 / Sampling rate
     * @param title 图表标题 / Plot title
     * @return 质量评估图对象 / Quality assessment plot object
     */
    public static IPlot plotSignalQuality(IVector<Double> signal, double samplingRate, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("质量指标 / Quality Metrics");
        plot.ylabel("数值 / Value");
        
        // 计算质量指标 / Calculate quality metrics
        double snr = Signals.signalToNoiseRatio(signal, signal); // 使用信号本身作为噪声估计
        double thd = 0.0; // 总谐波失真需要专门的计算方法
        double dynamicRange = signal.maxValue() - signal.minValue();
        double crestFactor = signal.abs().maxValue() / Math.sqrt(signal.apply(x -> x * x).meanValue());
        
        String[] qualityNames = {"信噪比 / SNR", "总谐波失真 / THD", 
                               "动态范围 / Dynamic Range", "峰值因子 / Crest Factor"};
        double[] qualityValues = {snr, thd, dynamicRange, crestFactor};
        
        IVector<Double> values = Linalg.vector(qualityValues);
        List<String> labels = new ArrayList<>();
        for (String name : qualityNames) {
            labels.add(name);
        }
        
        plot.bar(labels,values);
        
        return plot;
    }
    
    /**
     * 创建信号可视化仪表板 / Create signal visualization dashboard
     * <p>
     * 创建包含多个图表的信号可视化仪表板。
     * Create signal visualization dashboard with multiple charts.
     * </p>
     *
     * @param signal 信号数据 / Signal data
     * @param samplingRate 采样率 / Sampling rate
     * @param title 仪表板标题 / Dashboard title
     * @return 可视化图表列表 / List of visualization plots
     */
    public static List<IPlot> createSignalDashboard(IVector<Double> signal, double samplingRate, String title) {
        List<IPlot> plots = new ArrayList<>();
        
        // 添加各种图表 / Add various plots
        plots.add(plotWaveform(signal, samplingRate, title + " - 波形图 / Waveform"));
        plots.add(plotSpectrum(signal, samplingRate, title + " - 频谱图 / Spectrum"));
        plots.add(plotPowerSpectralDensity(signal, samplingRate, 1024, 0.5, 
                                         title + " - 功率谱密度 / PSD"));
        plots.add(plotAutocorrelation(signal, Math.min(100, signal.length() / 4), 
                                    title + " - 自相关 / Autocorrelation"));
        plots.add(plotSignalStatistics(signal, title + " - 统计信息 / Statistics"));
        plots.add(plotSignalQuality(signal, samplingRate, title + " - 质量评估 / Quality Assessment"));
        
        return plots;
    }
}
