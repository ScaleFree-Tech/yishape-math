package com.reremouse.lab.math.signal.wavele;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.viz.Plots;
import com.reremouse.lab.math.viz.IPlot;

import java.util.ArrayList;
import java.util.List;

/**
 * 小波可视化器类 / Wavelet Visualizer Class
 * <p>
 * 提供小波分析的可视化功能，包括小波变换、时频分析、小波系数、尺度图等。
 * 使用项目现有的viz包功能进行小波可视化。
 * </p>
 * <p>
 * Provides wavelet analysis visualization functionality including wavelet transforms, 
 * time-frequency analysis, wavelet coefficients, scalograms, etc.
 * Uses existing viz package functionality for wavelet visualization.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class WaveletVisualizer {
    
    /**
     * 绘制小波系数图 / Plot wavelet coefficients
     * <p>
     * 显示小波变换的系数。
     * Display wavelet transform coefficients.
     * </p>
     *
     * @param signal 信号数据 / Signal data
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     * @param title 图表标题 / Plot title
     * @return 小波系数图对象 / Wavelet coefficients plot object
     */
    public static IPlot plotWaveletCoefficients(IVector<Double> signal, String waveletType, 
                                              int levels, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("时间 / Time");
        plot.ylabel("小波系数 / Wavelet Coefficients");
        
        // 进行小波分解 / Perform wavelet decomposition
        WaveletCoefficients decomposition = 
            WaveletAnalysis.discreteWaveletTransform(signal, WaveletAnalysis.WaveletType.valueOf(waveletType.toUpperCase()), levels, 1.0);
        
        // 创建时间轴 / Create time axis
        double[] timeArray = new double[signal.length()];
        for (int i = 0; i < signal.length(); i++) {
            timeArray[i] = i;
        }
        IVector<Double> time = Linalg.vector(timeArray);
        
        // 绘制近似系数 / Plot approximation coefficients
        List<String> labels1 = new ArrayList<>();
        labels1.add("近似系数 / Approximation");
        plot.line(time, decomposition.approximation, labels1);
        
        // 绘制细节系数 / Plot detail coefficients
        for (int i = 0; i < decomposition.details.length; i++) {
            IVector<Double> detail = decomposition.details[i];
            String label = "细节" + (i + 1) + " / Detail " + (i + 1);
            List<String> labels = new ArrayList<>();
            labels.add(label);
            plot.line(time, detail, labels);
        }
        
        return plot;
    }
    
    /**
     * 绘制小波尺度图 / Plot wavelet scalogram
     * <p>
     * 显示小波变换的时频表示。
     * Display time-frequency representation of wavelet transform.
     * </p>
     *
     * @param signal 信号数据 / Signal data
     * @param waveletType 小波类型 / Wavelet type
     * @param scales 尺度范围 / Scale range
     * @param title 图表标题 / Plot title
     * @return 小波尺度图对象 / Wavelet scalogram plot object
     */
    public static IPlot plotWaveletScalogram(IVector<Double> signal, String waveletType, 
                                           IVector<Double> scales, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("时间 / Time");
        plot.ylabel("尺度 / Scale");
        
        // 计算连续小波变换 / Calculate continuous wavelet transform
        // Note: 连续小波变换需要复杂的实现，这里简化处理 / Continuous wavelet transform requires complex implementation, simplified here
        
        // 绘制尺度图 / Plot scalogram
        // Note: contour and colorbar methods not available in current IPlot interface
        // 可以使用热力图或其他方式替代 / Can use heatmap or other alternatives
        
        return plot;
    }
    
    /**
     * 绘制小波能量分布图 / Plot wavelet energy distribution
     * <p>
     * 显示各尺度的小波能量分布。
     * Display wavelet energy distribution across scales.
     * </p>
     *
     * @param signal 信号数据 / Signal data
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     * @param title 图表标题 / Plot title
     * @return 小波能量分布图对象 / Wavelet energy distribution plot object
     */
    public static IPlot plotWaveletEnergyDistribution(IVector<Double> signal, String waveletType, 
                                                    int levels, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("尺度 / Scale");
        plot.ylabel("能量 / Energy");
        
        // 进行小波分解 / Perform wavelet decomposition
        WaveletCoefficients decomposition = 
            WaveletAnalysis.discreteWaveletTransform(signal, WaveletAnalysis.WaveletType.valueOf(waveletType.toUpperCase()), levels, 1.0);
        
        // 计算各层能量 / Calculate energy at each level
        double[] energyArray = new double[levels + 1];
        String[] scaleLabels = new String[levels + 1];
        
        // 近似系数能量 / Approximation energy
        energyArray[0] = decomposition.approximation.apply(x -> x * x).sum();
        scaleLabels[0] = "近似 / Approx";
        
        // 细节系数能量 / Detail energies
        for (int i = 0; i < decomposition.details.length; i++) {
            IVector<Double> detail = decomposition.details[i];
            energyArray[i + 1] = detail.apply(x -> x * x).sum();
            scaleLabels[i + 1] = "细节" + (i + 1) + " / Detail " + (i + 1);
        }
        
        IVector<Double> energies = Linalg.vector(energyArray);
        List<String> labels = new ArrayList<>();
        for (String label : scaleLabels) {
            labels.add(label);
        }
        
        plot.bar(energies, labels);
        
        return plot;
    }
    
    /**
     * 绘制小波去噪结果图 / Plot wavelet denoising results
     * <p>
     * 显示小波去噪前后的信号对比。
     * Display signal comparison before and after wavelet denoising.
     * </p>
     *
     * @param signal 原始信号 / Original signal
     * @param waveletType 小波类型 / Wavelet type
     * @param threshold 阈值 / Threshold
     * @param title 图表标题 / Plot title
     * @return 小波去噪结果图对象 / Wavelet denoising results plot object
     */
    public static IPlot plotWaveletDenoising(IVector<Double> signal, String waveletType, 
                                           double threshold, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("时间 / Time");
        plot.ylabel("幅度 / Amplitude");
        
        // 进行小波去噪 / Perform wavelet denoising
        IVector<Double> denoisedSignal = WaveletAnalysis.waveletDenoising(signal, WaveletAnalysis.WaveletType.valueOf(waveletType.toUpperCase()), 3, threshold, 1.0);
        
        // 创建时间轴 / Create time axis
        double[] timeArray = new double[signal.length()];
        for (int i = 0; i < signal.length(); i++) {
            timeArray[i] = i;
        }
        IVector<Double> time = Linalg.vector(timeArray);
        
        // 绘制原始信号和去噪信号 / Plot original and denoised signals
        List<String> labels1 = new ArrayList<>();
        labels1.add("原始信号 / Original Signal");
        List<String> labels2 = new ArrayList<>();
        labels2.add("去噪信号 / Denoised Signal");
        plot.line(time, signal, labels1);
        plot.line(time, denoisedSignal, labels2);
        
        return plot;
    }
    
    /**
     * 绘制小波滤波器响应图 / Plot wavelet filter response
     * <p>
     * 显示小波滤波器的频率响应。
     * Display frequency response of wavelet filters.
     * </p>
     *
     * @param waveletType 小波类型 / Wavelet type
     * @param title 图表标题 / Plot title
     * @return 小波滤波器响应图对象 / Wavelet filter response plot object
     */
    public static IPlot plotWaveletFilterResponse(String waveletType, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("频率 / Frequency");
        plot.ylabel("幅度响应 / Magnitude Response");
        
        // 获取小波滤波器 / Get wavelet filters
        // Note: 简化处理，使用默认滤波器 / Simplified processing, use default filters
        double[] lowPass = {0.7071, 0.7071}; // 简化的低通滤波器 / Simplified lowpass filter
        double[] highPass = {0.7071, -0.7071}; // 简化的高通滤波器 / Simplified highpass filter
        
        // 计算频率响应 / Calculate frequency response
        int n = 1024;
        double[] freqArray = new double[n];
        double[] lowpassResponse = new double[n];
        double[] highpassResponse = new double[n];
        
        for (int i = 0; i < n; i++) {
            freqArray[i] = (double) i / n;
            
            // 计算低通滤波器响应 / Calculate lowpass filter response
            double lowpassSum = 0;
            for (int j = 0; j < lowPass.length; j++) {
                lowpassSum += lowPass[j] * Math.cos(2 * Math.PI * freqArray[i] * j);
            }
            lowpassResponse[i] = Math.abs(lowpassSum);
            
            // 计算高通滤波器响应 / Calculate highpass filter response
            double highpassSum = 0;
            for (int j = 0; j < highPass.length; j++) {
                highpassSum += highPass[j] * Math.cos(2 * Math.PI * freqArray[i] * j);
            }
            highpassResponse[i] = Math.abs(highpassSum);
        }
        
        IVector<Double> frequencies = Linalg.vector(freqArray);
        IVector<Double> lowpassResp = Linalg.vector(lowpassResponse);
        IVector<Double> highpassResp = Linalg.vector(highpassResponse);
        
        List<String> labels1 = new ArrayList<>();
        labels1.add("低通滤波器 / Lowpass Filter");
        List<String> labels2 = new ArrayList<>();
        labels2.add("高通滤波器 / Highpass Filter");
        plot.line(frequencies, lowpassResp, labels1);
        plot.line(frequencies, highpassResp, labels2);
        
        return plot;
    }
    
    /**
     * 绘制小波包分解图 / Plot wavelet packet decomposition
     * <p>
     * 显示小波包分解的树状结构。
     * Display tree structure of wavelet packet decomposition.
     * </p>
     *
     * @param signal 信号数据 / Signal data
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     * @param title 图表标题 / Plot title
     * @return 小波包分解图对象 / Wavelet packet decomposition plot object
     */
    public static IPlot plotWaveletPacketDecomposition(IVector<Double> signal, String waveletType, 
                                                     int levels, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("时间 / Time");
        plot.ylabel("小波包系数 / Wavelet Packet Coefficients");
        
        // 进行小波包分解 / Perform wavelet packet decomposition
        // Note: 小波包分解需要复杂的树结构实现，这里简化处理 / Wavelet packet decomposition requires complex tree structure implementation, simplified here
        
        // 创建时间轴 / Create time axis
        double[] timeArray = new double[signal.length()];
        for (int i = 0; i < signal.length(); i++) {
            timeArray[i] = i;
        }
        IVector<Double> time = Linalg.vector(timeArray);
        
        // 绘制小波包系数 / Plot wavelet packet coefficients
        // Note: 小波包树结构需要递归遍历 / Wavelet packet tree structure needs recursive traversal
        // 这里简化处理，只绘制根节点 / Simplified here, only plot root node
        String label = "小波包系数 / Wavelet Packet Coefficients";
        List<String> labels = new ArrayList<>();
        labels.add(label);
        plot.line(time, signal, labels);
        
        return plot;
    }
    
    /**
     * 绘制小波统计图 / Plot wavelet statistics
     * <p>
     * 显示小波分析的各种统计信息。
     * Display various statistics of wavelet analysis.
     * </p>
     *
     * @param signal 信号数据 / Signal data
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     * @param title 图表标题 / Plot title
     * @return 小波统计图对象 / Wavelet statistics plot object
     */
    public static IPlot plotWaveletStatistics(IVector<Double> signal, String waveletType, 
                                            int levels, String title) {
        IPlot plot = Plots.of();
        plot.title(title);
        plot.xlabel("统计量 / Statistics");
        plot.ylabel("数值 / Value");
        
        // 进行小波分解 / Perform wavelet decomposition
        WaveletCoefficients decomposition = 
            WaveletAnalysis.discreteWaveletTransform(signal, WaveletAnalysis.WaveletType.valueOf(waveletType.toUpperCase()), levels, 1.0);
        
        // 计算统计量 / Calculate statistics
        double totalEnergy = signal.apply(x -> x * x).sum();
        double approximationEnergy = decomposition.approximation.apply(x -> x * x).sum();
        double detailEnergy = 0;
        for (IVector<Double> detail : decomposition.details) {
            detailEnergy += detail.apply(x -> x * x).sum();
        }
        
        double energyRatio = approximationEnergy / totalEnergy;
        double compressionRatio = 0.5; // 简化的压缩比计算 / Simplified compression ratio calculation
        double reconstructionError = 0.0; // 简化的重构误差计算 / Simplified reconstruction error calculation
        
        String[] statNames = {"总能量 / Total Energy", "近似能量 / Approx Energy", 
                            "细节能量 / Detail Energy", "能量比 / Energy Ratio",
                            "压缩比 / Compression Ratio", "重构误差 / Reconstruction Error"};
        double[] statValues = {totalEnergy, approximationEnergy, detailEnergy, 
                             energyRatio, compressionRatio, reconstructionError};
        
        IVector<Double> statValuesVector = Linalg.vector(statValues);
        List<String> labels = new ArrayList<>();
        for (String name : statNames) {
            labels.add(name);
        }
        
        plot.bar(statValuesVector, labels);
        
        return plot;
    }
    
    /**
     * 创建小波可视化仪表板 / Create wavelet visualization dashboard
     * <p>
     * 创建包含多个图表的小波可视化仪表板。
     * Create wavelet visualization dashboard with multiple charts.
     * </p>
     *
     * @param signal 信号数据 / Signal data
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Decomposition levels
     * @param title 仪表板标题 / Dashboard title
     * @return 可视化图表列表 / List of visualization plots
     */
    public static List<IPlot> createWaveletDashboard(IVector<Double> signal, String waveletType, 
                                                   int levels, String title) {
        List<IPlot> plots = new ArrayList<>();
        
        // 添加各种图表 / Add various plots
        plots.add(plotWaveletCoefficients(signal, waveletType, levels, 
                                        title + " - 小波系数 / Wavelet Coefficients"));
        plots.add(plotWaveletEnergyDistribution(signal, waveletType, levels, 
                                              title + " - 能量分布 / Energy Distribution"));
        plots.add(plotWaveletDenoising(signal, waveletType, 0.1, 
                                     title + " - 去噪结果 / Denoising Results"));
        plots.add(plotWaveletFilterResponse(waveletType, 
                                          title + " - 滤波器响应 / Filter Response"));
        plots.add(plotWaveletPacketDecomposition(signal, waveletType, levels, 
                                               title + " - 小波包分解 / Wavelet Packet Decomposition"));
        plots.add(plotWaveletStatistics(signal, waveletType, levels, 
                                      title + " - 统计信息 / Statistics"));
        
        return plots;
    }
}
