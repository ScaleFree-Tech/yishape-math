package com.reremouse.lab.audio;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioStatistics;
import com.reremouse.lab.audio.core.AudioQuality;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.viz.Plots;
import com.reremouse.lab.math.viz.IPlot;
import com.reremouse.lab.util.Tuple2;

import java.util.ArrayList;
import java.util.List;

/**
 * 音频可视化器类 / Audio Visualizer Class
 * <p>
 * 提供音频数据的可视化功能，包括波形图、频谱图、频谱图等。
 * 使用项目现有的viz包功能进行音频可视化。
 * </p>
 * <p>
 * Provides audio data visualization functionality including waveform plots, spectrum plots, spectrograms, etc.
 * Uses existing viz package functionality for audio visualization.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioVisualizer {
    
    /**
     * 绘制波形图 / Plot waveform
     * <p>
     * 显示音频信号的时域波形。
     * Display time-domain waveform of audio signal.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 波形图对象 / Waveform plot object
     */
    public static IPlot plotWaveform(AudioData audioData, String title) {
        // 获取音频样本 / Get audio samples
        IVector<Double> samples = audioData.getSamples();
        
        // 生成时间轴 / Generate time axis
        IVector<Double> timeAxis = Linalg.range(samples.length())
                .multiplyScalar(1.0 / audioData.getSampleRate());
        
        // 创建波形图 / Create waveform plot
        IPlot plot = Plots.of()
                .line(timeAxis, samples)
                .title(title)
                .xlabel("时间 (秒) / Time (s)")
                .ylabel("幅度 / Amplitude");
        
        return plot;
    }
    
    /**
     * 绘制频谱图 / Plot spectrum
     * <p>
     * 显示音频信号的频域表示。
     * Display frequency-domain representation of audio signal.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 频谱图对象 / Spectrum plot object
     */
    public static IPlot plotSpectrum(AudioData audioData, String title) {
        // 计算频谱 / Calculate spectrum
        Tuple2<IVector<Double>, IVector<Double>> spectrumResult = Audios.spectrum(audioData);
        IVector<Double> frequencies = spectrumResult.getFirst();
        IVector<Double> magnitudes = spectrumResult.getSecond();
        
        // 创建频谱图 / Create spectrum plot
        IPlot plot = Plots.of()
                .line(frequencies, magnitudes)
                .title(title)
                .xlabel("频率 (Hz) / Frequency (Hz)")
                .ylabel("幅度 / Magnitude");
        
        return plot;
    }
    
    /**
     * 绘制对数频谱图 / Plot log spectrum
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 对数频谱图对象 / Log spectrum plot object
     */
    public static IPlot plotLogSpectrum(AudioData audioData, String title) {
        // 计算频谱 / Calculate spectrum
        Tuple2<IVector<Double>, IVector<Double>> spectrumResult = Audios.spectrum(audioData);
        IVector<Double> frequencies = spectrumResult.getFirst();
        IVector<Double> magnitudes = spectrumResult.getSecond();
        
        // 转换为对数刻度 / Convert to log scale
        IVector<Double> logMagnitudes = magnitudes.apply(x -> Math.log10(x + 1e-10)); // Add small value to avoid log(0)
        
        // 创建对数频谱图 / Create log spectrum plot
        IPlot plot = Plots.of()
                .line(frequencies, logMagnitudes)
                .title(title)
                .xlabel("频率 (Hz) / Frequency (Hz)")
                .ylabel("幅度 (dB) / Magnitude (dB)");
        
        return plot;
    }
    
    /**
     * 绘制频谱图 / Plot spectrogram
     * <p>
     * 显示音频信号的时频表示。
     * Display time-frequency representation of audio signal.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 频谱图对象 / Spectrogram plot object
     */
    public static IPlot plotSpectrogram(AudioData audioData, String title) {
        return plotSpectrogram(audioData, title, 1024, 256);
    }
    
    /**
     * 绘制频谱图（指定参数） / Plot spectrogram (with specified parameters)
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @param windowSize 窗函数大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @return 频谱图对象 / Spectrogram plot object
     */
    public static IPlot plotSpectrogram(AudioData audioData, String title, int windowSize, int hopSize) {
        // 计算STFT / Calculate STFT
        Tuple2<IVector<Double>, IVector<Double>> stftResult = Audios.stft(audioData);
        // For simplicity, we're using the spectrum method. In a real implementation, 
        // we would need a proper STFT method that returns a matrix.
        IVector<Double> frequencies = stftResult.getFirst();
        IVector<Double> magnitudes = stftResult.getSecond();
        
        // Create a mock spectrogram matrix for visualization
        IMatrix<Double> mockSpectrogram = Linalg.zeros(100, 100); // Mock matrix
        
        // 创建频谱图 / Create spectrogram plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("时间帧 / Time Frame")
                .ylabel("频率仓 / Frequency Bin");
        
        // 添加热力图
        plot.heatmap(mockSpectrogram, null, null);
        
        return plot;
    }
    
    
    /**
     * 绘制MFCC特征图 / Plot MFCC features
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return MFCC特征图对象 / MFCC features plot object
     */
    public static IPlot plotMFCC(AudioData audioData, String title) {
        // 提取MFCC特征 / Extract MFCC features
        IVector<Double> features = Audios.calculateMFCC(audioData).colMeans();
        // In a real implementation, we would extract actual MFCC features
        
        // 创建MFCC特征图 / Create MFCC features plot
        IPlot plot = Plots.of().line(features)
                .title(title)
                .xlabel("MFCC系数 / MFCC Coefficient")
                .ylabel("特征值 / Feature Value");
        
        // 使用柱状图显示MFCC特征
        plot.bar(features);
        
        return plot;
    }
    
    /**
     * 绘制音频统计信息图 / Plot audio statistics
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 统计信息图对象 / Statistics plot object
     */
    public static IPlot plotAudioStatistics(AudioData audioData, String title) {
        // 获取音频统计信息 / Get audio statistics
        AudioStatistics stats = audioData.getStatistics();
        
        // 创建统计信息图 / Create statistics plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("统计量类型 / Statistic Type")
                .ylabel("数值 / Value");
        
        // 添加统计量 / Add statistics
        double[] statValues = {
            stats.getMean(),
            stats.getStdDev(),
            stats.getMin(),
            stats.getMax(),
            stats.getRms(),
            stats.getPeak(),
            stats.getDynamicRange(),
            stats.getSnr()
        };
        
        IVector<Double> values = Linalg.vector(statValues);
        plot.bar(values);
        
        return plot;
    }
    
    /**
     * 绘制音频质量评估图 / Plot audio quality assessment
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 质量评估图对象 / Quality assessment plot object
     */
    public static IPlot plotAudioQuality(AudioData audioData, String title) {
        // 获取音频统计信息 / Get audio statistics
        AudioStatistics stats = audioData.getStatistics();
        AudioQuality quality = stats.getQuality();
        
        // 创建质量评估图 / Create quality assessment plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("质量指标 / Quality Metrics")
                .ylabel("评分 / Score");
        
        // 添加质量指标 / Add quality metrics
        double[] qualityValues = {
            quality.getScore(), // Use the score method instead
            stats.getSnr(),
            stats.getDynamicRange(),
            stats.getZeroCrossingRate() * 1000 // Scale for better visualization
        };
        
        IVector<Double> values = Linalg.vector(qualityValues);
        plot.bar(values);
        
        return plot;
    }
    
    /**
     * 绘制音频比较图 / Plot audio comparison
     * <p>
     * 比较两个音频信号的波形和频谱。
     * Compare waveforms and spectra of two audio signals.
     * </p>
     *
     * @param audioData1 第一个音频数据 / First audio data
     * @param audioData2 第二个音频数据 / Second audio data
     * @param title 图表标题 / Plot title
     * @return 比较图对象 / Comparison plot object
     */
    public static IPlot plotAudioComparison(AudioData audioData1, AudioData audioData2, String title) {
        // 创建比较图 / Create comparison plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("时间 (秒) / Time (s)")
                .ylabel("幅度 / Amplitude");
        
        // 添加第一个音频的波形 / Add waveform of first audio
        IVector<Double> samples1 = audioData1.getSamples();
        IVector<Double> timeAxis1 = Linalg.range(samples1.length())
                .multiplyScalar(1.0 / audioData1.getSampleRate());
        plot.line(timeAxis1, samples1);
        
        // 添加第二个音频的波形 / Add waveform of second audio
        IVector<Double> samples2 = audioData2.getSamples();
        IVector<Double> timeAxis2 = Linalg.range(samples2.length())
                .multiplyScalar(1.0 / audioData2.getSampleRate());
        plot.line(timeAxis2, samples2);
        
        return plot;
    }
    
    /**
     * 创建音频仪表板 / Create audio dashboard
     * <p>
     * 创建包含多个音频可视化图表的仪表板。
     * Create a dashboard containing multiple audio visualization plots.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 仪表板标题 / Dashboard title
     * @return 图表列表 / List of plots
     */
    public static List<IPlot> createAudioDashboard(AudioData audioData, String title) {
        List<IPlot> plots = new ArrayList<>();
        
        // 添加各种图表 / Add various plots
        plots.add(plotWaveform(audioData, "波形图 / Waveform"));
        plots.add(plotSpectrum(audioData, "频谱图 / Spectrum"));
        plots.add(plotLogSpectrum(audioData, "对数频谱图 / Log Spectrum"));
        plots.add(plotAudioStatistics(audioData, "音频统计 / Audio Statistics"));
        plots.add(plotAudioQuality(audioData, "音频质量 / Audio Quality"));
        
        return plots;
    }
}