package com.reremouse.lab.math.audio;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.viz.Plots;
import com.reremouse.lab.math.viz.IPlot;

import java.util.ArrayList;
import java.util.List;

/**
 * 音频可视化器类 / Audio Visualizer Class
 * <p>
 * 提供音频数据的可视化功能，包括波形图、频谱图、频谱图等。
 * 使用项目现有的viz包功能进行可视化。
 * </p>
 * <p>
 * Provides audio data visualization functionality including waveform plots, spectrum plots, spectrograms, etc.
 * Uses existing viz package functionality for visualization.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioVisualizer {
    
    /**
     * 绘制音频波形图 / Plot audio waveform
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
     * 绘制多声道波形图 / Plot multi-channel waveform
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 波形图对象 / Waveform plot object
     */
    public static IPlot plotMultiChannelWaveform(AudioData audioData, String title) {
        IPlot plot = Plots.of().title(title).xlabel("时间 (秒) / Time (s)").ylabel("幅度 / Amplitude");
        
        // 为每个声道绘制波形 / Plot waveform for each channel
        for (int ch = 0; ch < audioData.getChannels(); ch++) {
            IVector<Double> channelData = audioData.getChannel(ch);
            IVector<Double> timeAxis = Linalg.range(channelData.length())
                    .multiplyScalar(1.0 / audioData.getSampleRate());
            
            String channelName = audioData.getChannels() == 1 ? "单声道 / Mono" : 
                               "声道 " + (ch + 1) + " / Channel " + (ch + 1);
            
            // 为多声道创建单独的线图
            List<String> hue = new ArrayList<>();
            hue.add(channelName);
            plot.line(timeAxis, channelData, hue);
        }
        
        return plot;
    }
    
    /**
     * 绘制频谱图 / Plot spectrum
     * <p>
     * 显示音频信号的频域特性。
     * Display frequency-domain characteristics of audio signal.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 频谱图对象 / Spectrum plot object
     */
    public static IPlot plotSpectrum(AudioData audioData, String title) {
        // 计算频谱 / Calculate spectrum
        var spectrumResult = AudioAnalyzer.calculateSpectrum(audioData);
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
        var spectrumResult = AudioAnalyzer.calculateSpectrum(audioData);
        IVector<Double> frequencies = spectrumResult.getFirst();
        IVector<Double> magnitudes = spectrumResult.getSecond();
        
        // 转换为对数刻度 / Convert to log scale
        IVector<Double> logMagnitudes = magnitudes.apply(x -> Math.log10(x));
        
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
        IMatrix<Double> stftMatrix = AudioAnalyzer.calculateSTFT(audioData, windowSize, hopSize);
        
        // 转换为对数刻度 / Convert to log scale
        IMatrix<Double> logStftMatrix = stftMatrix.log();
        
        // 创建频谱图 / Create spectrogram plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("时间帧 / Time Frame")
                .ylabel("频率仓 / Frequency Bin");
        
        // 添加热力图
        plot.heatmap(logStftMatrix, null, null);
        
        return plot;
    }
    
    /**
     * 绘制音频特征图 / Plot audio features
     * <p>
     * 显示音频的各种特征随时间的变化。
     * Display various audio features over time.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 特征图对象 / Features plot object
     */
    public static IPlot plotAudioFeatures(AudioData audioData, String title) {
        // 提取音频特征 / Extract audio features
        AudioFeatures features = AudioAnalyzer.extractFeatures(audioData);
        
        // 创建特征图 / Create features plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("特征类型 / Feature Type")
                .ylabel("特征值 / Feature Value");
        
        // 添加基本特征 / Add basic features
        double[] basicFeatures = {
            features.getSpectralCentroid(),
            features.getSpectralBandwidth(),
            features.getSpectralRolloff(),
            features.getZeroCrossingRate()
        };
        
        IVector<Double> featureValues = Linalg.vector(basicFeatures);
        
        plot.bar(featureValues);
        
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
        AudioFeatures features = AudioAnalyzer.extractFeatures(audioData);
        double[] mfcc = features.getMfcc();
        IVector<Double> mfccVector = Linalg.vector(mfcc);
        // 创建MFCC特征图 / Create MFCC features plot
        IPlot plot = Plots.of().line(mfccVector)
                .title(title)
                .xlabel("MFCC系数 / MFCC Coefficient")
                .ylabel("特征值 / Feature Value");
        
        
        // 使用柱状图显示MFCC特征
        plot.bar(mfccVector);
        
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
                .xlabel("质量指标 / Quality Metric")
                .ylabel("数值 / Value");
        
        // 添加质量指标 / Add quality metrics
        
        double[] metricValues = {
            stats.getSnr(),
            stats.getDynamicRange(),
            stats.getCrestFactor(),
            stats.getZeroCrossingRate(),
            quality.getScore()
        };
        
        IVector<Double> values = Linalg.vector(metricValues);
        
        plot.bar(values);
        
        return plot;
    }
    
    /**
     * 绘制音频比较图 / Plot audio comparison
     * <p>
     * 比较两个音频文件的特征。
     * Compare features of two audio files.
     * </p>
     *
     * @param audioData1 第一个音频数据 / First audio data
     * @param audioData2 第二个音频数据 / Second audio data
     * @param title 图表标题 / Plot title
     * @return 比较图对象 / Comparison plot object
     */
    public static IPlot plotAudioComparison(AudioData audioData1, AudioData audioData2, String title) {
        // 提取两个音频的特征 / Extract features of both audio files
        AudioFeatures features1 = AudioAnalyzer.extractFeatures(audioData1);
        AudioFeatures features2 = AudioAnalyzer.extractFeatures(audioData2);
        
        // 创建比较图 / Create comparison plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("特征类型 / Feature Type")
                .ylabel("特征值 / Feature Value");
        
        // 添加基本特征比较 / Add basic features comparison
        String[] featureNames = {
            "频谱质心 / Spectral Centroid",
            "频谱带宽 / Spectral Bandwidth",
            "频谱滚降 / Spectral Rolloff",
            "零交叉率 / Zero Crossing Rate"
        };
        
        double[] features1Values = {
            features1.getSpectralCentroid(),
            features1.getSpectralBandwidth(),
            features1.getSpectralRolloff(),
            features1.getZeroCrossingRate()
        };
        
        double[] features2Values = {
            features2.getSpectralCentroid(),
            features2.getSpectralBandwidth(),
            features2.getSpectralRolloff(),
            features2.getZeroCrossingRate()
        };
        
        IVector<Double> values1 = Linalg.vector(features1Values);
        IVector<Double> values2 = Linalg.vector(features2Values);
        
        // 创建分组柱状图数据
        List<String> labels = new ArrayList<>();
        for (int i = 0; i < featureNames.length; i++) {
            labels.add(featureNames[i]);
        }
        
        // 创建合并的数据用于分组柱状图
        double[] allValuesArray = new double[values1.length() + values2.length()];
        for (int i = 0; i < values1.length(); i++) {
            allValuesArray[i] = values1.get(i);
        }
        for (int i = 0; i < values2.length(); i++) {
            allValuesArray[values1.length() + i] = values2.get(i);
        }
        
        IVector<Double> allValues = Linalg.vector(allValuesArray);
        List<String> hue = new ArrayList<>();
        for (int i = 0; i < featureNames.length; i++) {
            hue.add("音频1 / Audio 1");
        }
        for (int i = 0; i < featureNames.length; i++) {
            hue.add("音频2 / Audio 2");
        }
        
        plot.bar(allValues, hue);
        
        return plot;
    }
    
    /**
     * 创建音频可视化仪表板 / Create audio visualization dashboard
     * <p>
     * 创建包含多个图表的音频可视化仪表板。
     * Create audio visualization dashboard with multiple charts.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 仪表板标题 / Dashboard title
     * @return 可视化图表列表 / List of visualization plots
     */
    public static List<IPlot> createAudioDashboard(AudioData audioData, String title) {
        List<IPlot> plots = new ArrayList<>();
        
        // 添加各种图表 / Add various plots
        plots.add(plotWaveform(audioData, title + " - 波形图 / Waveform"));
        plots.add(plotSpectrum(audioData, title + " - 频谱图 / Spectrum"));
        plots.add(plotLogSpectrum(audioData, title + " - 对数频谱图 / Log Spectrum"));
        plots.add(plotSpectrogram(audioData, title + " - 频谱图 / Spectrogram"));
        plots.add(plotAudioStatistics(audioData, title + " - 统计信息 / Statistics"));
        plots.add(plotAudioQuality(audioData, title + " - 质量评估 / Quality Assessment"));
        
        return plots;
    }
}
