package com.yishape.lab.audio;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.core.AudioStatistics;
import com.yishape.lab.audio.core.AudioQuality;
import com.yishape.lab.audio.core.AudioProcessor;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.util.Tuple2;

import com.yishape.lab.audio.analysis.STFTAnalyzer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

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
public class AudioPlots {

    private static final Logger log = LoggerFactory.getLogger(AudioPlots.class);

    
    // Maximum number of points to display in waveform for visualization
    private static final int MAX_WAVEFORM_POINTS = 10000;
    
    // Apply pre-emphasis filter to reduce high frequency artifacts
    private static final double PRE_EMPHASIS_FACTOR = 0.95;
    
    /**
     * 根据音频时长计算最大时间帧数 / Calculate maximum time frames based on audio duration
     * 
     * @param duration 音频时长（秒） / Audio duration (seconds)
     * @return 最大时间帧数 / Maximum time frames
     */
    private static int calculateMaxTimeFramesForAudio(double duration) {
        // 对于短音频（<30秒），使用较高的分辨率
        if (duration < 30) {
            return 800;
        }
        // 对于中等音频（30-120秒），使用中等分辨率
        else if (duration < 120) {
            return 1200;
        }
        // 对于长音频（>=120秒），使用较低的分辨率但仍保持足够的细节
        else {
            return Math.max(800, Math.min(2000, (int)(duration * 10))); // 每秒约10帧，但限制在800-2000之间
        }
    }
    
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
        return plotWaveform(audioData, title, 1024, 256);
    }
    
    /**
     * 绘制波形图（指定参数） / Plot waveform (with specified parameters)
     * <p>
     * 显示音频信号的时域波形。
     * Display time-domain waveform of audio signal.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @param windowSize 窗函数大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @return 波形图对象 / Waveform plot object
     */
    public static IPlot plotWaveform(AudioData audioData, String title, int windowSize, int hopSize) {
        // 确保使用单声道音频进行波形图绘制，与MFCC处理逻辑保持一致
        AudioData monoAudioData = audioData;
        if (!audioData.isMono()) {
            // 将立体声转换为单声道
            monoAudioData = AudioProcessor.stereoToMono(audioData);
        }
        
        // 获取音频样本 / Get audio samples
        IVector<Double> samples = monoAudioData.getSamples();
        
        // Apply pre-emphasis filter to reduce high frequency artifacts
        IVector<Double> filteredSamples = applyPreEmphasis(samples);
        
        // Downsample for visualization if needed
        IVector<Double> displaySamples = filteredSamples;
        IVector<Double> displayTimeAxis = null;
        
        // Generate time axis that covers the full duration of the audio
        double totalDuration = monoAudioData.getDuration();
        
        if (filteredSamples.length() > MAX_WAVEFORM_POINTS) {
            // Downsample to reduce the number of points for visualization
            displaySamples = downsampleForVisualization(filteredSamples, MAX_WAVEFORM_POINTS);
            
            // Generate time axis for downsampled data that covers the full duration
            displayTimeAxis = Linalg.range(displaySamples.length())
                    .multiplyScalar(totalDuration / displaySamples.length());
        } else {
            // Generate time axis for original data that covers the full duration
            displayTimeAxis = Linalg.range(filteredSamples.length())
                    .multiplyScalar(totalDuration / filteredSamples.length());
        }
        
        // 创建波形图 / Create waveform plot
        IPlot plot = Plots.of()
                .line(displayTimeAxis, displaySamples)
                .title(title)
                .xlabel("时间 (秒) / Time (s)")
                .ylabel("幅度 / Amplitude");
        
        return plot;
    }
    
    /**
     * Apply pre-emphasis filter to reduce high frequency artifacts
     * 
     * @param samples Input audio samples
     * @return Pre-emphasized samples
     */
    private static IVector<Double> applyPreEmphasis(IVector<Double> samples) {
        int length = samples.length();
        IVector<Double> filtered = Linalg.zeros(length);
        
        // Apply pre-emphasis: y[n] = x[n] - a * x[n-1]
        filtered.set(0, samples.get(0)); // First sample remains unchanged
        for (int i = 1; i < length; i++) {
            double filteredValue = samples.get(i) - PRE_EMPHASIS_FACTOR * samples.get(i - 1);
            filtered.set(i, filteredValue);
        }
        
        return filtered;
    }
    
    /**
     * 为可视化目的对大数据进行降采样 / Downsample large data for visualization purposes
     *
     * @param samples 原始样本数据 / Original sample data
     * @param targetPoints 目标点数 / Target number of points
     * @return 降采样后的数据 / Downsampled data
     */
    private static IVector<Double> downsampleForVisualization(IVector<Double> samples, int targetPoints) {
        int originalLength = samples.length();
        if (originalLength <= targetPoints) {
            return samples;
        }
        
        double[] downsampled = new double[targetPoints];
        int pointsPerBucket = originalLength / targetPoints;
        
        // Use average instead of min-max decimation to reduce noise
        for (int i = 0; i < targetPoints; i++) {
            int startIdx = i * pointsPerBucket;
            int endIdx = Math.min(startIdx + pointsPerBucket, originalLength);
            
            if (startIdx >= endIdx) {
                downsampled[i] = samples.get(Math.min(startIdx, originalLength - 1));
                continue;
            }
            
            // Calculate average in this bucket
            double sum = 0.0;
            int count = 0;
            for (int j = startIdx; j < endIdx; j++) {
                sum += samples.get(j);
                count++;
            }
            
            downsampled[i] = sum / count;
        }
        
        return Linalg.vector(downsampled);
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
        return plotSpectrum(audioData, title, "line");
    }
    
    /**
     * 绘制频谱图（指定图表类型） / Plot spectrum (with specified chart type)
     * <p>
     * 显示音频信号的频域表示。
     * Display frequency-domain representation of audio signal.
     * </p>
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @param chartType 图表类型 ("line" 或 "bar") / Chart type ("line" or "bar")
     * @return 频谱图对象 / Spectrum plot object
     */
    public static IPlot plotSpectrum(AudioData audioData, String title, String chartType) {
        // 计算频谱 / Calculate spectrum
        Tuple2<IVector<Double>, IVector<Double>> spectrumResult = Audios.spectrum(audioData);
        IVector<Double> frequencies = spectrumResult.getFirst();
        IVector<Double> magnitudes = spectrumResult.getSecond();
        
        // 创建频谱图 / Create spectrum plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("频率 (Hz) / Frequency (Hz)")
                .ylabel("幅度 / Magnitude");
        
        // 根据图表类型选择可视化方式 / Choose visualization method based on chart type
        if ("bar".equalsIgnoreCase(chartType)) {
            // 对于柱状图，我们可能需要减少数据点以避免过多的柱子
            int maxBars = 50; // 限制最大柱子数量
            if (frequencies.length() > maxBars) {
                Tuple2<IVector<Double>, IVector<Double>> downsampled = downsampleSpectrumData(frequencies, magnitudes, maxBars);
                frequencies = downsampled.getFirst();
                magnitudes = downsampled.getSecond();
            }
            // 创建频率标签列表
            List<String> frequencyLabels = new ArrayList<>();
            for (int i = 0; i < frequencies.length(); i++) {
                frequencyLabels.add(String.format("%.0f", frequencies.get(i)));
            }
            // 使用带标签的柱状图
            plot.bar(frequencyLabels, magnitudes);
        } else {
            // 默认使用线图 / Default to line chart
            plot.line(frequencies, magnitudes);
        }
        
        return plot;
    }
    
    /**
     * 降采样频谱数据 / Downsample spectrum data
     *
     * @param frequencies 频率向量 / Frequency vector
     * @param magnitudes 幅度向量 / Magnitude vector
     * @param targetPoints 目标点数 / Target number of points
     * @return 降采样后的数据 / Downsampled data
     */
    private static Tuple2<IVector<Double>, IVector<Double>> downsampleSpectrumData(
            IVector<Double> frequencies, IVector<Double> magnitudes, int targetPoints) {
        int originalLength = frequencies.length();
        if (originalLength <= targetPoints) {
            return new Tuple2<>(frequencies, magnitudes);
        }
        
        double[] downsampledFreq = new double[targetPoints];
        double[] downsampledMag = new double[targetPoints];
        int pointsPerBucket = originalLength / targetPoints;
        
        for (int i = 0; i < targetPoints; i++) {
            int startIdx = i * pointsPerBucket;
            int endIdx = Math.min(startIdx + pointsPerBucket, originalLength);
            
            if (startIdx >= endIdx) {
                downsampledFreq[i] = frequencies.get(Math.min(startIdx, originalLength - 1));
                downsampledMag[i] = magnitudes.get(Math.min(startIdx, originalLength - 1));
                continue;
            }
            
            // 计算频率和幅度的平均值 / Calculate average of frequency and magnitude
            double freqSum = 0;
            double magSum = 0;
            for (int j = startIdx; j < endIdx; j++) {
                freqSum += frequencies.get(j);
                magSum += magnitudes.get(j);
            }
            downsampledFreq[i] = freqSum / (endIdx - startIdx);
            downsampledMag[i] = magSum / (endIdx - startIdx);
        }
        
        return new Tuple2<>(Linalg.vector(downsampledFreq), Linalg.vector(downsampledMag));
    }
    
    /**
     * 绘制对数频谱图 / Plot log spectrum
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return 对数频谱图对象 / Log spectrum plot object
     */
    public static IPlot plotLogSpectrum(AudioData audioData, String title) {
        return plotLogSpectrum(audioData, title, "line");
    }
    
    /**
     * 绘制对数频谱图（指定图表类型） / Plot log spectrum (with specified chart type)
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @param chartType 图表类型 ("line" 或 "bar") / Chart type ("line" or "bar")
     * @return 对数频谱图对象 / Log spectrum plot object
     */
    public static IPlot plotLogSpectrum(AudioData audioData, String title, String chartType) {
        // 计算频谱 / Calculate spectrum
        Tuple2<IVector<Double>, IVector<Double>> spectrumResult = Audios.spectrum(audioData);
        IVector<Double> frequencies = spectrumResult.getFirst();
        IVector<Double> magnitudes = spectrumResult.getSecond();
        
        // 转换为对数刻度 / Convert to log scale
        IVector<Double> logMagnitudes = magnitudes.apply(x -> Math.log10(x + 1e-10)); // Add small value to avoid log(0)
        
        // 创建对数频谱图 / Create log spectrum plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("频率 (Hz) / Frequency (Hz)")
                .ylabel("幅度 (dB) / Magnitude (dB)");
        
        // 根据图表类型选择可视化方式 / Choose visualization method based on chart type
        if ("bar".equalsIgnoreCase(chartType)) {
            // 对于柱状图，我们可能需要减少数据点以避免过多的柱子
            int maxBars = 50; // 限制最大柱子数量
            if (frequencies.length() > maxBars) {
                Tuple2<IVector<Double>, IVector<Double>> downsampled = downsampleSpectrumData(frequencies, logMagnitudes, maxBars);
                frequencies = downsampled.getFirst();
                logMagnitudes = downsampled.getSecond();
            }
            // 创建频率标签列表
            List<String> frequencyLabels = new ArrayList<>();
            for (int i = 0; i < frequencies.length(); i++) {
                frequencyLabels.add(String.format("%.0f", frequencies.get(i)));
            }
            // 使用带标签的柱状图
            plot.bar(frequencyLabels, logMagnitudes);
        } else {
            // 默认使用线图 / Default to line chart
            plot.line(frequencies, logMagnitudes);
        }
        
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
        try {
            // 创建STFT分析器并计算 spectrogram / Create STFT analyzer and calculate spectrogram
            STFTAnalyzer stftAnalyzer = new STFTAnalyzer();
            Map<String, Object> params = new HashMap<>();
            params.put("windowSize", windowSize);
            params.put("hopSize", hopSize);
            
            // 计算STFT矩阵 / Calculate STFT matrix
            IMatrix<Double> stftMatrix = stftAnalyzer.calculateSTFT(audioData, windowSize, hopSize);
            
            // 限制STFT矩阵的大小以减少文件大小 / Limit STFT matrix size to reduce file size
            // 调整参数以更好地支持长音频文件 / Adjust parameters to better support long audio files
            // 对于长音频文件，动态调整时间帧数限制以显示更多内容
            int maxTimeFrames = calculateMaxTimeFramesForAudio(audioData.getDuration());
            int maxFreqBins = 256;     // 增加频率仓数限制 / Increase frequency bin limit
            
            if (stftMatrix.cols() > maxTimeFrames || stftMatrix.rows() > maxFreqBins) {
                stftMatrix = downsampleSpectrogramMatrix(stftMatrix, maxFreqBins, maxTimeFrames);
            }
            
            // 创建时间标签 / Create time labels
            int numFrames = stftMatrix.cols();
            List<String> timeLabels = new ArrayList<>();
            double hopTime = (double) hopSize / audioData.getSampleRate();
            // 修正时间轴，使其与音频的实际时间对齐，确保覆盖整个音频时长
            double startTime = (double) windowSize / (2 * audioData.getSampleRate()); // 窗口中心时间
            for (int i = 0; i < numFrames; i++) {
                // 使用等比例映射确保时间标签覆盖整个音频时长
                double progress = numFrames > 1 ? (double) i / (numFrames - 1) : 0;
                double totalTime = audioData.getDuration();
                double time = progress * totalTime;
                timeLabels.add(String.format("%.2f", time));
            }
            
            // 创建频率标签 / Create frequency labels
            int numBins = stftMatrix.rows();
            List<String> freqLabels = new ArrayList<>();
            double freqResolution = audioData.getSampleRate() / (2.0 * (windowSize / 2));
            for (int i = 0; i < numBins; i++) {
                freqLabels.add(String.format("%.0f", i * freqResolution));
            }
            
            // 创建频谱图 / Create spectrogram plot
            IPlot plot = Plots.of()
                    .title(title)
                    .xlabel("时间 (秒) / Time (s)")
                    .ylabel("频率 (Hz) / Frequency (Hz)");
            
            // 添加热力图 / Add heatmap
            plot.heatmap(stftMatrix, timeLabels, freqLabels);
            
            return plot;
        } catch (Exception e) {
            // 如果STFT计算失败，回退到原来的mock实现并记录错误
            // If STFT calculation fails, fall back to the original mock implementation and log the error
            log.warn("Failed to calculate spectrogram, using mock data: " + e.getMessage());
            log.error("exception", e);
            
            // 计算STFT / Calculate STFT
            Tuple2<IVector<Double>, IVector<Double>> stftResult = Audios.stft(audioData);
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
    }
    
    /**
     * 对频谱图矩阵进行降采样 / Downsample spectrogram matrix
     *
     * @param original 原始矩阵 / Original matrix
     * @param maxRows 最大行数 / Maximum rows
     * @param maxCols 最大列数 / Maximum columns
     * @return 降采样后的矩阵 / Downsampled matrix
     */
    private static IMatrix<Double> downsampleSpectrogramMatrix(IMatrix<Double> original, int maxRows, int maxCols) {
        int originalRows = original.rows();
        int originalCols = original.cols();
        
        // 如果原始矩阵已经小于或等于最大尺寸，则直接返回 / If original matrix is already smaller than or equal to max size, return directly
        if (originalRows <= maxRows && originalCols <= maxCols) {
            return original;
        }
        
        // 计算降采样因子 / Calculate downsampling factors
        int rowStep = Math.max(1, (int) Math.ceil((double) originalRows / maxRows));
        int colStep = Math.max(1, (int) Math.ceil((double) originalCols / maxCols));
        
        // 计算新尺寸 / Calculate new dimensions
        int newRows = Math.max(1, originalRows / rowStep);
        int newCols = Math.max(1, originalCols / colStep);
        
        // 创建新的降采样矩阵 / Create new downsampled matrix
        IMatrix<Double> downsampled = Linalg.zeros(newRows, newCols);
        
        // 执行降采样，使用平均值而不是单一采样点 / Perform downsampling using average instead of single sample point
        for (int i = 0; i < newRows; i++) {
            for (int j = 0; j < newCols; j++) {
                int origRowStart = i * rowStep;
                int origRowEnd = Math.min(origRowStart + rowStep, originalRows);
                int origColStart = j * colStep;
                int origColEnd = Math.min(origColStart + colStep, originalCols);
                
                // 计算区块平均值 / Calculate block average
                double sum = 0;
                int count = 0;
                for (int r = origRowStart; r < origRowEnd; r++) {
                    for (int c = origColStart; c < origColEnd; c++) {
                        sum += original.get(r, c);
                        count++;
                    }
                }
                
                // 设置平均值 / Set average value
                downsampled.set(i, j, count > 0 ? sum / count : 0);
            }
        }
        
        return downsampled;
    }
    
    /**
     * 绘制MFCC特征图（带异常值处理） / Plot MFCC features (with outlier handling)
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @param mfccCount MFCC系数数量 / Number of MFCC coefficients
     * @param windowSize 窗函数大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @param useNoiseReduction 是否使用噪声去除 / Whether to use noise reduction
     * @param outlierThreshold 异常值阈值（标准差倍数） / Outlier threshold (number of standard deviations)
     * @return MFCC特征图对象 / MFCC features plot object
     */
    public static IPlot plotMFCC(AudioData audioData, String title, int mfccCount, int windowSize, int hopSize, 
                                boolean useNoiseReduction, double outlierThreshold) {
        // 可选：在MFCC计算前应用噪声去除
        AudioData processedAudio = audioData;
        if (useNoiseReduction) {
            // 应用噪声去除
            processedAudio = Audios.reduceNoise(audioData, 0.05);
        }
        
        // 提取MFCC特征 / Extract MFCC features
        IMatrix<Double> mfccMatrix = Audios.calculateMFCC(processedAudio, mfccCount, windowSize, hopSize);
        
        // 处理异常值
        if (outlierThreshold > 0) {
            mfccMatrix = removeOutliers(mfccMatrix, outlierThreshold);
        }
        
        // 限制MFCC矩阵的大小以减少文件大小 / Limit MFCC matrix size to reduce file size
        // 调整参数以更好地支持长音频文件 / Adjust parameters to better support long audio files
        // 对于长音频文件，动态调整时间帧数限制以显示更多内容
        int maxTimeFrames = calculateMaxTimeFramesForAudio(processedAudio.getDuration());
        
        // 修复：正确检查时间帧数（行数）而不是MFCC系数数（列数）
        if (mfccMatrix.rows() > maxTimeFrames) {
            // 对MFCC矩阵进行降采样，保持系数维度不变 / Downsample MFCC matrix, keep coefficient dimension unchanged
            int newRows = maxTimeFrames;
            int oldRows = mfccMatrix.rows();
            // 使用等间距采样而不是简单的步长采样，以更好地覆盖整个音频
            IMatrix<Double> downsampled = Linalg.zeros(newRows, mfccMatrix.cols());
            for (int i = 0; i < newRows; i++) {
                // 计算在原始矩阵中的对应位置，确保覆盖整个时间范围
                int origRow = (int) Math.round((double) i * (oldRows - 1) / (newRows - 1));
                origRow = Math.min(origRow, oldRows - 1); // 确保不越界
                for (int j = 0; j < mfccMatrix.cols(); j++) {
                    downsampled.set(i, j, mfccMatrix.get(origRow, j));
                }
            }
            mfccMatrix = downsampled;
        }
        
        // 创建时间标签 / Create time labels
        // 修复：正确使用行数作为时间帧数
        int numFrames = mfccMatrix.rows();
        List<String> timeLabels = new ArrayList<>();
        double hopTime = (double) hopSize / processedAudio.getSampleRate();
        // 修正时间轴，使其与音频的实际时间对齐
        double startTime = (double) windowSize / (2 * processedAudio.getSampleRate()); // 窗口中心时间
        for (int i = 0; i < numFrames; i++) {
            // 使用等比例映射确保时间标签覆盖整个音频时长
            double progress = numFrames > 1 ? (double) i / (numFrames - 1) : 0;
            double totalTime = processedAudio.getDuration();
            double time = progress * totalTime;
            timeLabels.add(String.format("%.2f", time));
        }
        
        // 创建MFCC系数标签 / Create MFCC coefficient labels
        List<String> mfccLabels = new ArrayList<>();
        for (int i = 0; i < mfccCount; i++) {
            mfccLabels.add("MFCC " + i);
        }
        
        // 创建MFCC特征图 / Create MFCC features plot
        IPlot plot = Plots.of()
                .title(title)
                .xlabel("时间 (秒) / Time (s)")
                .ylabel("MFCC系数 / MFCC Coefficient");
        
        // 使用热力图显示MFCC特征矩阵 / Use heatmap to display MFCC feature matrix
        plot.heatmap(mfccMatrix.t().toFloatMatrix(), timeLabels, mfccLabels);
        
        return plot;
    }
    
    /**
     * 移除矩阵中的异常值 / Remove outliers from matrix
     * 
     * @param matrix 输入矩阵 / Input matrix
     * @param threshold 标准差倍数阈值 / Standard deviation multiplier threshold
     * @return 处理后的矩阵 / Processed matrix
     */
    private static IMatrix<Double> removeOutliers(IMatrix<Double> matrix, double threshold) {
        // 计算矩阵的均值和标准差
        double mean = matrix.mean().doubleValue();
        double std = matrix.std().doubleValue();
        
        // 定义异常值范围
        double lowerBound = mean - threshold * std;
        double upperBound = mean + threshold * std;
        
        // 创建新的矩阵存储处理后的数据
        IMatrix<Double> processedMatrix = Linalg.zeros(matrix.rows(), matrix.cols());
        
        // 处理每个元素
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                double value = matrix.get(i, j).doubleValue();
                // 将异常值替换为边界值
                if (value < lowerBound) {
                    processedMatrix.set(i, j, lowerBound);
                } else if (value > upperBound) {
                    processedMatrix.set(i, j, upperBound);
                } else {
                    processedMatrix.set(i, j, value);
                }
            }
        }
        
        return processedMatrix;
    }

    /**
     * 绘制MFCC特征图 / Plot MFCC features
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @return MFCC特征图对象 / MFCC features plot object
     */
    public static IPlot plotMFCC(AudioData audioData, String title) {
        // 默认启用异常值处理，阈值设为2.0个标准差
        return plotMFCC(audioData, title, 13, 1024, 256, true, 2.0);
    }
    
    /**
     * 绘制MFCC特征图（指定参数） / Plot MFCC features (with specified parameters)
     *
     * @param audioData 音频数据 / Audio data
     * @param title 图表标题 / Plot title
     * @param mfccCount MFCC系数数量 / Number of MFCC coefficients
     * @param windowSize 窗函数大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @return MFCC特征图对象 / MFCC features plot object
     */
    public static IPlot plotMFCC(AudioData audioData, String title, int mfccCount, int windowSize, int hopSize) {
        // 默认启用异常值处理，阈值设为2.0个标准差
        return plotMFCC(audioData, title, mfccCount, windowSize, hopSize, true, 2.0);
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
        Double[] statValues = {
            stats.getMean(),
            stats.getStdDev(),
            stats.getMin(),
            stats.getMax(),
            stats.getRms(),
            stats.getPeak(),
            stats.getDynamicRange(),
            stats.getSnr()
        };
        
        // 创建对应的标签 / Create corresponding labels
        List<String> statLabels = Arrays.asList(
            "均值 / Mean",
            "标准差 / Std Dev",
            "最小值 / Min",
            "最大值 / Max",
            "RMS",
            "峰值 / Peak",
            "动态范围 / Dynamic Range",
            "信噪比 / SNR"
        );
        
        IVector<Double> values = Linalg.vector(statValues);
        // 使用标签创建带标签的柱状图 / Create bar chart with labels
        plot.bar(statLabels,values);
        
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
        Double[] qualityValues = {
            (double) quality.getScore(), // Convert int to Double
            stats.getSnr(),
            stats.getDynamicRange(),
            stats.getZeroCrossingRate() * 1000 // Scale for better visualization
        };
        
        // 创建对应的标签 / Create corresponding labels
        List<String> qualityLabels = Arrays.asList(
            "总体评分 / Overall Score",
            "信噪比 / SNR",
            "动态范围 / Dynamic Range",
            "过零率 (x1000) / Zero Crossing Rate (x1000)"
        );
        
        IVector<Double> values = Linalg.vector(qualityValues);
        // 使用标签创建带标签的柱状图 / Create bar chart with labels
        plot.bar(qualityLabels,values);
        
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
        
        // 确保使用单声道音频进行波形图绘制，与MFCC处理逻辑保持一致
        AudioData monoAudioData1 = audioData1;
        if (!audioData1.isMono()) {
            // 将立体声转换为单声道
            monoAudioData1 = AudioProcessor.stereoToMono(audioData1);
        }
        
        // 添加第一个音频的波形 / Add waveform of first audio
        IVector<Double> samples1 = monoAudioData1.getSamples();
        IVector<Double> timeAxis1 = null;
        
        // Using default parameters for comparison
        int windowSize = 1024;
        int hopSize = 256;
        
        // Generate time axis that covers the full duration of the audio
        double totalDuration1 = monoAudioData1.getDuration();
        
        if (samples1.length() > MAX_WAVEFORM_POINTS) {
            IVector<Double> displaySamples1 = downsampleForVisualization(samples1, MAX_WAVEFORM_POINTS);
            // Generate time axis that covers the full duration
            timeAxis1 = Linalg.range(displaySamples1.length())
                    .multiplyScalar(totalDuration1 / displaySamples1.length());
            plot.line(timeAxis1, displaySamples1);
        } else {
            // Generate time axis that covers the full duration
            timeAxis1 = Linalg.range(samples1.length())
                    .multiplyScalar(totalDuration1 / samples1.length());
            plot.line(timeAxis1, samples1);
        }
        
        // 确保使用单声道音频进行波形图绘制，与MFCC处理逻辑保持一致
        AudioData monoAudioData2 = audioData2;
        if (!audioData2.isMono()) {
            // 将立体声转换为单声道
            monoAudioData2 = AudioProcessor.stereoToMono(audioData2);
        }
        
        // 添加第二个音频的波形 / Add waveform of second audio
        IVector<Double> samples2 = monoAudioData2.getSamples();
        IVector<Double> timeAxis2 = null;
        
        // Generate time axis that covers the full duration of the audio
        double totalDuration2 = monoAudioData2.getDuration();
        
        if (samples2.length() > MAX_WAVEFORM_POINTS) {
            IVector<Double> displaySamples2 = downsampleForVisualization(samples2, MAX_WAVEFORM_POINTS);
            // Generate time axis that covers the full duration
            timeAxis2 = Linalg.range(displaySamples2.length())
                    .multiplyScalar(totalDuration2 / displaySamples2.length());
            plot.line(timeAxis2, displaySamples2);
        } else {
            // Generate time axis that covers the full duration
            timeAxis2 = Linalg.range(samples2.length())
                    .multiplyScalar(totalDuration2 / samples2.length());
            plot.line(timeAxis2, samples2);
        }
        
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