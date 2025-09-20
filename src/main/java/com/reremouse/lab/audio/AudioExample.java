package com.reremouse.lab.audio;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.signal.Signals;
import com.reremouse.lab.math.viz.IPlot;

import java.util.List;

/**
 * 音频处理示例类 / Audio Processing Example Class
 * <p>
 * 演示音频处理包的各种功能，包括音频生成、处理、分析和可视化。
 * Demonstrates various functionalities of the audio processing package, including audio generation, processing, analysis, and visualization.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioExample {
    
    /**
     * 主方法 / Main method
     *
     * @param args 命令行参数 / Command line arguments
     */
    public static void main(String[] args) {
        System.out.println("=== 音频处理示例 / Audio Processing Example ===");
        
        // 1. 生成测试音频 / Generate test audio
        System.out.println("\n1. 生成测试音频 / Generating test audio...");
        AudioData testAudio = generateTestAudio();
        System.out.println("生成的音频信息 / Generated audio info: " + testAudio);
        
        // 2. 音频基本操作 / Basic audio operations
        System.out.println("\n2. 音频基本操作 / Basic audio operations...");
        demonstrateBasicOperations(testAudio);
        
        // 3. 音频分析 / Audio analysis
        System.out.println("\n3. 音频分析 / Audio analysis...");
        demonstrateAudioAnalysis(testAudio);
        
        // 4. 音频处理 / Audio processing
        System.out.println("\n4. 音频处理 / Audio processing...");
        demonstrateAudioProcessing(testAudio);
        
        // 5. 音频可视化 / Audio visualization
        System.out.println("\n5. 音频可视化 / Audio visualization...");
        demonstrateAudioVisualization(testAudio);
        
        System.out.println("\n=== 示例完成 / Example completed ===");
    }
    
    /**
     * 生成测试音频 / Generate test audio
     *
     * @return 测试音频数据 / Test audio data
     */
    private static AudioData generateTestAudio() {
        // 生成440Hz的正弦波（A4音符） / Generate 440Hz sine wave (A4 note)
        IVector<Double> sineWave = Signals.sineWave(44100, 440.0, 44100.0, 0.5, 0.0);
        
        // 生成一些噪声 / Generate some noise
        IVector<Double> noise = Signals.whiteNoise(44100, 0.1);
        
        // 混合信号 / Mix signals
        IVector<Double> mixedSignal = sineWave.add(noise);
        
        // 创建音频数据 / Create audio data
        return new AudioData(mixedSignal, 44100.0, 1, 16, AudioFormat.WAV);
    }
    
    /**
     * 演示基本操作 / Demonstrate basic operations
     *
     * @param audioData 音频数据 / Audio data
     */
    private static void demonstrateBasicOperations(AudioData audioData) {
        System.out.println("原始音频统计 / Original audio statistics:");
        AudioStatistics originalStats = audioData.getStatistics();
        System.out.println(originalStats);
        
        // 音量调节 / Volume adjustment
        AudioData adjustedAudio = AudioProcessor.adjustVolume(audioData, 0.5);
        System.out.println("\n音量调节后 / After volume adjustment:");
        System.out.println(adjustedAudio.getStatistics());
        
        // 音频归一化 / Audio normalization
        AudioData normalizedAudio = AudioProcessor.normalize(audioData);
        System.out.println("\n归一化后 / After normalization:");
        System.out.println(normalizedAudio.getStatistics());
        
        // 声道转换 / Channel conversion
        if (audioData.isMono()) {
            AudioData stereoAudio = AudioProcessor.monoToStereo(audioData);
            System.out.println("\n转换为立体声 / Converted to stereo:");
            System.out.println("声道数 / Channels: " + stereoAudio.getChannels());
        }
    }
    
    /**
     * 演示音频分析 / Demonstrate audio analysis
     *
     * @param audioData 音频数据 / Audio data
     */
    private static void demonstrateAudioAnalysis(AudioData audioData) {
        // 提取音频特征 / Extract audio features
        AudioFeatures features = AudioAnalyzer.extractFeatures(audioData);
        System.out.println("音频特征 / Audio features:");
        System.out.println(features);
        
        // 检测音调 / Detect pitch
        double pitch = AudioAnalyzer.detectPitch(audioData);
        System.out.println("检测到的音调 / Detected pitch: " + pitch + " Hz");
        
        // 计算频谱 / Calculate spectrum
        var spectrumResult = AudioAnalyzer.calculateSpectrum(audioData);
        IVector<Double> frequencies = spectrumResult.getFirst();
        IVector<Double> magnitudes = spectrumResult.getSecond();
        System.out.println("频谱分析 / Spectrum analysis:");
        System.out.println("频率范围 / Frequency range: " + frequencies.get(0) + " - " + 
                         frequencies.get(frequencies.length() - 1) + " Hz");
        System.out.println("最大幅度频率 / Max magnitude frequency: " + 
                         frequencies.get(magnitudes.argMax()) + " Hz");
    }
    
    /**
     * 演示音频处理 / Demonstrate audio processing
     *
     * @param audioData 音频数据 / Audio data
     */
    private static void demonstrateAudioProcessing(AudioData audioData) {
        // 降噪处理 / Noise reduction
        System.out.println("降噪处理 / Noise reduction...");
        AudioData denoisedAudio = AudioEnhancer.reduceNoise(audioData, null);
        System.out.println("降噪前信噪比 / SNR before denoising: " + 
                         audioData.getStatistics().getSnr() + " dB");
        System.out.println("降噪后信噪比 / SNR after denoising: " + 
                         denoisedAudio.getStatistics().getSnr() + " dB");
        
        // 均衡处理 / Equalization
        System.out.println("\n均衡处理 / Equalization...");
        EqualizerSettings eqSettings = EqualizerSettings.createPreset(EqualizerSettings.EqualizerPreset.VOCAL);
        AudioData equalizedAudio = AudioEnhancer.equalize(audioData, eqSettings);
        System.out.println("均衡器设置 / Equalizer settings: " + eqSettings);
        
        // 压缩处理 / Compression
        System.out.println("\n压缩处理 / Compression...");
        CompressionSettings compSettings = CompressionSettings.createPreset(CompressionSettings.CompressionPreset.MEDIUM);
        AudioData compressedAudio = AudioEnhancer.compress(audioData, compSettings);
        System.out.println("压缩器设置 / Compressor settings: " + compSettings);
        
        // 混响处理 / Reverb processing
        System.out.println("\n混响处理 / Reverb processing...");
        ReverbSettings reverbSettings = ReverbSettings.createPreset(ReverbSettings.ReverbPreset.HALL);
        AudioData reverbAudio = AudioEnhancer.addReverb(audioData, reverbSettings);
        System.out.println("混响设置 / Reverb settings: " + reverbSettings);
    }
    
    /**
     * 演示音频可视化 / Demonstrate audio visualization
     *
     * @param audioData 音频数据 / Audio data
     */
    private static void demonstrateAudioVisualization(AudioData audioData) {
        System.out.println("创建音频可视化图表 / Creating audio visualization plots...");
        
        try {
            // 创建各种图表 / Create various plots
            IPlot waveformPlot = AudioVisualizer.plotWaveform(audioData, "测试音频波形 / Test Audio Waveform");
            IPlot spectrumPlot = AudioVisualizer.plotSpectrum(audioData, "测试音频频谱 / Test Audio Spectrum");
            IPlot logSpectrumPlot = AudioVisualizer.plotLogSpectrum(audioData, "测试音频对数频谱 / Test Audio Log Spectrum");
            IPlot spectrogramPlot = AudioVisualizer.plotSpectrogram(audioData, "测试音频频谱图 / Test Audio Spectrogram");
            IPlot featuresPlot = AudioVisualizer.plotAudioFeatures(audioData, "测试音频特征 / Test Audio Features");
            IPlot statisticsPlot = AudioVisualizer.plotAudioStatistics(audioData, "测试音频统计 / Test Audio Statistics");
            IPlot qualityPlot = AudioVisualizer.plotAudioQuality(audioData, "测试音频质量 / Test Audio Quality");
            
            System.out.println("波形图 / Waveform plot: " + waveformPlot);
            System.out.println("频谱图 / Spectrum plot: " + spectrumPlot);
            System.out.println("对数频谱图 / Log spectrum plot: " + logSpectrumPlot);
            System.out.println("频谱图 / Spectrogram plot: " + spectrogramPlot);
            System.out.println("特征图 / Features plot: " + featuresPlot);
            System.out.println("统计图 / Statistics plot: " + statisticsPlot);
            System.out.println("质量图 / Quality plot: " + qualityPlot);
            
            // 创建仪表板 / Create dashboard
            List<IPlot> dashboard = AudioVisualizer.createAudioDashboard(audioData, "音频处理仪表板 / Audio Processing Dashboard");
            System.out.println("仪表板包含 " + dashboard.size() + " 个图表 / Dashboard contains " + dashboard.size() + " plots");
            
        } catch (Exception e) {
            System.err.println("可视化过程中出现错误 / Error during visualization: " + e.getMessage());
        }
    }
    
    /**
     * 演示音频比较 / Demonstrate audio comparison
     *
     * @param audioData1 第一个音频 / First audio
     * @param audioData2 第二个音频 / Second audio
     */
    public static void demonstrateAudioComparison(AudioData audioData1, AudioData audioData2) {
        System.out.println("\n=== 音频比较 / Audio Comparison ===");
        
        // 提取特征 / Extract features
        AudioFeatures features1 = AudioAnalyzer.extractFeatures(audioData1);
        AudioFeatures features2 = AudioAnalyzer.extractFeatures(audioData2);
        
        // 计算相似度 / Calculate similarity
        double distance = features1.distanceTo(features2);
        double similarity = features1.cosineSimilarityTo(features2);
        
        System.out.println("特征距离 / Feature distance: " + distance);
        System.out.println("余弦相似度 / Cosine similarity: " + similarity);
        
        // 创建比较图 / Create comparison plot
        try {
            IPlot comparisonPlot = AudioVisualizer.plotAudioComparison(audioData1, audioData2, "音频比较 / Audio Comparison");
            System.out.println("比较图 / Comparison plot: " + comparisonPlot);
        } catch (Exception e) {
            System.err.println("比较图创建失败 / Failed to create comparison plot: " + e.getMessage());
        }
    }
}
