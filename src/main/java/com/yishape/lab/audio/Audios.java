package com.yishape.lab.audio;

import com.yishape.lab.audio.analysis.IAudioAnalyzer;
import com.yishape.lab.audio.effect.IAudioEffect;
import com.yishape.lab.audio.enhancement.IAudioEnhancer;
import com.yishape.lab.audio.factory.AudioComponentFactory;
import com.yishape.lab.audio.filter.IBaseAudioFilter;
import com.yishape.lab.audio.processing.ChannelProcessor;
import com.yishape.lab.audio.processing.IAdvancedAudioProcessor;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.audio.core.*;
import com.yishape.lab.audio.embedding.IVectorEmbedding;
import com.yishape.lab.audio.embedding.OnlineIVectorEmbedding;
import com.yishape.lab.audio.feature.IAudioFeatureExtractor;
import com.yishape.lab.math.signal.core.Complex;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Audio处理的静态工厂入口类 / Audio Processing Entry Factory Class
 * <p>
 * 提供统一的音频处理接口，封装了音频处理、分析、滤波、效果等核心功能。 Provides a unified audio processing
 * interface that encapsulates core functions such as audio processing,
 * analysis, filtering, and effects.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class Audios {

    /**
     * 创建在线向量嵌入器 / Create online vector embedder
     *
     * @param dimNum 向量维度 / Vector dimension
     * @return 在线向量嵌入器实例 / Online vector embedder instance
     */
    public OnlineIVectorEmbedding createOnlineAudioEmbedder(int dimNum) {
        OnlineIVectorEmbedding emb = new OnlineIVectorEmbedding(dimNum);
        return emb;
    }
    
    /**
     * 创建音频嵌入器 / Create audio embedder
     *
     * @param dimNum 向量维度 / Vector dimension
     * @return 音频嵌入器实例 / Audio embedder instance
     */
    public IVectorEmbedding createAudioEmbedder(int dimNum) {
        IVectorEmbedding emb = new IVectorEmbedding(dimNum);
        return emb;
    }


    // ========== 音频处理器创建方法 / Audio Processor Creation Methods ==========
    /**
     * 创建音频处理器 / Create audio processor
     *
     * @param processorType 处理器类型 / Processor type
     * @return 音频处理器实例 / Audio processor instance
     */
    public static IAdvancedAudioProcessor createProcessor(String processorType) {
        try {
            return AudioComponentFactory.getInstance().createProcessor(processorType);
        } catch (Exception e) {
            throw new RuntimeException("Failed to create processor: " + processorType, e);
        }
    }

    /**
     * 创建音量处理器 / Create volume processor
     *
     * @return 音量处理器实例 / Volume processor instance
     */
    public static IAdvancedAudioProcessor createVolumeProcessor() {
        return createProcessor("volume");
    }

    /**
     * 创建标准化处理器 / Create normalize processor
     *
     * @return 标准化处理器实例 / Normalize processor instance
     */
    public static IAdvancedAudioProcessor createNormalizeProcessor() {
        return createProcessor("normalize");
    }

    /**
     * 创建声道处理器 / Create channel processor
     *
     * @return 声道处理器实例 / Channel processor instance
     */
    public static IAdvancedAudioProcessor createChannelProcessor() {
        return createProcessor("channel");
    }

    // ========== 音频分析器创建方法 / Audio Analyzer Creation Methods ==========
    /**
     * 创建音频分析器 / Create audio analyzer
     *
     * @param analyzerType 分析器类型 / Analyzer type
     * @return 音频分析器实例 / Audio analyzer instance
     */
    public static IAudioAnalyzer createAnalyzer(String analyzerType) {
        try {
            return AudioComponentFactory.getInstance().createAnalyzer(analyzerType);
        } catch (Exception e) {
            throw new RuntimeException("Failed to create analyzer: " + analyzerType, e);
        }
    }

    /**
     * 创建频谱分析器 / Create spectrum analyzer
     *
     * @return 频谱分析器实例 / Spectrum analyzer instance
     */
    public static IAudioAnalyzer createSpectrumAnalyzer() {
        return createAnalyzer("spectrum");
    }

    /**
     * 创建音高检测器 / Create pitch detector
     *
     * @return 音高检测器实例 / Pitch detector instance
     */
    public static IAudioAnalyzer createPitchDetector() {
        return createAnalyzer("pitch");
    }

    /**
     * 创建STFT分析器 / Create STFT analyzer
     *
     * @return STFT分析器实例 / STFT analyzer instance
     */
    public static IAudioAnalyzer createSTFTAnalyzer() {
        return createAnalyzer("stft");
    }

    // ========== 音频滤波器创建方法 / Audio Filter Creation Methods ==========
    /**
     * 创建音频滤波器 / Create audio filter
     *
     * @param filterType 滤波器类型 / Filter type
     * @return 音频滤波器实例 / Audio filter instance
     */
    public static IBaseAudioFilter createFilter(String filterType) {
        try {
            return AudioComponentFactory.getInstance().createFilter(filterType);
        } catch (Exception e) {
            throw new RuntimeException("Failed to create filter: " + filterType, e);
        }
    }

    /**
     * 创建低通滤波器 / Create low-pass filter
     *
     * @return 低通滤波器实例 / Low-pass filter instance
     */
    public static IBaseAudioFilter createLowPassFilter() {
        return createFilter("lowpass");
    }

    /**
     * 创建高级低通滤波器 / Create advanced low-pass filter
     *
     * @return 高级低通滤波器实例 / Advanced low-pass filter instance
     */
    public static IBaseAudioFilter createAdvancedLowPassFilter() {
        return createFilter("advanced_lowpass");
    }

    // ========== 音频效果器创建方法 / Audio Effect Creation Methods ==========
    /**
     * 创建音频效果器 / Create audio effect
     *
     * @param effectType 效果器类型 / Effect type
     * @return 音频效果器实例 / Audio effect instance
     */
    public static IAudioEffect createEffect(String effectType) {
        try {
            return AudioComponentFactory.getInstance().createEffect(effectType);
        } catch (Exception e) {
            throw new RuntimeException("Failed to create effect: " + effectType, e);
        }
    }

    /**
     * 创建混响效果器 / Create reverb effect
     *
     * @return 混响效果器实例 / Reverb effect instance
     */
    public static IAudioEffect createReverbEffect() {
        return createEffect("reverb");
    }

    // ========== 音频增强器创建方法 / Audio Enhancer Creation Methods ==========
    /**
     * 创建音频增强器 / Create audio enhancer
     *
     * @param enhancerType 增强器类型 / Enhancer type
     * @return 音频增强器实例 / Audio enhancer instance
     */
    public static IAudioEnhancer createEnhancer(String enhancerType) {
        try {
            return AudioComponentFactory.getInstance().createEnhancer(enhancerType);
        } catch (Exception e) {
            throw new RuntimeException("Failed to create enhancer: " + enhancerType, e);
        }
    }

    /**
     * 创建降噪增强器 / Create noise reduction enhancer
     *
     * @return 降噪增强器实例 / Noise reduction enhancer instance
     */
    public static IAudioEnhancer createNoiseReductionEnhancer() {
        return createEnhancer("noise_reduction");
    }

    /**
     * 创建均衡器增强器 / Create equalizer enhancer
     *
     * @return 均衡器增强器实例 / Equalizer enhancer instance
     */
    public static IAudioEnhancer createEqualizerEnhancer() {
        return createEnhancer("equalizer");
    }

    /**
     * 创建压缩器增强器 / Create compressor enhancer
     *
     * @return 压缩器增强器实例 / Compressor enhancer instance
     */
    public static IAudioEnhancer createCompressorEnhancer() {
        return createEnhancer("compressor");
    }

    // ========== 音频编解码器创建方法 / Audio Codec Creation Methods ==========
    /**
     * 创建音频编解码器 / Create audio codec
     *
     * @param codecType 编解码器类型 / Codec type
     * @return 音频编解码器实例 / Audio codec instance
     */
    public static IAudioCodec createCodec(String codecType) {
        try {
            return AudioComponentFactory.getInstance().createCodec(codecType);
        } catch (Exception e) {
            throw new RuntimeException("Failed to create codec: " + codecType, e);
        }
    }

    // ========== 音频特征提取器创建方法 / Audio Feature Extractor Creation Methods ==========
    /**
     * 创建音频特征提取器 / Create audio feature extractor
     *
     * @param extractorType 特征提取器类型 / Feature extractor type
     * @return 音频特征提取器实例 / Audio feature extractor instance
     */
    public static IAudioFeatureExtractor createFeatureExtractor(String extractorType) {
        try {
            return AudioComponentFactory.getInstance().createFeatureExtractor(extractorType);
        } catch (Exception e) {
            throw new RuntimeException("Failed to create feature extractor: " + extractorType, e);
        }
    }

    /**
     * 创建标准特征提取器 / Create standard feature extractor
     *
     * @return 标准特征提取器实例 / Standard feature extractor instance
     */
    public static IAudioFeatureExtractor createStandardFeatureExtractor() {
        return createFeatureExtractor("standard");
    }

    // ========== 音频处理方法 / Audio Processing Methods ==========
    /**
     * 调节音量 / Adjust Volume
     *
     * @param audio 音频数据 / Audio data
     * @param gain 增益值 / Gain value
     * @return 调节后的音频数据 / Adjusted audio data
     */
    public static AudioData adjustVolume(AudioData audio, double gain) {
        try {
            IAdvancedAudioProcessor processor = createVolumeProcessor();
            Map<String, Object> params = new HashMap<>();
            params.put("gain", gain);
            processor.setParameters(params);
            return processor.process(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to adjust volume", e);
        }
    }

    // ========== 音频输入输出方法 / Audio Input/Output Methods ==========
    /**
     * 从文件读取音频数据 / Read audio data from file
     * <p>
     * 根据文件扩展名自动识别音频格式并读取。
     * Automatically identifies audio format by file extension and reads.
     * </p>
     *
     * @param filePath 音频文件路径 / Audio file path
     * @return 音频数据对象 / Audio data object
     * @throws IOException 如果文件读取失败 / If file reading fails
     * @throws UnsupportedAudioFormatException 如果音频格式不支持 / If audio format is not supported
     */
    public static AudioData readAudio(String filePath) throws IOException, UnsupportedAudioFormatException {
        return AudioIO.readAudio(filePath);
    }
    
    /**
     * 从文件读取音频数据（指定格式） / Read audio data from file (specified format)
     *
     * @param filePath 音频文件路径 / Audio file path
     * @param format 音频格式 / Audio format
     * @return 音频数据对象 / Audio data object
     * @throws IOException 如果文件读取失败 / If file reading fails
     * @throws UnsupportedAudioFormatException 如果音频格式不支持 / If audio format is not supported
     */
    public static AudioData readAudio(String filePath, AudioFormat format) throws IOException, UnsupportedAudioFormatException {
        return AudioIO.readAudio(filePath, format);
    }
    
    /**
     * 将音频数据写入文件 / Write audio data to file
     *
     * @param audioData 音频数据 / Audio data
     * @param filePath 输出文件路径 / Output file path
     * @throws IOException 如果文件写入失败 / If file writing fails
     * @throws UnsupportedAudioFormatException 如果音频格式不支持 / If audio format is not supported
     */
    public static void writeAudio(AudioData audioData, String filePath) throws IOException, UnsupportedAudioFormatException {
        AudioIO.writeAudio(audioData, filePath);
    }
    
    /**
     * 将音频数据写入文件（指定格式） / Write audio data to file (specified format)
     *
     * @param audioData 音频数据 / Audio data
     * @param filePath 输出文件路径 / Output file path
     * @param format 音频格式 / Audio format
     * @throws IOException 如果文件写入失败 / If file writing fails
     * @throws UnsupportedAudioFormatException 如果音频格式不支持 / If audio format is not supported
     */
    public static void writeAudio(AudioData audioData, String filePath, AudioFormat format) throws IOException, UnsupportedAudioFormatException {
        AudioIO.writeAudio(audioData, filePath, format);
    }

    // ========== 音频处理方法 / Audio Processing Methods ==========
    /**
     * 归一化音频 / Normalize Audio
     *
     * @param audio 音频数据 / Audio data
     * @return 归一化后的音频数据 / Normalized audio data
     */
    public static AudioData normalize(AudioData audio) {
        try {
            IAdvancedAudioProcessor processor = createNormalizeProcessor();
            return processor.process(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to normalize audio", e);
        }
    }

    /**
     * 转换声道 / Convert Channels
     *
     * @param audio 音频数据 / Audio data
     * @param targetChannels 目标声道数 / Target number of channels
     * @return 转换后的音频数据 / Converted audio data
     */
    public static AudioData convertChannels(AudioData audio, int targetChannels) {
        try {
            IAdvancedAudioProcessor processor = createChannelProcessor();
            Map<String, Object> params = new HashMap<>();
            // Use the correct parameter name and value type
            if (targetChannels == 1 && audio.getChannels() > 1) {
                params.put("operation", ChannelProcessor.ChannelOperation.STEREO_TO_MONO);
            } else if (targetChannels == 2 && audio.getChannels() == 1) {
                params.put("operation", ChannelProcessor.ChannelOperation.MONO_TO_STEREO);
            }
            // For other cases, we'll use the default operation
            processor.setParameters(params);
            return processor.process(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to convert channels", e);
        }
    }

    /**
     * 转换为单声道 / Convert to Mono
     *
     * @param audio 音频数据 / Audio data
     * @return 单声道音频数据 / Mono audio data
     */
    public static AudioData toMono(AudioData audio) {
        return convertChannels(audio, 1);
    }

    /**
     * 转换为立体声 / Convert to Stereo
     *
     * @param audio 音频数据 / Audio data
     * @return 立体声音频数据 / Stereo audio data
     */
    public static AudioData toStereo(AudioData audio) {
        return convertChannels(audio, 2);
    }

    // ========== 音频分析方法 / Audio Analysis Methods ==========
    /**
     * 计算频谱 / Calculate Spectrum
     *
     * @param audio 音频数据 / Audio data
     * @return 频谱结果（元组：频率向量，幅度向量）/ Spectrum result (tuple: frequency vector, magnitude vector)
     */
    public static Tuple2<IVector<Double>, IVector<Double>> spectrum(AudioData audio) {
        try {
            IAudioAnalyzer analyzer = createSpectrumAnalyzer();
            return analyzer.calculateSpectrum(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate spectrum", e);
        }
    }

    /**
     * 计算频谱（带参数） / Calculate Spectrum (with parameters)
     *
     * @param audio 音频数据 / Audio data
     * @param windowSize 窗口大小 / Window size
     * @param overlap 重叠大小 / Overlap size
     * @return 频谱结果（元组：频率向量，幅度向量）/ Spectrum result (tuple: frequency vector, magnitude vector)
     */
    public static Tuple2<IVector<Double>, IVector<Double>> spectrum(AudioData audio, int windowSize, double overlap) {
        try {
            IAudioAnalyzer analyzer = createSpectrumAnalyzer();
            Map<String, Object> params = new HashMap<>();
            params.put("windowSize", windowSize);
            params.put("overlap", overlap);
            return analyzer.calculateSpectrum(audio, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate spectrum", e);
        }
    }


    /**
     * STFT分析 / STFT Analysis
     *
     * @param audio 音频数据 / Audio data
     * @return STFT结果（元组：频率向量，幅度向量）/ STFT result (tuple: frequency vector, magnitude vector)
     */
    public static Tuple2<IVector<Double>, IVector<Double>> stft(AudioData audio) {
        try {
            IAudioAnalyzer analyzer = createSTFTAnalyzer();
            return analyzer.calculateSpectrum(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to perform STFT analysis", e);
        }
    }

    /**
     * 音高检测 / Pitch Detection
     *
     * @param audio 音频数据 / Audio data
     * @return 检测到的音高值（Hz）/ Detected pitch value (Hz)
     */
    public static double detectPitch(AudioData audio) {
        try {
            IAudioAnalyzer analyzer = createPitchDetector();
            Tuple2<IVector<Double>, IVector<Double>> result = analyzer.calculateSpectrum(audio);
            // The pitch detector returns the pitch in the first element of frequencies
            return result._1.get(0);
        } catch (Exception e) {
            throw new RuntimeException("Failed to detect pitch", e);
        }
    }

    // ========== 音频滤波方法 / Audio Filtering Methods ==========
    /**
     * 低通滤波 / Low-pass Filter
     *
     * @param audio 音频数据 / Audio data
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @return 滤波后的音频数据 / Filtered audio data
     */
    public static AudioData lowPassFilter(AudioData audio, double cutoffFreq) {
        try {
            IBaseAudioFilter filter = createLowPassFilter();
            filter.setCutoffFrequency(cutoffFreq);
            return filter.filter(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply low-pass filter", e);
        }
    }

    // ========== 音频效果方法 / Audio Effect Methods ==========
    /**
     * 混响效果 / Reverb Effect
     *
     * @param audio 音频数据 / Audio data
     * @param decay 衰减时间 / Decay time
     * @param wetMix 混音比例 / Wet mix ratio
     * @return 添加混响后的音频数据 / Audio data with reverb effect
     */
    public static AudioData reverb(AudioData audio, double decay, double wetMix) {
        try {
            IAudioEffect effect = createReverbEffect();
            // Set parameters using the effect's parameter methods
            effect.setIntensity(decay);
            effect.setDryWetMix(wetMix);
            return effect.applyEffect(audio);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply reverb effect", e);
        }
    }

    // ========== 音频增强方法 / Audio Enhancement Methods ==========
    /**
     * 降噪 / Noise Reduction
     *
     * @param audio 音频数据 / Audio data
     * @param threshold 噪声阈值 / Noise threshold
     * @return 降噪后的音频数据 / Noise-reduced audio data
     */
    public static AudioData reduceNoise(AudioData audio, double threshold) {
        try {
            IAudioEnhancer enhancer = createNoiseReductionEnhancer();
            Map<String, Object> params = new HashMap<>();
            params.put("noiseThreshold", threshold);
            return enhancer.enhance(audio, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to reduce noise", e);
        }
    }

    /**
     * 均衡器 / Equalizer
     *
     * @param audio 音频数据 / Audio data
     * @param bandGains 频段增益映射 / Band gains map
     * @return 均衡后的音频数据 / Equalized audio data
     */
    public static AudioData equalize(AudioData audio, Map<String, Double> bandGains) {
        try {
            IAudioEnhancer enhancer = createEqualizerEnhancer();
            Map<String, Object> params = new HashMap<>();
            if (bandGains != null) {
                if (bandGains.containsKey("lowGain")) {
                    params.put("lowGain", bandGains.get("lowGain"));
                }
                if (bandGains.containsKey("midGain")) {
                    params.put("midGain", bandGains.get("midGain"));
                }
                if (bandGains.containsKey("highGain")) {
                    params.put("highGain", bandGains.get("highGain"));
                }
            }
            return enhancer.enhance(audio, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply equalizer", e);
        }
    }

    /**
     * 压缩器 / Compressor
     *
     * @param audio 音频数据 / Audio data
     * @param threshold 阈值 / Threshold
     * @param ratio 压缩比 / Compression ratio
     * @return 压缩后的音频数据 / Compressed audio data
     */
    public static AudioData compress(AudioData audio, double threshold, double ratio) {
        try {
            IAudioEnhancer enhancer = createCompressorEnhancer();
            Map<String, Object> params = new HashMap<>();
            params.put("threshold", threshold);
            params.put("ratio", ratio);
            return enhancer.enhance(audio, params);
        } catch (Exception e) {
            throw new RuntimeException("Failed to apply compressor", e);
        }
    }

    // ========== 工厂方法 / Factory Methods ==========
    /**
     * 获取音频组件工厂实例 / Get audio component factory instance
     *
     * @return 音频组件工厂 / Audio component factory
     */
    public static AudioComponentFactory getFactory() {
        return AudioComponentFactory.getInstance();
    }

    // ========== 常用音频处理方法 / Common Audio Processing Methods ==========
    
    /**
     * 计算音频的RMS值 / Calculate RMS value of audio
     *
     * @param audioData 音频数据 / Audio data
     * @return RMS值 / RMS value
     */
    public static double calculateRMS(AudioData audioData) {
        return AudioUtil.calculateRMS(audioData.getSamples());
    }

    /**
     * 计算音频的零交叉率 / Calculate zero crossing rate of audio
     *
     * @param audioData 音频数据 / Audio data
     * @return 零交叉率 / Zero crossing rate
     */
    public static double calculateZeroCrossingRate(AudioData audioData) {
        return AudioUtil.calculateZeroCrossingRate(audioData.getSamples());
    }

    /**
     * 计算音频的能量 / Calculate energy of audio
     *
     * @param audioData 音频数据 / Audio data
     * @return 能量值 / Energy value
     */
    public static double calculateEnergy(AudioData audioData) {
        return AudioUtil.calculateEnergy(audioData.getSamples());
    }

    /**
     * 计算音频的MFCC矩阵 / Calculate MFCC matrix of audio
     *
     * @param audioData 音频数据 / Audio data
     * @param mfccCount MFCC系数数量 / Number of MFCC coefficients
     * @param windowSize 窗口大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @return MFCC矩阵 / MFCC matrix
     */
    public static IMatrix<Double> calculateMFCC(AudioData audioData, int mfccCount, int windowSize, int hopSize) {
        try {
            return AudioUtil.calculateMFCCMatrix(audioData, mfccCount, windowSize, hopSize);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate MFCC", e);
        }
    }

    /**
     * 计算音频的MFCC矩阵（默认参数） / Calculate MFCC matrix of audio (default parameters)
     *
     * @param audioData 音频数据 / Audio data
     * @return MFCC矩阵 / MFCC matrix
     */
    public static IMatrix<Double> calculateMFCC(AudioData audioData) {
        try {
            return AudioUtil.calculateMFCCMatrix(audioData);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate MFCC", e);
        }
    }

    /**
     * 对音频执行FFT处理 / Perform FFT processing on audio
     *
     * @param audioData 音频数据 / Audio data
     * @param windowSize 窗口大小 / Window size
     * @return FFT结果复数数组 / FFT result complex array
     */
    public static Complex[] processFFT(AudioData audioData, int windowSize) {
        try {
            return AudioUtil.processFFT(audioData, windowSize);
        } catch (Exception e) {
            throw new RuntimeException("Failed to process FFT", e);
        }
    }

    /**
     * 对音频执行FFT处理（默认窗口大小） / Perform FFT processing on audio (default window size)
     *
     * @param audioData 音频数据 / Audio data
     * @return FFT结果复数数组 / FFT result complex array
     */
    public static Complex[] processFFT(AudioData audioData) {
        try {
            return AudioUtil.processFFT(audioData);
        } catch (Exception e) {
            throw new RuntimeException("Failed to process FFT", e);
        }
    }

    /**
     * 对信号应用窗函数 / Apply window function to signal
     *
     * @param audioData 音频数据 / Audio data
     * @param windowSize 窗口大小 / Window size
     * @return 加窗后的信号向量 / Windowed signal vector
     */
    public static IVector<Double> applyWindow(AudioData audioData, int windowSize) {
        return AudioUtil.applyWindow(audioData.getSamples(), windowSize);
    }

    /**
     * 计算频谱重心 / Calculate spectral centroid
     *
     * @param audioData 音频数据 / Audio data
     * @param windowSize 窗口大小 / Window size
     * @return 频谱重心频率（Hz）/ Spectral centroid frequency (Hz)
     */
    public static double calculateSpectralCentroid(AudioData audioData, int windowSize) {
        try {
            Complex[] spectrum = AudioUtil.processFFT(audioData, windowSize);
            return AudioUtil.calculateSpectralCentroid(spectrum, audioData.getSampleRate(), windowSize);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate spectral centroid", e);
        }
    }

    /**
     * 计算频谱滚降点 / Calculate spectral rolloff
     *
     * @param audioData 音频数据 / Audio data
     * @param windowSize 窗口大小 / Window size
     * @return 频谱滚降频率（Hz）/ Spectral rolloff frequency (Hz)
     */
    public static double calculateSpectralRolloff(AudioData audioData, int windowSize) {
        try {
            Complex[] spectrum = AudioUtil.processFFT(audioData, windowSize);
            return AudioUtil.calculateSpectralRolloff(spectrum, audioData.getSampleRate(), windowSize);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate spectral rolloff", e);
        }
    }

    /**
     * 计算频谱带宽 / Calculate spectral bandwidth
     *
     * @param audioData 音频数据 / Audio data
     * @param windowSize 窗口大小 / Window size
     * @return 频谱带宽（Hz）/ Spectral bandwidth (Hz)
     */
    public static double calculateSpectralBandwidth(AudioData audioData, int windowSize) {
        try {
            Complex[] spectrum = AudioUtil.processFFT(audioData, windowSize);
            double centroid = AudioUtil.calculateSpectralCentroid(spectrum, audioData.getSampleRate(), windowSize);
            return AudioUtil.calculateSpectralBandwidth(spectrum, audioData.getSampleRate(), windowSize, centroid);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate spectral bandwidth", e);
        }
    }

    /**
     * 计算频谱对比度 / Calculate spectral contrast
     *
     * @param audioData 音频数据 / Audio data
     * @return 频谱对比度值 / Spectral contrast value
     */
    public static double calculateSpectralContrast(AudioData audioData) {
        try {
            Complex[] spectrum = AudioUtil.processFFT(audioData);
            return AudioUtil.calculateSpectralContrast(spectrum);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate spectral contrast", e);
        }
    }

    /**
     * 计算频谱平坦度 / Calculate spectral flatness
     *
     * @param audioData 音频数据 / Audio data
     * @return 频谱平坦度值 / Spectral flatness value
     */
    public static double calculateSpectralFlatness(AudioData audioData) {
        try {
            Complex[] spectrum = AudioUtil.processFFT(audioData);
            return AudioUtil.calculateSpectralFlatness(spectrum);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate spectral flatness", e);
        }
    }

    /**
     * 计算频谱流量 / Calculate spectral flux
     *
     * @param audioData 音频数据 / Audio data
     * @param parameters 参数映射 / Parameters map
     * @return 频谱流量值 / Spectral flux value
     */
    public static double calculateSpectralFlux(AudioData audioData, Map<String, Object> parameters) {
        try {
            return AudioUtil.calculateSpectralFlux(audioData, parameters);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate spectral flux", e);
        }
    }

    /**
     * 计算粗糙度 / Calculate roughness
     *
     * @param audioData 音频数据 / Audio data
     * @return 粗糙度值 / Roughness value
     */
    public static double calculateRoughness(AudioData audioData) {
        return AudioUtil.calculateRoughness(audioData);
    }

    /**
     * 计算语音频率范围能量 / Calculate speech frequency energy
     *
     * @param audioData 音频数据 / Audio data
     * @return 语音频率能量占比 / Speech frequency energy ratio
     */
    public static double calculateSpeechFrequencyEnergy(AudioData audioData) {
        return AudioUtil.calculateSpeechFrequencyEnergy(audioData);
    }

    /**
     * 计算人声频率范围能量 / Calculate vocal frequency energy
     *
     * @param audioData 音频数据 / Audio data
     * @return 人声频率能量占比 / Vocal frequency energy ratio
     */
    public static double calculateVocalFrequencyEnergy(AudioData audioData) {
        return AudioUtil.calculateVocalFrequencyEnergy(audioData);
    }

    /**
     * 计算背景噪声水平 / Calculate background noise level
     *
     * @param audioData 音频数据 / Audio data
     * @return 背景噪声水平 / Background noise level
     */
    public static double calculateBackgroundNoise(AudioData audioData) {
        return AudioUtil.calculateBackgroundNoise(audioData);
    }

    // ========== FBank特征提取方法 / FBank Feature Extraction Methods ==========

    /**
     * 计算音频的FBank特征矩阵 / Calculate FBank feature matrix of audio
     *
     * @param audioData 音频数据 / Audio data
     * @param fbankCount FBank系数数量 / Number of FBank coefficients
     * @param windowSize 窗口大小 / Window size
     * @param hopSize 跳跃大小 / Hop size
     * @return FBank特征矩阵 / FBank feature matrix
     */
    public static IMatrix<Double> calculateFBank(AudioData audioData, int fbankCount, int windowSize, int hopSize) {
        try {
            return AudioUtil.calculateFBankMatrix(audioData, fbankCount, windowSize, hopSize);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate FBank features", e);
        }
    }

    /**
     * 计算音频的FBank特征矩阵（默认参数） / Calculate FBank feature matrix of audio (default parameters)
     *
     * @param audioData 音频数据 / Audio data
     * @return FBank特征矩阵 / FBank feature matrix
     */
    public static IMatrix<Double> calculateFBank(AudioData audioData) {
        try {
            return AudioUtil.calculateFBankMatrix(audioData);
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate FBank features", e);
        }
    }
}