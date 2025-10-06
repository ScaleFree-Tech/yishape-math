package com.yishape.lab.audio.analysis;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple2;

import java.util.Map;

/**
 * 频谱分析器实现 / Spectrum Analyzer Implementation
 * <p>
 * 计算音频的频谱表示，包括频率和幅度信息。
 * Calculate spectral representation of audio, including frequency and magnitude information.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class SpectrumAnalyzer extends AbstractAudioAnalyzer {
    
    // Supported feature types for this analyzer
    private static final String[] SUPPORTED_FEATURE_TYPES = {"spectrum", "magnitude"};

    /**
     * 构造函数 / Constructor
     */
    public SpectrumAnalyzer() {
        super("spectrum", "Spectrum analyzer");
        addSupportedParameter("windowSize", 1024);
        addSupportedParameter("overlap", 0.5);
    }
    
    @Override
    protected IVector<Double> doExtractFeatures(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 对于频谱分析器，特征提取返回幅度谱 / For spectrum analyzer, feature extraction returns magnitude spectrum
            Tuple2<IVector<Double>, IVector<Double>> spectrum = calculateSpectrum(input, parameters);
            return spectrum.getSecond(); // 返回幅度 / Return magnitude
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to extract spectrum features", e);
        }
    }
    
    @Override
    protected Tuple2<IVector<Double>, IVector<Double>> doCalculateSpectrum(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        // 使用优化的FFT处理器 / Use optimized FFT processor
        return FFTProcessor.getInstance().processFFT(input, parameters);
    }
    
    /**
     * 应用窗口函数 / Apply window function
     */
    private IVector<Double> applyWindow(IVector<Double> samples, int windowSize) {
        int length = Math.min(samples.length(), windowSize);
        IVector<Double> windowed = Linalg.zeros(length);
        
        // 应用汉宁窗 / Apply Hanning window
        for (int i = 0; i < length; i++) {
            double windowValue = 0.5 * (1 - Math.cos(2 * Math.PI * i / (length - 1)));
            windowed.set(i, samples.get(i) * windowValue);
        }
        
        return windowed;
    }
    
    @Override
    public IAudioAnalyzer clone() {
        return new SpectrumAnalyzer();
    }
    
    @Override
    public String[] getSupportedFeatureTypes() {
        return SUPPORTED_FEATURE_TYPES.clone();
    }
    
    @Override
    protected int getDefaultFeatureDimension(String featureType) {
        if ("spectrum".equalsIgnoreCase(featureType) || "magnitude".equalsIgnoreCase(featureType)) {
            // Default spectrum size
            return 513; // For 1024-point FFT
        }
        return super.getDefaultFeatureDimension(featureType);
    }
}