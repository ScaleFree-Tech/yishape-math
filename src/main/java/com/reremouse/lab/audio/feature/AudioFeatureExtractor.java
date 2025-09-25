package com.reremouse.lab.audio.feature;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.analysis.AbstractAudioAnalyzer;
import com.reremouse.lab.audio.analysis.SpectrumAnalyzer;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.util.Tuple2;
import java.util.Map;
import com.reremouse.lab.audio.analysis.IAudioAnalyzer;

/**
 * 音频特征提取器实现 / Audio Feature Extractor Implementation
 * <p>
 * 提取音频的基本特征，如均方根能量、过零率、频谱质心等。
 * Extract basic audio features such as RMS energy, zero-crossing rate, spectral centroid, etc.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class AudioFeatureExtractor extends AbstractAudioAnalyzer {
    
    // Supported feature types for this analyzer
    private static final String[] SUPPORTED_FEATURE_TYPES = {"rms", "zcr", "spectral_centroid"};

    /**
     * 构造函数 / Constructor
     */
    public AudioFeatureExtractor() {
        super("feature", "Audio feature extractor");
        addSupportedParameter("frameSize", 1024);
        addSupportedParameter("hopSize", 512);
    }
    
    @Override
    protected IVector<Double> doExtractFeatures(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 获取参数 / Get parameters
            int frameSize = 1024;
            int hopSize = 512;
            
            if (parameters != null) {
                if (parameters.containsKey("frameSize")) {
                    frameSize = (Integer) parameters.get("frameSize");
                }
                if (parameters.containsKey("hopSize")) {
                    hopSize = (Integer) parameters.get("hopSize");
                }
            }
            
            // 提取特征 / Extract features
            double rms = calculateRMS(input.getSamples());
            double zcr = calculateZCR(input.getSamples());
            double spectralCentroid = calculateSpectralCentroid(input);
            
            // 创建特征向量 / Create feature vector
            IVector<Double> features = Linalg.zeros(3);
            features.set(0, rms);
            features.set(1, zcr);
            features.set(2, spectralCentroid);
            
            return features;
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to extract audio features", e);
        }
    }
    
    @Override
    protected Tuple2<IVector<Double>, IVector<Double>> doCalculateSpectrum(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        // 对于特征提取器，频谱计算可以委托给频谱分析器 / For feature extractor, spectrum calculation can be delegated to spectrum analyzer
        SpectrumAnalyzer spectrumAnalyzer = new SpectrumAnalyzer();
        return spectrumAnalyzer.calculateSpectrum(input, parameters);
    }
    
    /**
     * 计算均方根能量 / Calculate RMS energy
     */
    private double calculateRMS(IVector<Double> samples) {
        double sum = 0.0;
        int length = samples.length();
        
        for (int i = 0; i < length; i++) {
            double sample = samples.get(i);
            sum += sample * sample;
        }
        
        return Math.sqrt(sum / length);
    }
    
    /**
     * 计算过零率 / Calculate zero-crossing rate
     */
    private double calculateZCR(IVector<Double> samples) {
        int zeroCrossings = 0;
        int length = samples.length();
        
        for (int i = 1; i < length; i++) {
            if ((samples.get(i - 1) >= 0 && samples.get(i) < 0) || 
                (samples.get(i - 1) < 0 && samples.get(i) >= 0)) {
                zeroCrossings++;
            }
        }
        
        return (double) zeroCrossings / length;
    }
    
    /**
     * 计算频谱质心 / Calculate spectral centroid
     */
    private double calculateSpectralCentroid(AudioData input) throws AudioProcessingException {
        // 简化实现：使用频谱分析器计算频谱 / Simplified implementation: use spectrum analyzer to calculate spectrum
        SpectrumAnalyzer spectrumAnalyzer = new SpectrumAnalyzer();
        Tuple2<IVector<Double>, IVector<Double>> spectrum = spectrumAnalyzer.calculateSpectrum(input);
        
        IVector<Double> frequencies = spectrum.getFirst();
        IVector<Double> magnitudes = spectrum.getSecond();
        
        double weightedSum = 0.0;
        double magnitudeSum = 0.0;
        
        int length = frequencies.length();
        for (int i = 0; i < length; i++) {
            double freq = frequencies.get(i);
            double mag = magnitudes.get(i);
            weightedSum += freq * mag;
            magnitudeSum += mag;
        }
        
        return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0.0;
    }
    
    @Override
    public IAudioAnalyzer clone() {
        return new AudioFeatureExtractor();
    }
    
    @Override
    public String[] getSupportedFeatureTypes() {
        return SUPPORTED_FEATURE_TYPES.clone();
    }
    
    @Override
    protected int getDefaultFeatureDimension(String featureType) {
        if ("rms".equalsIgnoreCase(featureType) || 
            "zcr".equalsIgnoreCase(featureType) || 
            "spectral_centroid".equalsIgnoreCase(featureType)) {
            return 1; // Single value features
        }
        return super.getDefaultFeatureDimension(featureType);
    }
}