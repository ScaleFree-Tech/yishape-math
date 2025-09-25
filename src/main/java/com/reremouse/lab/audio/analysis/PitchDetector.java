package com.reremouse.lab.audio.analysis;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioProcessor;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.util.Tuple2;
import java.util.Map;

/**
 * 音调检测器实现 / Pitch Detector Implementation
 * <p>
 * 实现音调检测功能，使用自相关方法检测音频的主频率。
 * Implements pitch detection functionality, using autocorrelation method to detect main frequency of audio.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class PitchDetector extends AbstractAudioAnalyzer {
    
    // Supported feature types for this analyzer
    private static final String[] SUPPORTED_FEATURE_TYPES = {"pitch"};

    /**
     * 构造函数 / Constructor
     */
    public PitchDetector() {
        super("pitch", "Pitch detector");
        addSupportedParameter("minFrequency", 80.0);
        addSupportedParameter("maxFrequency", 1000.0);
    }
    
    @Override
    protected IVector<Double> doExtractFeatures(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 对于音调检测器，特征提取返回检测到的音调 / For pitch detector, feature extraction returns detected pitch
            double pitch = detectPitch(input, parameters);
            IVector<Double> result = Linalg.zeros(1);
            result.set(0, pitch);
            return result;
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to extract pitch features", e);
        }
    }
    
    @Override
    protected Tuple2<IVector<Double>, IVector<Double>> doCalculateSpectrum(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 音调检测器不直接计算频谱，但可以返回音调信息 / Pitch detector doesn't directly calculate spectrum, but can return pitch information
            double pitch = detectPitch(input, parameters);
            IVector<Double> frequencies = Linalg.zeros(1);
            IVector<Double> magnitudes = Linalg.zeros(1);
            frequencies.set(0, pitch);
            magnitudes.set(0, 1.0); // 简化实现 / Simplified implementation
            return new Tuple2<>(frequencies, magnitudes);
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to calculate pitch spectrum", e);
        }
    }
    
    /**
     * 检测音调 / Detect pitch
     */
    public double detectPitch(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 获取参数 / Get parameters
            double minFrequency = 80.0;
            double maxFrequency = 1000.0;
            
            if (parameters != null) {
                if (parameters.containsKey("minFrequency")) {
                    minFrequency = (Double) parameters.get("minFrequency");
                }
                if (parameters.containsKey("maxFrequency")) {
                    maxFrequency = (Double) parameters.get("maxFrequency");
                }
            }
            
            // 使用单声道数据进行分析 / Use mono data for analysis
            IVector<Double> samples = audioData.isMono() ? 
                audioData.getSamples() : 
                AudioProcessor.stereoToMono(audioData).getSamples();
            
            // 使用自相关方法 / Use autocorrelation method
            IVector<Double> autocorr = calculateAutocorrelation(samples);
            
            // 寻找峰值 / Find peaks
            double maxCorr = 0;
            int maxLag = 0;
            
            // 在合理的音调范围内搜索 / Search in reasonable pitch range
            int minLag = (int) (audioData.getSampleRate() / maxFrequency); // 最高音调 / Highest pitch
            int maxLagLimit = (int) (audioData.getSampleRate() / minFrequency); // 最低音调 / Lowest pitch
            
            for (int lag = minLag; lag < Math.min(maxLagLimit, autocorr.length()); lag++) {
                if (autocorr.get(lag) > maxCorr) {
                    maxCorr = autocorr.get(lag);
                    maxLag = lag;
                }
            }
            
            return maxLag > 0 ? audioData.getSampleRate() / maxLag : 0;
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to detect pitch", e);
        }
    }
    
    /**
     * 计算自相关 / Calculate autocorrelation
     */
    private IVector<Double> calculateAutocorrelation(IVector<Double> samples) {
        int length = samples.length();
        IVector<Double> autocorr = Linalg.zeros(length / 2);
        
        for (int lag = 0; lag < autocorr.length(); lag++) {
            double sum = 0;
            for (int i = 0; i < length - lag; i++) {
                sum += samples.get(i) * samples.get(i + lag);
            }
            autocorr.set(lag, sum / (length - lag));
        }
        
        return autocorr;
    }
    
    @Override
    public IAudioAnalyzer clone() {
        return new PitchDetector();
    }
    
    @Override
    public String[] getSupportedFeatureTypes() {
        return SUPPORTED_FEATURE_TYPES.clone();
    }
    
    @Override
    protected int getDefaultFeatureDimension(String featureType) {
        if ("pitch".equalsIgnoreCase(featureType)) {
            return 1; // Single value
        }
        return super.getDefaultFeatureDimension(featureType);
    }
}