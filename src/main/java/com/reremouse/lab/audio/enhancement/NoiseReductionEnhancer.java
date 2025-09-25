package com.reremouse.lab.audio.enhancement;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import java.util.Map;

/**
 * 降噪增强器实现 / Noise Reduction Enhancer Implementation
 * <p>
 * 实现音频降噪功能，使用谱减法进行音频降噪，基于噪声谱估计。
 * Implements audio noise reduction functionality, using spectral subtraction for audio noise reduction based on noise spectrum estimation.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class NoiseReductionEnhancer extends AbstractAudioEnhancer {
    
    /**
     * 构造函数 / Constructor
     */
    public NoiseReductionEnhancer() {
        super("noise_reduction", "Noise reduction enhancer", EnhancerType.NOISE_REDUCTION);
        addSupportedParameter("noiseThreshold", 0.05);
        addSupportedParameter("attenuationFactor", 0.1);
        addSupportedParameter("frameSize", 1024);
        addSupportedParameter("hopSize", 512);
    }
    
    @Override
    protected AudioData doEnhance(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 获取参数 / Get parameters
            double noiseThreshold = 0.05;
            double attenuationFactor = 0.1;
            int frameSize = 1024;
            int hopSize = 512;
            
            if (parameters != null) {
                if (parameters.containsKey("noiseThreshold")) {
                    noiseThreshold = (Double) parameters.get("noiseThreshold");
                }
                if (parameters.containsKey("attenuationFactor")) {
                    attenuationFactor = (Double) parameters.get("attenuationFactor");
                }
                if (parameters.containsKey("frameSize")) {
                    frameSize = (Integer) parameters.get("frameSize");
                }
                if (parameters.containsKey("hopSize")) {
                    hopSize = (Integer) parameters.get("hopSize");
                }
            }
            
            IVector<Double> samples = input.getSamples();
            IVector<Double> enhancedSamples = Linalg.zeros(samples.length());
            
            // 分帧处理 / Frame-based processing
            int numFrames = (samples.length() - frameSize) / hopSize + 1;
            
            for (int frame = 0; frame < numFrames; frame++) {
                int start = frame * hopSize;
                int end = Math.min(start + frameSize, samples.length());
                
                // 提取帧 / Extract frame
                IVector<Double> frameData = samples.slice(start, end);
                
                // 应用窗函数 / Apply window function
                IVector<Double> window = createHanningWindow(frameSize);
                IVector<Double> windowedFrame = frameData.multiply(window);
                
                // 谱减法降噪 / Spectral subtraction noise reduction
                IVector<Double> enhancedFrame = spectralSubtraction(windowedFrame, noiseThreshold, attenuationFactor);
                
                // 重叠相加 / Overlap-add
                for (int i = 0; i < enhancedFrame.length(); i++) {
                    int idx = start + i;
                    if (idx < enhancedSamples.length()) {
                        enhancedSamples.set(idx, enhancedSamples.get(idx) + enhancedFrame.get(i));
                    }
                }
            }
            
            return new AudioData(enhancedSamples, input.getSampleRate(), input.getChannels(), 
                               input.getBitDepth(), input.getFormat());
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to reduce noise", e);
        }
    }
    
    /**
     * 谱减法降噪 / Spectral subtraction noise reduction
     *
     * @param frame 音频帧 / Audio frame
     * @param noiseThreshold 噪声阈值 / Noise threshold
     * @param attenuationFactor 衰减因子 / Attenuation factor
     * @return 降噪后的帧 / Noise-reduced frame
     */
    private IVector<Double> spectralSubtraction(IVector<Double> frame, double noiseThreshold, double attenuationFactor) {
        // 简化的谱减法实现 / Simplified spectral subtraction implementation
        IVector<Double> enhancedFrame = Linalg.zeros(frame.length());
        
        for (int i = 0; i < frame.length(); i++) {
            double sample = frame.get(i);
            
            // 如果样本低于噪声阈值，则衰减 / If sample is below noise threshold, attenuate
            if (Math.abs(sample) < noiseThreshold) {
                enhancedFrame.set(i, sample * attenuationFactor); // 衰减 / Attenuate
            } else {
                enhancedFrame.set(i, sample);
            }
        }
        
        return enhancedFrame;
    }
    
    /**
     * 创建汉宁窗 / Create Hanning window
     *
     * @param size 窗函数大小 / Window size
     * @return 汉宁窗向量 / Hanning window vector
     */
    private IVector<Double> createHanningWindow(int size) {
        IVector<Double> window = Linalg.zeros(size);
        for (int i = 0; i < size; i++) {
            double value = 0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1)));
            window.set(i, value);
        }
        return window;
    }
    
    @Override
    public NoiseReductionEnhancer clone() {
        return new NoiseReductionEnhancer();
    }
}