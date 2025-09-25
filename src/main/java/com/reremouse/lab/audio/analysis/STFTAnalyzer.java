package com.reremouse.lab.audio.analysis;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioProcessor;
import com.reremouse.lab.audio.analysis.AbstractAudioAnalyzer;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.signal.core.RereFFT;
import com.reremouse.lab.math.signal.core.Complex;
import com.reremouse.lab.util.Tuple2;
import java.util.Map;

/**
 * 短时傅里叶变换分析器实现 / Short-Time Fourier Transform Analyzer Implementation
 * <p>
 * 实现STFT分析功能，将音频信号分解为时频表示。
 * Implements STFT analysis functionality, decomposing audio signal into time-frequency representation.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class STFTAnalyzer extends AbstractAudioAnalyzer {
    
    // Supported feature types for this analyzer
    private static final String[] SUPPORTED_FEATURE_TYPES = {"stft", "spectral_centroid"};

    /**
     * 构造函数 / Constructor
     */
    public STFTAnalyzer() {
        super("stft", "Short-Time Fourier Transform analyzer");
        addSupportedParameter("windowSize", 1024);
        addSupportedParameter("hopSize", 512);
    }
    
    @Override
    protected IVector<Double> doExtractFeatures(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 对于STFT分析器，特征提取返回频谱质心 / For STFT analyzer, feature extraction returns spectral centroid
            Tuple2<IVector<Double>, IVector<Double>> spectrum = calculateSpectrum(input, parameters);
            return spectrum.getSecond(); // 返回幅度 / Return magnitude
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to extract STFT features", e);
        }
    }
    
    @Override
    protected Tuple2<IVector<Double>, IVector<Double>> doCalculateSpectrum(AudioData input, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 获取参数 / Get parameters
            int windowSize = 1024;
            int hopSize = 512;
            
            if (parameters != null) {
                if (parameters.containsKey("windowSize")) {
                    windowSize = (Integer) parameters.get("windowSize");
                }
                if (parameters.containsKey("hopSize")) {
                    hopSize = (Integer) parameters.get("hopSize");
                }
            }
            
            // 计算STFT / Calculate STFT
            IMatrix<Double> stftMatrix = calculateSTFT(input, windowSize, hopSize);
            
            // 为了符合接口要求，返回频率和幅度 / To comply with interface requirements, return frequency and magnitude
            // 这里简化实现，返回第一帧的频谱 / Simplified implementation, return spectrum of first frame
            int numBins = stftMatrix.rows();
            IVector<Double> frequencies = Linalg.zeros(numBins);
            IVector<Double> magnitudes = stftMatrix.getColumn(0);
            
            // 计算频率 / Calculate frequencies
            double sampleRate = input.getSampleRate();
            for (int i = 0; i < numBins; i++) {
                frequencies.set(i, i * sampleRate / (2 * numBins));
            }
            
            return new Tuple2<>(frequencies, magnitudes);
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to calculate STFT spectrum", e);
        }
    }
    
    /**
     * 计算短时傅里叶变换 / Calculate Short-Time Fourier Transform
     */
    public IMatrix<Double> calculateSTFT(AudioData audioData, int windowSize, int hopSize) throws AudioProcessingException {
        try {
            // 使用单声道数据进行分析 / Use mono data for analysis
            IVector<Double> samples = audioData.isMono() ? 
                audioData.getSamples() : 
                AudioProcessor.stereoToMono(audioData).getSamples();
            
            int numFrames = (samples.length() - windowSize) / hopSize + 1;
            int numBins = windowSize / 2 + 1;
            
            IMatrix<Double> stftMatrix = Linalg.zeros(numBins, numFrames);
            
            // 创建汉宁窗 / Create Hanning window
            IVector<Double> window = createHanningWindow(windowSize);
            
            for (int frame = 0; frame < numFrames; frame++) {
                int start = frame * hopSize;
                int end = Math.min(start + windowSize, samples.length());
                
                // 提取窗口信号 / Extract window signal
                IVector<Double> windowSignal = samples.slice(start, end);
                
                // 应用窗函数 / Apply window function
                IVector<Double> windowedSignal = windowSignal.multiply(window);
                
                // 计算FFT / Calculate FFT
                Complex[] input = convertToComplex(windowedSignal);
                // 零填充确保长度为2的幂
                Complex[] paddedInput = RereFFT.zeroPadToPowerOfTwo(input);
                Complex[] fftResult = RereFFT.fft(paddedInput);
                
                // 存储幅度谱 / Store magnitude spectrum
                for (int bin = 0; bin < numBins; bin++) {
                    double magnitude = fftResult[bin].magnitude();
                    stftMatrix.set(bin, frame, magnitude);
                }
            }
            
            return stftMatrix;
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to calculate STFT", e);
        }
    }
    
    /**
     * 创建汉宁窗 / Create Hanning window
     */
    private IVector<Double> createHanningWindow(int windowSize) {
        IVector<Double> window = Linalg.zeros(windowSize);
        for (int i = 0; i < windowSize; i++) {
            window.set(i, 0.5 * (1 - Math.cos(2 * Math.PI * i / (windowSize - 1))));
        }
        return window;
    }
    
    /**
     * 转换为复数数组 / Convert to complex array
     */
    private Complex[] convertToComplex(IVector<Double> samples) {
        int length = samples.length();
        Complex[] complexSamples = new Complex[length];
        for (int i = 0; i < length; i++) {
            complexSamples[i] = new Complex(samples.get(i), 0);
        }
        return complexSamples;
    }
    
    @Override
    public IAudioAnalyzer clone() {
        return new STFTAnalyzer();
    }
    
    @Override
    public String[] getSupportedFeatureTypes() {
        return SUPPORTED_FEATURE_TYPES.clone();
    }
    
    @Override
    protected int getDefaultFeatureDimension(String featureType) {
        if ("stft".equalsIgnoreCase(featureType)) {
            // Default STFT size
            return 513; // For 1024-point FFT
        } else if ("spectral_centroid".equalsIgnoreCase(featureType)) {
            return 1; // Single value
        }
        return super.getDefaultFeatureDimension(featureType);
    }
}