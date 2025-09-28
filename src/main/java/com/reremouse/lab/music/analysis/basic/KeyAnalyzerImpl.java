package com.reremouse.lab.music.analysis.basic;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.music.theory.KeyTheory;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.core.RereFFT;
import com.reremouse.lab.math.signal.core.Complex;
import com.reremouse.lab.music.analysis.StandardizedConfidenceCalculator;

import java.util.Map;
import java.util.HashMap;

/**
 * 调性分析器实现 / Key Analyzer Implementation
 * <p>
 * 基于色度特征的调性检测实现。
 * Key detection implementation based on chroma features.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class KeyAnalyzerImpl implements IKeyAnalyzer {

    // 默认参数 / Default parameters
    private static final int DEFAULT_WINDOW_SIZE = 8192;
    private static final int DEFAULT_HOP_SIZE = 2048;
    private static final double DEFAULT_CONFIDENCE_THRESHOLD = 0.6;

    // 色度特征维度 / Chroma feature dimensions
    private static final int CHROMA_BINS = 12;

    // 标准化置信度计算器 / Standardized confidence calculator
    private final StandardizedConfidenceCalculator confidenceCalculator = new StandardizedConfidenceCalculator();
    
    // 改进的调性模板 / Improved key templates
    // 使用更精确的权重而不是简单的0/1
    // Use more precise weights instead of simple 0/1
    private static final double[] MAJOR_TEMPLATE = {1.0, 0.0, 0.8, 0.0, 0.9, 0.7, 0.0, 1.0, 0.0, 0.8, 0.0, 0.6};
    private static final double[] MINOR_TEMPLATE = {1.0, 0.0, 0.6, 0.9, 0.0, 0.8, 0.0, 1.0, 0.7, 0.0, 0.8, 0.0};
    
    // 流行音乐专用调性模板 / Pop music specific key templates
    private static final double[] POP_MAJOR_TEMPLATE = {1.0, 0.0, 0.7, 0.0, 0.9, 0.8, 0.0, 1.0, 0.0, 0.7, 0.0, 0.5};
    private static final double[] POP_MINOR_TEMPLATE = {1.0, 0.0, 0.5, 0.9, 0.0, 0.8, 0.0, 1.0, 0.6, 0.0, 0.7, 0.0};
    
    // 添加更多调性模板 / Add more key templates
    private static final double[] DORIAN_TEMPLATE = {1.0, 0.0, 0.8, 0.7, 0.0, 0.9, 0.0, 1.0, 0.0, 0.8, 0.6, 0.0};
    private static final double[] MIXOLYDIAN_TEMPLATE = {1.0, 0.0, 0.8, 0.0, 0.9, 0.7, 0.0, 1.0, 0.0, 0.8, 0.6, 0.0};
    
    // 修改布鲁斯模板，降低其匹配权重 / Modified blues template with reduced matching weight
    private static final double[] BLUES_TEMPLATE = {1.0, 0.0, 0.0, 0.6, 0.0, 0.7, 0.5, 1.0, 0.0, 0.0, 0.4, 0.0};

    @Override
    public KeyDetectionResult detectKey(AudioData audioData) throws AudioProcessingException {
        return detectKey(audioData, getDefaultParameters());
    }

    @Override
    public KeyDetectionResult detectKey(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        try {
            // 计算色度特征 / Calculate chroma features
            double[] chromaFeatures = analyzeChromaFeatures(audioData, parameters);

            // 与调性模板匹配 / Match with key templates
            KeyMatchResult bestMatch = findBestKeyMatch(chromaFeatures);

            // 计算置信度 / Calculate confidence
            double confidence = calculateImprovedConfidence(chromaFeatures, bestMatch.confidence);
            // 确保置信度不为零 / Ensure confidence is not zero
            if (confidence <= 0.0) {
                confidence = Math.max(0.1, bestMatch.confidence); // Use at least 0.1 or the original score
            }

            // 创建结果 / Create result
            KeyDetectionResult result = new KeyDetectionResult();
            result.setKeyName(bestMatch.keyName);
            result.setMode(bestMatch.mode);
            result.setConfidence(confidence); // Use calculated confidence
            result.setChromaFeatures(chromaFeatures);
            result.setAlgorithm("chroma_template");

            return result;

        } catch (Exception e) {
            throw new AudioProcessingException("Error in key detection: " + e.getMessage(), e);
        }
    }

    @Override
    public double[] analyzeChromaFeatures(AudioData audioData) throws AudioProcessingException {
        return analyzeChromaFeatures(audioData, getDefaultParameters());
    }

    @Override
    public double[] analyzeChromaFeatures(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        if (audioData == null) {
            throw new AudioProcessingException("Audio data cannot be null");
        }

        // 获取参数 / Get parameters
        // Fix: Safely convert parameters to prevent ClassCastException
        int windowSize = getIntegerParameter(parameters, "windowSize", DEFAULT_WINDOW_SIZE);
        int hopSize = getIntegerParameter(parameters, "hopSize", DEFAULT_HOP_SIZE);
        
        // 改进参数设置，使用更适合音乐分析的窗口大小
        // Improve parameter settings with window sizes more suitable for music analysis
        if (windowSize <= 0) {
            windowSize = 8192; // 增大窗口以获得更好的频率分辨率
        }
        if (hopSize <= 0) {
            hopSize = windowSize / 4; // 更小的跳跃以获得更好的时间分辨率
        }
        if (hopSize > windowSize) {
            hopSize = windowSize / 4;
        }

        try {
            IVector<Double> signal = audioData.getSamples();
            double sampleRate = audioData.getSampleRate();

            // 使用改进的色度特征计算方法
            // Use improved chroma feature calculation method
            double[] chromaFeatures = computeImprovedChromaFeatures(signal, sampleRate, windowSize, hopSize);

            return chromaFeatures;

        } catch (Exception e) {
            throw new AudioProcessingException("Error in chroma analysis: " + e.getMessage(), e);
        }
    }

    @Override
    public String[] getSupportedParameters() {
        return new String[]{"windowSize", "hopSize", "confidenceThreshold"};
    }

    @Override
    public Map<String, Object> getDefaultParameters() {
        Map<String, Object> params = new HashMap<>();
        params.put("windowSize", DEFAULT_WINDOW_SIZE);
        params.put("hopSize", DEFAULT_HOP_SIZE);
        params.put("confidenceThreshold", DEFAULT_CONFIDENCE_THRESHOLD);
        return params;
    }

    /**
     * 设置分析器参数 / Set analyzer parameters
     *
     * @param parameters 要设置的参数 / Parameters to set
     * @throws AudioProcessingException 参数无效时抛出异常 / Thrown when parameters are invalid
     */
    public void setParameters(Map<String, Object> parameters) throws AudioProcessingException {
        if (parameters == null) {
            return;
        }

        // 验证参数有效性 / Validate parameters
        for (String key : parameters.keySet()) {
            if (!java.util.Arrays.asList(getSupportedParameters()).contains(key)) {
                throw new AudioProcessingException("Unsupported parameter: " + key);
            }
        }

        // 这里可以添加参数验证逻辑，但由于当前实现使用方法参数传递，
        // 暂时只做基础验证 / Basic validation for now since current implementation uses method parameters
    }

    /**
     * 计算频谱 / Compute spectrum
     */
    private Complex[] computeSpectrum(IVector<Double> signal, int windowSize) {
        // 添加保护性检查
        // Add protective checks
        if (windowSize <= 0) {
            windowSize = DEFAULT_WINDOW_SIZE;
        }
        
        // 确保窗口大小不超过信号长度
        // Ensure window size does not exceed signal length
        if (windowSize > signal.length()) {
            windowSize = signal.length();
        }
        
        // 如果信号为空或窗口大小为0，返回空频谱
        // If signal is empty or window size is 0, return empty spectrum
        if (signal.length() <= 0 || windowSize <= 0) {
            return new Complex[1];
        }

        // 应用窗函数 / Apply window function
        IVector<Double> windowed = applyHammingWindow(signal, windowSize);

        // Convert IVector<Double> to Complex[] for FFT
        Complex[] input = new Complex[windowed.length()];
        for (int i = 0; i < windowed.length(); i++) {
            input[i] = new Complex(windowed.get(i), 0.0);
        }

        // 确保输入长度是2的幂
        // Ensure input length is a power of 2
        Complex[] paddedInput = com.reremouse.lab.math.signal.core.RereFFT.zeroPadToPowerOfTwo(input);

        // 计算FFT / Compute FFT
        return com.reremouse.lab.math.signal.core.RereFFT.fft(paddedInput);
    }

    /**
     * 应用汉明窗 / Apply Hamming window
     */
    private IVector<Double> applyHammingWindow(IVector<Double> signal, int windowSize) {
        // 添加保护性检查
        // Add protective checks
        if (windowSize <= 0) {
            windowSize = DEFAULT_WINDOW_SIZE;
        }
        
        // 确保窗口大小不超过信号长度
        // Ensure window size does not exceed signal length
        windowSize = Math.min(signal.length(), windowSize);
        
        // 确保窗口大小是2的幂
        // Ensure window size is a power of 2
        windowSize = nextPowerOfTwo(windowSize);
        
        // 确保窗口大小至少为1
        // Ensure window size is at least 1
        windowSize = Math.max(1, windowSize);
        
        IVector<Double> windowed = Linalg.zeros(windowSize);

        // 使用实际的窗口大小来应用窗函数
        // Use actual window size to apply window function
        int actualLength = Math.min(signal.length(), windowSize);
        for (int i = 0; i < actualLength; i++) {
            double window = 0.54 - 0.46 * Math.cos(2.0 * Math.PI * i / (actualLength - 1));
            windowed.set(i, signal.get(i) * window);
        }

        return windowed;
    }
    
    /**
     * 计算大于等于n的最小2的幂 / Calculate smallest power of 2 >= n
     */
    private static int nextPowerOfTwo(int n) {
        if (n <= 0) return 1;
        if ((n & (n - 1)) == 0) return n;
        
        int power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }

    /**
     * 改进的色度特征计算方法 / Improved chroma feature calculation method
     */
    private double[] computeImprovedChromaFeatures(IVector<Double> signal, double sampleRate, int windowSize, int hopSize) {
        return computeAdvancedChromaFeatures(signal, sampleRate, windowSize, hopSize);
    }

    /**
     * 高级色度特征计算 / Advanced chroma feature calculation
     */
    private double[] computeAdvancedChromaFeatures(IVector<Double> signal, double sampleRate, int windowSize, int hopSize) {
        double[] globalChroma = new double[CHROMA_BINS];
        double[] weightedChroma = new double[CHROMA_BINS];
        int frameCount = 0;
        double totalWeight = 0.0;
        
        // 确保参数有效
        // Ensure parameters are valid
        if (windowSize <= 0 || hopSize <= 0 || signal.length() <= 0) {
            // 返回零数组而不是均匀分布
            // Return zero array instead of uniform distribution
            for (int i = 0; i < CHROMA_BINS; i++) {
                globalChroma[i] = 0.0;
            }
            return globalChroma;
        }
        
        // 确保窗口大小不超过信号长度
        // Ensure window size does not exceed signal length
        windowSize = Math.min(windowSize, signal.length());
        
        // 使用更多帧来获得更稳定的结果
        // Use more frames for more stable results
        int maxFrames = Math.min(200, Math.max(1, (signal.length() - windowSize) / hopSize + 1));
        
        // 至少处理一帧
        // Process at least one frame
        if (maxFrames <= 0) {
            maxFrames = 1;
        }
        
        // 分帧处理音频信号
        // Process audio signal frame by frame
        for (int start = 0; start + windowSize <= signal.length() && frameCount < maxFrames; start += hopSize) {
            // 提取当前帧
            // Extract current frame
            IVector<Double> frame = extractFrame(signal, start, windowSize);
            
            // 计算当前帧的能量，用于加权
            // Calculate frame energy for weighting
            double frameEnergy = calculateFrameEnergy(frame);
            
            // 计算当前帧的频谱
            // Calculate spectrum for current frame
            Complex[] spectrum = computeSpectrum(frame, windowSize);
            
            // 计算当前帧的色度特征
            // Calculate chroma features for current frame
            double[] frameChroma = computeEnhancedFrameChromaFeatures(spectrum, sampleRate, windowSize);
            
            // 应用能量加权
            // Apply energy weighting
            double weight = Math.log(1 + frameEnergy * 1000); // 对数加权
            
            // 累加到加权色度特征
            // Accumulate to weighted chroma features
            for (int i = 0; i < CHROMA_BINS; i++) {
                weightedChroma[i] += frameChroma[i] * weight;
            }
            totalWeight += weight;
            frameCount++;
        }
        
        // 如果没有处理任何帧，尝试处理整个信号
        // If no frames were processed, try processing the entire signal
        if (frameCount == 0) {
            IVector<Double> frame = extractFrame(signal, 0, Math.min(windowSize, signal.length()));
            Complex[] spectrum = computeSpectrum(frame, windowSize);
            double[] frameChroma = computeEnhancedFrameChromaFeatures(spectrum, sampleRate, windowSize);
            
            // 简单复制，不加权
            // Simple copy, no weighting
            System.arraycopy(frameChroma, 0, globalChroma, 0, CHROMA_BINS);
        } else {
            // 计算加权平均
            // Calculate weighted average
            if (totalWeight > 0) {
                for (int i = 0; i < CHROMA_BINS; i++) {
                    globalChroma[i] = weightedChroma[i] / totalWeight;
                }
            }
        }
        
        // 应用高级后处理
        // Apply advanced post-processing
        globalChroma = applyAdvancedChromaPostProcessing(globalChroma);
        
        return globalChroma;
    }

    /**
     * 计算帧能量 / Calculate frame energy
     */
    private double calculateFrameEnergy(IVector<Double> frame) {
        double energy = 0.0;
        for (int i = 0; i < frame.length(); i++) {
            double sample = frame.get(i);
            energy += sample * sample;
        }
        return energy / frame.length();
    }

    /**
     * 增强的单帧色度特征计算 / Enhanced single frame chroma feature calculation
     */
    private double[] computeEnhancedFrameChromaFeatures(Complex[] spectrum, double sampleRate, int windowSize) {
        double[] chroma = new double[CHROMA_BINS];
        
        if (spectrum == null || spectrum.length == 0) {
            return chroma;
        }
        
        // 确保spectrumLength不会超过实际频谱长度
        // Ensure spectrumLength doesn't exceed actual spectrum length
        int spectrumLength = Math.min(spectrum.length / 2, windowSize / 2); // 只使用正频率部分
        
        // 添加保护性检查
        // Add protective checks
        if (spectrumLength <= 1) {
            return chroma;
        }
        
        // 使用多个倍频程进行色度计算
        // Use multiple octaves for chroma calculation
        for (int octave = 0; octave <= 6; octave++) { // 覆盖更多倍频程
            double octaveWeight = 1.0 / (octave + 1); // 高倍频程权重递减
            
            for (int i = 1; i < spectrumLength; i++) {
                double frequency = (i * sampleRate) / windowSize;
                double magnitude = spectrum[i].magnitude();
                
                // 只处理音乐频率范围内的信号 (20Hz - 20000Hz)
                // Only process signals within musical frequency range (20Hz - 20000Hz)
                if (frequency >= 20.0 && frequency <= 20000.0 && magnitude > 1e-10) {
                    int chromaClass = improvedFrequencyToChromaClass(frequency);
                    if (chromaClass >= 0 && chromaClass < CHROMA_BINS) {
                        // 使用对数幅度和倍频程加权
                        // Use logarithmic magnitude and octave weighting
                        double logMagnitude = Math.log(1 + magnitude * 1000);
                        chroma[chromaClass] += logMagnitude * octaveWeight;
                    }
                }
            }
        }
        
        return chroma;
    }

    /**
     * 高级色度后处理 / Advanced chroma post-processing
     */
    private double[] applyAdvancedChromaPostProcessing(double[] chroma) {
        if (chroma == null || chroma.length != CHROMA_BINS) {
            return new double[CHROMA_BINS];
        }

        double[] processed = chroma.clone();

        // 1. 平滑处理 - 减少噪声
        // 1. Smoothing - reduce noise
        processed = applyChromaticSmoothing(processed);

        // 2. 增强主要音符
        // 2. Enhance dominant notes
        processed = enhanceDominantNotes(processed);

        // 3. 归一化
        // 3. Normalization
        processed = normalizeChromaFeaturesAdvanced(processed);

        return processed;
    }

    /**
     * 应用色度平滑 / Apply chromatic smoothing
     */
    private double[] applyChromaticSmoothing(double[] chroma) {
        double[] smoothed = new double[CHROMA_BINS];
        
        // 使用简单的3点平滑
        // Use simple 3-point smoothing
        for (int i = 0; i < CHROMA_BINS; i++) {
            int prev = (i - 1 + CHROMA_BINS) % CHROMA_BINS;
            int next = (i + 1) % CHROMA_BINS;
            
            smoothed[i] = 0.25 * chroma[prev] + 0.5 * chroma[i] + 0.25 * chroma[next];
        }
        
        return smoothed;
    }

    /**
     * 增强主要音符 / Enhance dominant notes
     */
    private double[] enhanceDominantNotes(double[] chroma) {
        double[] enhanced = chroma.clone();
        
        // 找到最大值
        // Find maximum value
        double maxValue = 0.0;
        for (double value : chroma) {
            maxValue = Math.max(maxValue, value);
        }
        
        if (maxValue > 0) {
            // 计算平均值和阈值
            // Calculate average and threshold
            double average = 0.0;
            for (double value : chroma) {
                average += value;
            }
            average /= CHROMA_BINS;
            
            // 使用更高的阈值来识别真正的峰值
            // Use higher threshold to identify true peaks
            double threshold = average + (maxValue - average) * 0.3;
            
            for (int i = 0; i < CHROMA_BINS; i++) {
                if (enhanced[i] > threshold) {
                    // 增强峰值：使用大于1的指数来增强对比度
                    // Enhance peaks: use exponent > 1 to increase contrast
                    double normalizedValue = enhanced[i] / maxValue;
                    enhanced[i] = Math.pow(normalizedValue, 0.5) * maxValue * 1.2;
                } else if (enhanced[i] < average) {
                    // 轻微抑制低于平均值的音符
                    // Slightly suppress notes below average
                    enhanced[i] *= 0.8;
                }
            }
        }
        
        return enhanced;
    }

    /**
     * 高级色度特征归一化 / Advanced chroma feature normalization
     */
    private double[] normalizeChromaFeaturesAdvanced(double[] chroma) {
        if (chroma == null || chroma.length == 0) {
            // Return zero array instead of uniform distribution
            double[] zeroChroma = new double[CHROMA_BINS];
            return zeroChroma;
        }

        // Create a copy to avoid modifying the original array
        double[] normalized = chroma.clone();

        // 计算总能量和最大值
        // Calculate total energy and maximum value
        double totalEnergy = 0.0;
        double maxValue = 0.0;
        
        for (double value : normalized) {
            totalEnergy += value;
            maxValue = Math.max(maxValue, value);
        }

        // 使用L2归一化策略
        // Use L2 normalization strategy
        if (totalEnergy > 1e-10) {
            // 计算L2范数
            // Calculate L2 norm
            double l2Norm = 0.0;
            for (double value : normalized) {
                l2Norm += value * value;
            }
            l2Norm = Math.sqrt(l2Norm);
            
            // 应用L2归一化
            // Apply L2 normalization
            for (int i = 0; i < normalized.length; i++) {
                normalized[i] /= l2Norm;
            }
        } else {
            // 如果没有有效信号，返回零数组而不是均匀分布
            // If no valid signal, return zero array instead of uniform distribution
            for (int i = 0; i < normalized.length; i++) {
                normalized[i] = 0.0;
            }
        }
        
        return normalized;
    }
    
    /**
     * 提取音频帧 / Extract audio frame
     */
    private IVector<Double> extractFrame(IVector<Double> signal, int start, int windowSize) {
        IVector<Double> frame = Linalg.zeros(windowSize);
        int end = Math.min(start + windowSize, signal.length());
        
        for (int i = 0; i < end - start; i++) {
            frame.set(i, signal.get(start + i));
        }
        
        return frame;
    }
    
    /**
     * 计算单帧的色度特征 / Compute chroma features for a single frame
     */
    private double[] computeFrameChromaFeatures(Complex[] spectrum, double sampleRate, int windowSize) {
        double[] chroma = new double[CHROMA_BINS];
        
        if (spectrum == null || spectrum.length == 0) {
            return chroma;
        }
        
        int spectrumLength = spectrum.length / 2; // 只使用正频率部分
        
        // 改进的频率到色度映射
        // Improved frequency to chroma mapping
        for (int i = 1; i < spectrumLength; i++) {
            double frequency = (i * sampleRate) / windowSize;
            double magnitude = spectrum[i].magnitude();
            
            // 只处理音乐频率范围内的信号 (20Hz - 20000Hz)
            // Only process signals within musical frequency range (20Hz - 20000Hz)
            if (frequency >= 20.0 && frequency <= 20000.0 && magnitude > 1e-10) {
                int chromaClass = improvedFrequencyToChromaClass(frequency);
                if (chromaClass >= 0 && chromaClass < CHROMA_BINS) {
                    // 使用对数幅度以更好地表示音乐信号
                    // Use logarithmic magnitude for better musical signal representation
                    chroma[chromaClass] += Math.log(1 + magnitude * 1000);
                }
            }
        }
        
        return chroma;
    }

    /**
     * 改进的频率到色度类映射 / Improved frequency to chroma class mapping
     */
    private int improvedFrequencyToChromaClass(double frequency) {
        if (frequency <= 0 || !Double.isFinite(frequency)) {
            return -1;
        }

        // 使用更精确的MIDI音符计算
        // Use more precise MIDI note calculation
        // A4 = 440Hz = MIDI note 69
        double midiNote = 12.0 * (Math.log(frequency / 440.0) / Math.log(2.0)) + 69.0;
        
        if (!Double.isFinite(midiNote)) {
            return -1;
        }

        // 映射到色度类，使用四舍五入而不是截断
        // Map to chroma class using rounding instead of truncation
        int chromaClass = ((int) Math.round(midiNote)) % 12;
        
        // 确保结果在0-11范围内
        // Ensure result is within 0-11 range
        if (chromaClass < 0) {
            chromaClass += 12;
        }
        
        return chromaClass;
    }

    /**
     * 改进的色度特征归一化 / Improved chroma feature normalization
     */
    private void normalizeChromaFeaturesImproved(double[] chroma) {
        if (chroma == null || chroma.length == 0) {
            return;
        }

        // 计算总能量
        // Calculate total energy
        double totalEnergy = 0.0;
        for (double value : chroma) {
            totalEnergy += value;
        }

        // 如果有有效的能量分布，进行归一化
        // If there's valid energy distribution, normalize
        if (totalEnergy > 1e-10) { // 使用更小的阈值
            for (int i = 0; i < chroma.length; i++) {
                chroma[i] /= totalEnergy;
            }
        } else {
            // 如果没有有效信号，不设置为均匀分布，而是保持为零
            // If no valid signal, keep as zero instead of uniform distribution
            // 这样可以让调性检测算法知道没有足够的信息
            // This allows the key detection algorithm to know there's insufficient information
        }
    }

    /**
     * 计算色度特征 / Compute chroma features
     */
    private double[] computeChromaFeatures(Complex[] spectrum, double sampleRate, int windowSize) {
        // 添加保护性检查
        // Add protective checks
        if (spectrum == null || spectrum.length == 0) {
            return new double[CHROMA_BINS];
        }
        
        if (sampleRate <= 0) {
            sampleRate = 44100.0; // 默认采样率
        }
        
        if (windowSize <= 0) {
            windowSize = DEFAULT_WINDOW_SIZE;
        }

        double[] chroma = new double[CHROMA_BINS];
        int spectrumLength = spectrum.length / 2; // 只使用正频率部分 / Only use positive frequencies
        
        // 确保我们有有效的频谱数据
        // Ensure we have valid spectrum data
        if (spectrumLength <= 1) {
            return chroma;
        }

        // 添加一个标志来检查是否有任何值被添加到色度特征中
        // Add a flag to check if any values were added to chroma features
        boolean hasValues = false;

        for (int i = 1; i < spectrumLength; i++) {
            double frequency = (i * sampleRate) / windowSize;
            double magnitude = spectrum[i].magnitude();

            // 将频率映射到色度类 / Map frequency to chroma class
            int chromaClass = frequencyToChromaClass(frequency);
            if (chromaClass >= 0 && chromaClass < CHROMA_BINS) {
                chroma[chromaClass] += magnitude;
                hasValues = true;
            }
        }

        // 如果没有添加任何值，尝试使用不同的方法
        // If no values were added, try a different approach
        if (!hasValues && spectrumLength > 1) {
            // 使用简化的色度计算方法
            // Use a simplified chroma calculation method
            for (int i = 1; i < Math.min(spectrumLength, 100); i++) {
                double frequency = (i * sampleRate) / windowSize;
                double magnitude = spectrum[i].magnitude();
                
                // 确保频率和幅度有效
                // Ensure frequency and magnitude are valid
                if (frequency > 0 && Double.isFinite(magnitude) && magnitude > 0) {
                    int chromaClass = (int) (12 * Math.log(frequency / 440.0) / Math.log(2) + 69) % 12;
                    if (chromaClass >= 0 && chromaClass < CHROMA_BINS) {
                        chroma[chromaClass] += magnitude;
                    }
                }
            }
        }

        return chroma;
    }

    /**
     * 将频率映射到色度类 / Map frequency to chroma class
     */
    private int frequencyToChromaClass(double frequency) {
        // 添加保护性检查以防止无效频率
        // Add protective checks to prevent invalid frequencies
        if (frequency <= 0 || !Double.isFinite(frequency)) {
            return -1;
        }

        // 计算MIDI音符号 / Calculate MIDI note number
        double midiNote = 12.0 * Math.log(frequency / 440.0) / Math.log(2.0) + 69.0;
        
        // 检查MIDI音符是否有效
        // Check if MIDI note is valid
        if (!Double.isFinite(midiNote)) {
            return -1;
        }

        // 映射到色度类 (0-11) / Map to chroma class (0-11)
        int chromaClass = (int) Math.round(midiNote) % 12;
        // 确保结果在有效范围内
        // Ensure result is within valid range
        return chromaClass >= 0 ? chromaClass : -1;
    }

    /**
     * 归一化色度特征 / Normalize chroma features
     */
    private void normalizeChromaFeatures(double[] chroma) {
        // 添加保护性检查
        // Add protective checks
        if (chroma == null || chroma.length == 0) {
            return;
        }

        double sum = 0.0;
        for (double value : chroma) {
            // 累加所有值，包括0，以确保正确的归一化
            // Accumulate all values, including 0, to ensure proper normalization
            sum += Math.abs(value);
        }

        // 添加保护性检查以防止除零错误
        // Add protective checks to prevent division by zero
        if (sum > 0) {
            for (int i = 0; i < chroma.length; i++) {
                chroma[i] /= sum;
            }
        } else {
            // 如果所有值都是0，则设置为零数组
            // If all values are 0, set to zero array
            for (int i = 0; i < chroma.length; i++) {
                chroma[i] = 0.0;
            }
        }
    }

    /**
     * 寻找最佳调性匹配 / Find best key match
     */
    private KeyMatchResult findBestKeyMatch(double[] chromaFeatures) {
        return findAdvancedKeyMatch(chromaFeatures);
    }

    /**
     * 高级调性匹配 / Advanced key matching
     */
    private KeyMatchResult findAdvancedKeyMatch(double[] chromaFeatures) {
        // 添加保护性检查
        // Add protective checks
        if (chromaFeatures == null || chromaFeatures.length != CHROMA_BINS) {
            return new KeyMatchResult("C", "major", 0.1); // Return minimum confidence instead of zero
        }

        String[] noteNames = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

        // 检查色度特征的总能量
        // Check total energy of chroma features
        double totalEnergy = 0.0;
        for (double value : chromaFeatures) {
            totalEnergy += value;
        }

        // 如果总能量太低，返回低置信度结果
        // If total energy is too low, return low confidence result
        if (totalEnergy < 1e-8) {
            return new KeyMatchResult("C", "major", 0.1); // Return minimum confidence instead of zero
        }

        // 多模板匹配结果
        // Multi-template matching results
        KeyMatchCandidate[] candidates = new KeyMatchCandidate[12 * 7]; // 12 keys * 7 modes
        int candidateCount = 0;

        // 优先测试流行音乐调性模板 / Prioritize pop music key templates
        for (int root = 0; root < 12; root++) {
            double popMajorScore = calculateAdvancedTemplateMatch(chromaFeatures, POP_MAJOR_TEMPLATE, root);
            candidates[candidateCount++] = new KeyMatchCandidate(noteNames[root], "pop_major", popMajorScore);
        }

        for (int root = 0; root < 12; root++) {
            double popMinorScore = calculateAdvancedTemplateMatch(chromaFeatures, POP_MINOR_TEMPLATE, root);
            candidates[candidateCount++] = new KeyMatchCandidate(noteNames[root], "pop_minor", popMinorScore);
        }

        // 测试传统大调 / Test traditional major keys
        for (int root = 0; root < 12; root++) {
            double score = calculateAdvancedTemplateMatch(chromaFeatures, MAJOR_TEMPLATE, root);
            candidates[candidateCount++] = new KeyMatchCandidate(noteNames[root], "major", score);
        }

        // 测试传统小调 / Test traditional minor keys
        for (int root = 0; root < 12; root++) {
            double score = calculateAdvancedTemplateMatch(chromaFeatures, MINOR_TEMPLATE, root);
            candidates[candidateCount++] = new KeyMatchCandidate(noteNames[root], "minor", score);
        }

        // 测试多利亚调式 / Test Dorian mode
        for (int root = 0; root < 12; root++) {
            double score = calculateAdvancedTemplateMatch(chromaFeatures, DORIAN_TEMPLATE, root);
            candidates[candidateCount++] = new KeyMatchCandidate(noteNames[root], "dorian", score);
        }

        // 测试混合利底亚调式 / Test Mixolydian mode
        for (int root = 0; root < 12; root++) {
            double score = calculateAdvancedTemplateMatch(chromaFeatures, MIXOLYDIAN_TEMPLATE, root);
            candidates[candidateCount++] = new KeyMatchCandidate(noteNames[root], "mixolydian", score);
        }

        // 最后测试布鲁斯调式（降低优先级） / Test Blues mode last (lower priority)
        for (int root = 0; root < 12; root++) {
            double score = calculateAdvancedTemplateMatch(chromaFeatures, BLUES_TEMPLATE, root);
            // 对布鲁斯调式应用额外的惩罚因子
            score *= 0.7; // 降低布鲁斯调式的匹配分数
            candidates[candidateCount++] = new KeyMatchCandidate(noteNames[root], "blues", score);
        }

        // 找到最佳匹配
        // Find best match
        KeyMatchCandidate bestCandidate = findBestCandidate(candidates, candidateCount);

        // 计算改进的置信度
        // Calculate improved confidence
        double confidence = calculateImprovedConfidence(chromaFeatures, bestCandidate.score);
        
        // 确保置信度不为零
        // Ensure confidence is not zero
        if (confidence <= 0.0) {
            confidence = Math.max(0.1, bestCandidate.score); // Use at least 0.1 or the original score
        }

        // 后处理：优化流行音乐调性选择
        bestCandidate = postProcessKeySelection(bestCandidate, candidates, candidateCount);

        return new KeyMatchResult(bestCandidate.keyName, bestCandidate.mode, confidence);
    }

    /**
     * 高级模板匹配 / Advanced template matching
     */
    private double calculateAdvancedTemplateMatch(double[] chromaFeatures, double[] template, int rootShift) {
        if (chromaFeatures == null || template == null || 
            chromaFeatures.length != CHROMA_BINS || template.length != CHROMA_BINS) {
            return 0.0;
        }

        if (rootShift < 0 || rootShift >= CHROMA_BINS) {
            return 0.0;
        }

        // 归一化色度特征
        // Normalize chroma features
        double[] normalizedChroma = normalizeForMatching(chromaFeatures);

        // 计算多种相似度度量的组合
        // Calculate combination of multiple similarity measures
        double cosineSimilarity = calculateCosineSimilarity(normalizedChroma, template, rootShift);
        double correlationSimilarity = calculateCorrelationSimilarity(normalizedChroma, template, rootShift);
        double euclideanSimilarity = calculateEuclideanSimilarity(normalizedChroma, template, rootShift);

        // 加权组合不同的相似度度量
        // Weighted combination of different similarity measures
        double combinedScore = 0.6 * cosineSimilarity + 0.2 * correlationSimilarity + 0.2 * euclideanSimilarity;

        return Math.max(0.0, Math.min(1.0, combinedScore));
    }

    /**
     * 为匹配归一化色度特征 / Normalize chroma features for matching
     */
    private double[] normalizeForMatching(double[] chroma) {
        double[] normalized = chroma.clone();
        
        // 计算L2范数
        // Calculate L2 norm
        double norm = 0.0;
        for (double value : normalized) {
            norm += value * value;
        }
        norm = Math.sqrt(norm);
        
        if (norm > 1e-10) {
            for (int i = 0; i < normalized.length; i++) {
                normalized[i] /= norm;
            }
        }
        
        return normalized;
    }

    /**
     * 计算余弦相似度 / Calculate cosine similarity
     */
    private double calculateCosineSimilarity(double[] chroma, double[] template, int rootShift) {
        double dotProduct = 0.0;
        double chromaNorm = 0.0;
        double templateNorm = 0.0;

        for (int i = 0; i < CHROMA_BINS; i++) {
            int shiftedIndex = (i + rootShift) % CHROMA_BINS;
            dotProduct += chroma[i] * template[shiftedIndex];
            chromaNorm += chroma[i] * chroma[i];
            templateNorm += template[shiftedIndex] * template[shiftedIndex];
        }

        double denominator = Math.sqrt(chromaNorm * templateNorm);
        return denominator > 1e-10 ? dotProduct / denominator : 0.0;
    }

    /**
     * 计算相关性相似度 / Calculate correlation similarity
     */
    private double calculateCorrelationSimilarity(double[] chroma, double[] template, int rootShift) {
        // 计算均值
        // Calculate means
        double chromaMean = 0.0;
        double templateMean = 0.0;
        
        for (int i = 0; i < CHROMA_BINS; i++) {
            chromaMean += chroma[i];
            templateMean += template[(i + rootShift) % CHROMA_BINS];
        }
        chromaMean /= CHROMA_BINS;
        templateMean /= CHROMA_BINS;

        // 计算相关系数
        // Calculate correlation coefficient
        double numerator = 0.0;
        double chromaVar = 0.0;
        double templateVar = 0.0;

        for (int i = 0; i < CHROMA_BINS; i++) {
            double chromaDiff = chroma[i] - chromaMean;
            double templateDiff = template[(i + rootShift) % CHROMA_BINS] - templateMean;
            
            numerator += chromaDiff * templateDiff;
            chromaVar += chromaDiff * chromaDiff;
            templateVar += templateDiff * templateDiff;
        }

        double denominator = Math.sqrt(chromaVar * templateVar);
        return denominator > 1e-10 ? numerator / denominator : 0.0;
    }

    /**
     * 计算欧几里得相似度 / Calculate Euclidean similarity
     */
    private double calculateEuclideanSimilarity(double[] chroma, double[] template, int rootShift) {
        double sumSquaredDiff = 0.0;
        
        for (int i = 0; i < CHROMA_BINS; i++) {
            double diff = chroma[i] - template[(i + rootShift) % CHROMA_BINS];
            sumSquaredDiff += diff * diff;
        }
        
        double euclideanDistance = Math.sqrt(sumSquaredDiff);
        // 转换为相似度 (距离越小，相似度越高)
        // Convert to similarity (smaller distance, higher similarity)
        return 1.0 / (1.0 + euclideanDistance);
    }

    /**
     * 找到最佳候选 / Find best candidate
     */
    private KeyMatchCandidate findBestCandidate(KeyMatchCandidate[] candidates, int count) {
        KeyMatchCandidate best = candidates[0];
        
        for (int i = 1; i < count; i++) {
            if (candidates[i].score > best.score) {
                best = candidates[i];
            }
        }
        
        return best;
    }
    
    /**
     * 后处理调性选择 / Post-process key selection
     * 优化流行音乐调性的选择逻辑，包括等音调替换
     */
    private KeyMatchCandidate postProcessKeySelection(KeyMatchCandidate best, KeyMatchCandidate[] candidates, int count) {
        // 等音调替换处理：将不常用的调性替换为等音的常用调性
        best = processEnharmonicEquivalents(best, candidates, count);
        
        // 如果最佳候选是流行音乐调性，直接返回但简化模式名称
        if (best.mode.startsWith("pop_")) {
            String simplifiedMode = best.mode.replace("pop_", "");
            return new KeyMatchCandidate(best.keyName, simplifiedMode, best.score);
        }
        
        // 检查是否有相近分数的流行音乐调性候选
        for (int i = 0; i < count; i++) {
            KeyMatchCandidate candidate = candidates[i];
            if (candidate.mode.startsWith("pop_")) {
                // 如果流行音乐调性的分数与最佳分数相差不大（在10%以内），优先选择流行音乐调性
                if (candidate.score >= best.score * 0.9) {
                    String simplifiedMode = candidate.mode.replace("pop_", "");
                    return new KeyMatchCandidate(candidate.keyName, simplifiedMode, candidate.score);
                }
            }
        }
        
        // 如果最佳候选是布鲁斯调性，检查是否有更合适的替代
        if (best.mode.equals("blues")) {
            // 寻找相同根音的大调或小调替代
            for (int i = 0; i < count; i++) {
                KeyMatchCandidate candidate = candidates[i];
                if (candidate.keyName.equals(best.keyName) && 
                    (candidate.mode.equals("major") || candidate.mode.equals("minor") || 
                     candidate.mode.equals("pop_major") || candidate.mode.equals("pop_minor"))) {
                    // 如果大调/小调的分数在布鲁斯调性的80%以上，选择大调/小调
                    if (candidate.score >= best.score * 0.8) {
                        String mode = candidate.mode.startsWith("pop_") ? 
                                     candidate.mode.replace("pop_", "") : candidate.mode;
                        return new KeyMatchCandidate(candidate.keyName, mode, candidate.score);
                    }
                }
            }
        }
        
        return best;
    }
    
    /**
     * 等音调替换处理 / Enharmonic equivalent processing
     * 将音乐理论中不常用的调性替换为等音的常用调性
     */
    private KeyMatchCandidate processEnharmonicEquivalents(KeyMatchCandidate best, KeyMatchCandidate[] candidates, int count) {
        // 定义等音调映射（从冷门调性到常用调性）
        // D# major -> Eb major (9个升号 -> 3个降号)
        // A# major -> Bb major (10个升号 -> 2个降号)  
        // G# major -> Ab major (8个升号 -> 4个降号)
        // C# major -> Db major (7个升号 -> 5个降号)
        // F# major -> Gb major (6个升号 -> 6个降号，两者使用频率相近，但Gb在流行音乐中更常见)
        if (best.keyName.equals("D#") && best.mode.equals("major")) {
            // 寻找Eb major候选
            for (int i = 0; i < count; i++) {
                if (candidates[i].keyName.equals("Eb") && candidates[i].mode.equals("major")) {
                    // 如果Eb major的分数在D# major的85%以上，优先选择Eb major
                    if (candidates[i].score >= best.score * 0.85) {
                        return new KeyMatchCandidate("Eb", "major", candidates[i].score);
                    }
                }
            }
        }
        
        if (best.keyName.equals("A#") && best.mode.equals("major")) {
            // 寻找Bb major候选
            for (int i = 0; i < count; i++) {
                if (candidates[i].keyName.equals("Bb") && candidates[i].mode.equals("major")) {
                    if (candidates[i].score >= best.score * 0.85) {
                        return new KeyMatchCandidate("Bb", "major", candidates[i].score);
                    }
                }
            }
        }
        
        if (best.keyName.equals("G#") && best.mode.equals("major")) {
            // 寻找Ab major候选
            for (int i = 0; i < count; i++) {
                if (candidates[i].keyName.equals("Ab") && candidates[i].mode.equals("major")) {
                    if (candidates[i].score >= best.score * 0.85) {
                        return new KeyMatchCandidate("Ab", "major", candidates[i].score);
                    }
                }
            }
        }
        
        if (best.keyName.equals("C#") && best.mode.equals("major")) {
            // 寻找Db major候选
            for (int i = 0; i < count; i++) {
                if (candidates[i].keyName.equals("Db") && candidates[i].mode.equals("major")) {
                    if (candidates[i].score >= best.score * 0.85) {
                        return new KeyMatchCandidate("Db", "major", candidates[i].score);
                    }
                }
            }
        }
        
        // 小调的等音调替换
        // D# minor -> Eb minor
        // A# minor -> Bb minor
        if (best.keyName.equals("D#") && best.mode.equals("minor")) {
            for (int i = 0; i < count; i++) {
                if (candidates[i].keyName.equals("Eb") && candidates[i].mode.equals("minor")) {
                    if (candidates[i].score >= best.score * 0.85) {
                        return new KeyMatchCandidate("Eb", "minor", candidates[i].score);
                    }
                }
            }
        }
        
        if (best.keyName.equals("A#") && best.mode.equals("minor")) {
            for (int i = 0; i < count; i++) {
                if (candidates[i].keyName.equals("Bb") && candidates[i].mode.equals("minor")) {
                    if (candidates[i].score >= best.score * 0.85) {
                        return new KeyMatchCandidate("Bb", "minor", candidates[i].score);
                    }
                }
            }
        }
        
        return best;
    }

    /**
     * 计算高级调性置信度 / Calculate advanced key confidence
     */
    private double calculateAdvancedKeyConfidence(double[] chromaFeatures, KeyMatchCandidate bestCandidate, 
                                                  KeyMatchCandidate[] allCandidates, int candidateCount) {
        if (bestCandidate.score <= 0) {
            return 0.0;
        }

        // 1. 基础匹配分数
        // 1. Base match score
        double baseConfidence = bestCandidate.score;

        // 2. 计算与第二好的候选的差距
        // 2. Calculate gap with second best candidate
        double secondBestScore = 0.0;
        for (int i = 0; i < candidateCount; i++) {
            if (!allCandidates[i].equals(bestCandidate) && allCandidates[i].score > secondBestScore) {
                secondBestScore = allCandidates[i].score;
            }
        }
        
        double scoreDifference = bestCandidate.score - secondBestScore;
        double differenceBonus = Math.min(0.3, scoreDifference * 2.0); // 最多增加30%

        // 3. 色度特征质量评估
        // 3. Chroma feature quality assessment
        double qualityFactor = assessChromaQuality(chromaFeatures);

        // 4. 调性类型置信度调整
        // 4. Key type confidence adjustment
        double modeConfidence = getModeConfidenceMultiplier(bestCandidate.mode);

        // 综合置信度计算
        // Combined confidence calculation
        double finalConfidence = baseConfidence * qualityFactor * modeConfidence + differenceBonus;

        // 确保置信度在合理范围内
        // Ensure confidence is within reasonable range
        return Math.max(0.1, Math.min(1.0, finalConfidence)); // Minimum 0.1 to avoid zero confidence
    }

    /**
     * 评估色度特征质量 / Assess chroma feature quality
     */
    private double assessChromaQuality(double[] chroma) {
        // 计算色度特征的清晰度和一致性
        // Calculate clarity and consistency of chroma features
        double maxValue = 0.0;
        double totalEnergy = 0.0;
        
        for (double value : chroma) {
            maxValue = Math.max(maxValue, value);
            totalEnergy += value;
        }
        
        if (totalEnergy <= 0) {
            return 0.1; // Return minimum quality instead of zero
        }
        
        // 峰值比 - 主要音符的突出程度
        // Peak ratio - prominence of main notes
        double peakRatio = maxValue / (totalEnergy / CHROMA_BINS);
        
        // 能量分布的不均匀性 (好的调性应该有明显的峰值)
        // Energy distribution unevenness (good tonality should have clear peaks)
        double variance = 0.0;
        double mean = totalEnergy / CHROMA_BINS;
        for (double value : chroma) {
            variance += (value - mean) * (value - mean);
        }
        variance /= CHROMA_BINS;
        
        double clarity = Math.min(1.0, peakRatio / 3.0);
        double consistency = Math.min(1.0, Math.sqrt(variance) / mean);
        
        // Return a minimum quality of 0.1 to avoid zero confidence
        return Math.max(0.1, 0.7 * clarity + 0.3 * consistency);
    }

    /**
     * 获取调式置信度乘数 / Get mode confidence multiplier
     */
    private double getModeConfidenceMultiplier(String mode) {
        switch (mode.toLowerCase()) {
            case "pop_major":
            case "pop_minor":
                return 1.1; // 流行音乐调性，提高置信度
            case "major":
            case "minor":
                return 1.0; // 标准调式，不调整
            case "dorian":
            case "mixolydian":
                return 0.9; // 教会调式，略微降低置信度
            case "blues":
                return 0.6; // 布鲁斯调式，大幅降低置信度（减少误判）
            default:
                return 0.7; // 未知调式
        }
    }

    /**
     * 调性匹配候选内部类 / Key match candidate inner class
     */
    private static class KeyMatchCandidate {
        final String keyName;
        final String mode;
        final double score;

        KeyMatchCandidate(String keyName, String mode, double score) {
            this.keyName = keyName;
            this.mode = mode;
            this.score = score;
        }
    }
    
    /**
     * 改进的置信度计算 / Improved confidence calculation
     */
    private double calculateImprovedConfidence(double[] chromaFeatures, double bestScore) {
        // 使用标准化置信度计算器 / Use standardized confidence calculator
        return confidenceCalculator.calculateFeatureBasedConfidence(chromaFeatures, bestScore);
    }

    /**
     * 计算模板匹配度 / Calculate template match score
     */
    private double calculateTemplateMatch(double[] chroma, double[] template, int rootShift) {
        // 添加保护性检查
        // Add protective checks
        if (chroma == null || template == null) {
            return 0.0;
        }
        
        if (chroma.length != CHROMA_BINS || template.length != CHROMA_BINS) {
            return 0.0;
        }
        
        // 检查rootShift是否有效
        // Check if rootShift is valid
        if (rootShift < 0 || rootShift >= CHROMA_BINS) {
            rootShift = 0;
        }

        double dotProduct = 0.0;
        double chromaNorm = 0.0;
        double templateNorm = 0.0;

        for (int i = 0; i < CHROMA_BINS; i++) {
            int templateIndex = (i - rootShift + CHROMA_BINS) % CHROMA_BINS;
            dotProduct += chroma[i] * template[templateIndex];
            chromaNorm += chroma[i] * chroma[i];
            templateNorm += template[templateIndex] * template[templateIndex];
        }

        // 计算余弦相似度 / Calculate cosine similarity
        // 添加保护性检查以防止除零错误
        // Add protective checks to prevent division by zero
        if (chromaNorm > 0 && templateNorm > 0) {
            double cosineSimilarity = dotProduct / (Math.sqrt(chromaNorm) * Math.sqrt(templateNorm));
            // 确保返回值在[0,1]范围内
            // Ensure return value is within [0,1] range
            return Math.max(0.0, Math.min(1.0, cosineSimilarity));
        }

        return 0.0;
    }

    /**
     * 调性匹配结果内部类 / Key match result inner class
     */
    private static class KeyMatchResult {
        final String keyName;
        final String mode;
        final double confidence;

        KeyMatchResult(String keyName, String mode, double confidence) {
            this.keyName = keyName;
            this.mode = mode;
            // 确保置信度在合理范围内
            // Ensure confidence is within reasonable range
            this.confidence = Math.max(0.0, Math.min(1.0, confidence));
        }
    }

    /**
     * 安全地从参数中获取int值 / Safely get int value from parameters
     */
    private int getIntegerParameter(Map<String, Object> parameters, String key, int defaultValue) {
        if (parameters == null) {
            return defaultValue;
        }
        
        Object value = parameters.get(key);
        if (value == null) {
            return defaultValue;
        }
        
        // Handle different numeric types
        if (value instanceof Integer) {
            return (Integer) value;
        } else if (value instanceof Double) {
            return ((Double) value).intValue();
        } else if (value instanceof Float) {
            return ((Float) value).intValue();
        } else if (value instanceof Long) {
            return ((Long) value).intValue();
        } else if (value instanceof String) {
            try {
                return Integer.parseInt((String) value);
            } catch (NumberFormatException e) {
                return defaultValue;
            }
        } else {
            // Try to convert to int
            try {
                return ((Number) value).intValue();
            } catch (Exception e) {
                return defaultValue;
            }
        }
    }

    /**
     * 安全地从参数中获取double值 / Safely get double value from parameters
     */
    private double getDoubleParameter(Map<String, Object> parameters, String key, double defaultValue) {
        if (parameters == null) {
            return defaultValue;
        }
        
        Object value = parameters.get(key);
        if (value == null) {
            return defaultValue;
        }
        
        // Handle different numeric types
        if (value instanceof Double) {
            return (Double) value;
        } else if (value instanceof Integer) {
            return ((Integer) value).doubleValue();
        } else if (value instanceof Float) {
            return ((Float) value).doubleValue();
        } else if (value instanceof Long) {
            return ((Long) value).doubleValue();
        } else if (value instanceof String) {
            try {
                return Double.parseDouble((String) value);
            } catch (NumberFormatException e) {
                return defaultValue;
            }
        } else {
            // Try to convert to double
            try {
                return ((Number) value).doubleValue();
            } catch (Exception e) {
                return defaultValue;
            }
        }
    }
}