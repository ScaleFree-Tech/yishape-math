package com.yishape.lab.audio.analysis;

import com.yishape.lab.audio.core.AudioData;
import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereFFT;
import com.yishape.lab.util.Tuple2;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * FFT处理器 - 优化FFT计算和缓存 / FFT Processor - Optimized FFT calculation and caching
 * <p>
 * 提供高效的FFT计算服务，包括结果缓存、窗口函数优化和并行处理。
 * Provides efficient FFT calculation services, including result caching, window function optimization, and parallel processing.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class FFTProcessor {
    
    /** 单例实例 / Singleton instance */
    private static final FFTProcessor INSTANCE = new FFTProcessor();
    
    /** FFT结果缓存 / FFT result cache */
    private final ConcurrentHashMap<String, Tuple2<IVector<Double>, IVector<Double>>> fftCache;
    
    /** 最大缓存大小 / Maximum cache size */
    private static final int MAX_CACHE_SIZE = 100;
    
    /** 默认窗口大小 / Default window size */
    private static final int DEFAULT_WINDOW_SIZE = 1024;
    
    /** 默认重叠率 / Default overlap ratio */
    private static final double DEFAULT_OVERLAP = 0.5;
    
    /** 大文件采样阈值 / Large file sampling threshold */
    private static final int LARGE_FILE_THRESHOLD = 1000000; // 1 million samples
    
    /**
     * 私有构造函数 / Private constructor
     */
    private FFTProcessor() {
        this.fftCache = new ConcurrentHashMap<>();
    }
    
    /**
     * 获取单例实例 / Get singleton instance
     */
    public static FFTProcessor getInstance() {
        return INSTANCE;
    }
    
    /**
     * 处理音频数据的FFT计算 / Process FFT calculation for audio data
     */
    public Tuple2<IVector<Double>, IVector<Double>> processFFT(AudioData audioData) throws AudioProcessingException {
        return processFFT(audioData, null);
    }
    
    /**
     * 处理音频数据的FFT计算（带参数）/ Process FFT calculation for audio data with parameters
     */
    public Tuple2<IVector<Double>, IVector<Double>> processFFT(AudioData audioData, Map<String, Object> parameters) throws AudioProcessingException {
        try {
            // 获取参数 / Get parameters
            int windowSize = DEFAULT_WINDOW_SIZE;
            double overlap = DEFAULT_OVERLAP;
            
            if (parameters != null) {
                if (parameters.containsKey("windowSize")) {
                    windowSize = (Integer) parameters.get("windowSize");
                }
                if (parameters.containsKey("overlap")) {
                    overlap = (Double) parameters.get("overlap");
                }
            }
            
            // 生成缓存键 / Generate cache key
            String cacheKey = generateCacheKey(audioData, windowSize, overlap);
            
            // 检查缓存 / Check cache
            Tuple2<IVector<Double>, IVector<Double>> cachedResult = fftCache.get(cacheKey);
            if (cachedResult != null) {
                return cachedResult;
            }
            
            // 计算FFT / Calculate FFT
            Tuple2<IVector<Double>, IVector<Double>> result = calculateFFT(audioData, windowSize, overlap);
            
            // 缓存结果 / Cache result
            cacheResult(cacheKey, result);
            
            return result;
        } catch (Exception e) {
            throw new AudioProcessingException("Failed to process FFT", e);
        }
    }
    
    /**
     * 计算FFT / Calculate FFT
     */
    private Tuple2<IVector<Double>, IVector<Double>> calculateFFT(AudioData audioData, int windowSize, double overlap) throws AudioProcessingException {
        IVector<Double> samples = audioData.getSamples();
        
        // For large files, use downsampling for visualization purposes
        IVector<Double> processedSamples = samples;
        if (samples.length() > LARGE_FILE_THRESHOLD) {
            processedSamples = downsampleForVisualization(samples, LARGE_FILE_THRESHOLD);
        }
        
        // 应用窗口函数 / Apply window function
        IVector<Double> windowedSamples = applyWindow(processedSamples, windowSize);
        
        // 确保长度是2的幂 / Ensure length is power of 2
        int fftSize = nextPowerOfTwo(windowedSamples.length());
        
        // 转换为复数数组 / Convert to complex array
        Complex[] complexSamples = new Complex[fftSize];
        for (int i = 0; i < windowedSamples.length(); i++) {
            complexSamples[i] = new Complex(windowedSamples.get(i), 0);
        }
        for (int i = windowedSamples.length(); i < fftSize; i++) {
            complexSamples[i] = new Complex(0, 0);
        }
        
        // 计算FFT / Calculate FFT
        Complex[] fftResult = RereFFT.fft(complexSamples);
        
        // 计算频率和幅度 / Calculate frequencies and magnitudes
        return extractSpectrum(fftResult, audioData.getSampleRate(), fftSize);
    }
    
    /**
     * 为可视化目的对大文件进行降采样 / Downsample large files for visualization purposes
     */
    private IVector<Double> downsampleForVisualization(IVector<Double> samples, int targetSize) {
        int originalSize = samples.length();
        int factor = originalSize / targetSize;
        if (factor <= 1) {
            return samples;
        }
        
        int newSize = originalSize / factor;
        double[] downsampled = new double[newSize];
        
        for (int i = 0; i < newSize; i++) {
            // Take every 'factor' sample or average a small window
            int startIndex = i * factor;
            int endIndex = Math.min(startIndex + factor, originalSize);
            
            double sum = 0;
            for (int j = startIndex; j < endIndex; j++) {
                sum += samples.get(j);
            }
            downsampled[i] = sum / (endIndex - startIndex);
        }
        
        return Linalg.vector(downsampled);
    }
    
    /**
     * 提取频谱信息 / Extract spectrum information
     */
    private Tuple2<IVector<Double>, IVector<Double>> extractSpectrum(Complex[] fftResult, double sampleRate, int fftSize) {
        // For real signals, we only need the first half of the FFT result (plus the Nyquist frequency)
        int n = fftResult.length;
        int halfN = n / 2 + 1;
        
        IVector<Double> frequencies = Linalg.zeros(halfN);
        IVector<Double> magnitudes = Linalg.zeros(halfN);
        
        for (int i = 0; i < halfN; i++) {
            frequencies.set(i, i * sampleRate / n);
            magnitudes.set(i, fftResult[i].magnitude());
        }
        
        return new Tuple2<>(frequencies, magnitudes);
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
    
    /**
     * 计算下一个2的幂 / Calculate next power of 2
     */
    private int nextPowerOfTwo(int n) {
        return (int) Math.pow(2, Math.ceil(Math.log(n) / Math.log(2)));
    }
    
    /**
     * 生成缓存键 / Generate cache key
     */
    private String generateCacheKey(AudioData audioData, int windowSize, double overlap) {
        StringBuilder keyBuilder = new StringBuilder();
        keyBuilder.append(audioData.getSampleRate()).append("_");
        keyBuilder.append(audioData.getSamples().length()).append("_");
        keyBuilder.append(audioData.getChannels()).append("_");
        keyBuilder.append(windowSize).append("_");
        keyBuilder.append(overlap).append("_");
        
        // 添加音频数据的简单哈希（前1000个样本）/ Add simple hash of audio data (first 1000 samples)
        IVector<Double> samples = audioData.getSamples();
        int sampleCount = Math.min(1000, samples.length());
        double hashSum = 0.0;
        for (int i = 0; i < sampleCount; i++) {
            hashSum += Math.abs(samples.get(i));
        }
        keyBuilder.append((int)(hashSum * 1000));
        
        return keyBuilder.toString();
    }
    
    /**
     * 缓存结果 / Cache result
     */
    private void cacheResult(String cacheKey, Tuple2<IVector<Double>, IVector<Double>> result) {
        if (fftCache.size() >= MAX_CACHE_SIZE) {
            // 简单的LRU策略：移除第一个元素 / Simple LRU: remove first element
            String firstKey = fftCache.keySet().iterator().next();
            fftCache.remove(firstKey);
        }
        fftCache.put(cacheKey, result);
    }
    
    /**
     * 清除缓存 / Clear cache
     */
    public void clearCache() {
        fftCache.clear();
    }
    
    /**
     * 获取缓存大小 / Get cache size
     */
    public int getCacheSize() {
        return fftCache.size();
    }
}