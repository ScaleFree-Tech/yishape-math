package com.yishape.lab.audio.factory;

import com.yishape.lab.audio.analysis.PitchDetector;
import com.yishape.lab.audio.analysis.SpectrumAnalyzer;
import com.yishape.lab.audio.analysis.STFTAnalyzer;
import com.yishape.lab.audio.core.IAudioCodec;
import com.yishape.lab.audio.effect.ReverbEffect;
import com.yishape.lab.audio.filter.IBaseAudioFilter;
import com.yishape.lab.audio.filter.LowPassFilter;
import com.yishape.lab.audio.filter.AdvancedLowPassFilter;
import com.yishape.lab.audio.processing.ChannelProcessor;
import com.yishape.lab.audio.processing.IAdvancedAudioProcessor;
import com.yishape.lab.audio.processing.NormalizeProcessor;
import com.yishape.lab.audio.processing.VolumeProcessor;
import com.yishape.lab.audio.enhancement.IAudioEnhancer;
import com.yishape.lab.audio.enhancement.NoiseReductionEnhancer;
import com.yishape.lab.audio.enhancement.EqualizerEnhancer;
import com.yishape.lab.audio.enhancement.CompressorEnhancer;
import com.yishape.lab.audio.enhancement.ReverbEnhancer;

import java.util.HashMap;
import java.util.Map;
import java.util.ServiceLoader;
import com.yishape.lab.audio.effect.IAudioEffect;
import com.yishape.lab.audio.analysis.IAudioAnalyzer;
import com.yishape.lab.audio.feature.IAudioFeatureExtractor;
import com.yishape.lab.audio.feature.AudioFeatureExtractorImpl;

/**
 * 音频组件工厂类 / Audio Component Factory Class
 * <p>
 * 使用工厂模式创建各种音频处理组件，包括处理器、分析器、滤波器、效果器、特征提取器等。
 * 支持动态注册和服务发现机制。
 * </p>
 * <p>
 * Uses factory pattern to create various audio processing components including processors, analyzers, filters, effects, feature extractors, etc.
 * Supports dynamic registration and service discovery mechanism.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioComponentFactory {
    
    public enum ComponentType {
        PROCESSOR,
        ANALYZER,
        FILTER,
        EFFECT,
        CODEC,
        FEATURE_EXTRACTOR,
        ENHANCER
    }
    
    private static AudioComponentFactory instance;
    
    // 组件注册表 / Component registries
    private final Map<String, Class<? extends IAdvancedAudioProcessor>> processorRegistry = new HashMap<>();
    private final Map<String, Class<? extends IAudioAnalyzer>> analyzerRegistry = new HashMap<>();
    private final Map<String, Class<? extends IBaseAudioFilter>> filterRegistry = new HashMap<>();
    private final Map<String, Class<? extends IAudioEffect>> effectRegistry = new HashMap<>();
    private final Map<String, Class<? extends IAudioCodec>> codecRegistry = new HashMap<>();
    private final Map<String, Class<? extends IAudioFeatureExtractor>> featureExtractorRegistry = new HashMap<>();
    private final Map<String, Class<? extends IAudioEnhancer>> enhancerRegistry = new HashMap<>();
    
    /**
     * 私有构造函数，实现单例模式 / Private constructor for singleton pattern
     */
    private AudioComponentFactory() {
        initializeDefaultComponents();
        loadServicesFromClasspath();
    }
    
    /**
     * 获取工厂实例 / Get factory instance
     * <p>
     * 使用单例模式，确保全局只有一个工厂实例。
     * Uses singleton pattern to ensure only one factory instance globally.
     * </p>
     *
     * @return 工厂实例 / Factory instance
     */
    public static synchronized AudioComponentFactory getInstance() {
        if (instance == null) {
            instance = new AudioComponentFactory();
        }
        return instance;
    }
    
    // ================ 音频处理器创建方法 / Audio Processor Creation Methods ================
    
    /**
     * 创建音频处理器 / Create audio processor
     * <p>
     * 根据处理器类型名称创建相应的音频处理器实例。
     * Create audio processor instance based on processor type name.
     * </p>
     *
     * @param processorType 处理器类型名称 / Processor type name
     * @return 音频处理器实例 / Audio processor instance
     * @throws IllegalArgumentException 当处理器类型不存在时抛出 / Thrown when processor type doesn't exist
     */
    public IAdvancedAudioProcessor createProcessor(String processorType) throws IllegalArgumentException {
        Class<? extends IAdvancedAudioProcessor> processorClass = processorRegistry.get(processorType.toLowerCase());
        if (processorClass == null) {
            throw new IllegalArgumentException("Unknown processor type: " + processorType);
        }
        
        try {
            return processorClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Failed to create processor: " + processorType, e);
        }
    }
    
    /**
     * 创建音频处理器（带参数） / Create audio processor (with parameters)
     *
     * @param processorType 处理器类型名称 / Processor type name
     * @param parameters 初始化参数 / Initialization parameters
     * @return 音频处理器实例 / Audio processor instance
     * @throws IllegalArgumentException 当处理器类型不存在时抛出 / Thrown when processor type doesn't exist
     */
    public IAdvancedAudioProcessor createProcessor(String processorType, Map<String, Object> parameters) throws IllegalArgumentException {
        IAdvancedAudioProcessor processor = createProcessor(processorType);
        
        if (parameters != null) {
            for (Map.Entry<String, Object> entry : parameters.entrySet()) {
                processor.setParameter(entry.getKey(), entry.getValue());
            }
        }
        
        return processor;
    }
    
    // ================ 音频分析器创建方法 / Audio Analyzer Creation Methods ================
    
    /**
     * 创建音频分析器 / Create audio analyzer
     *
     * @param analyzerType 分析器类型名称 / Analyzer type name
     * @return 音频分析器实例 / Audio analyzer instance
     * @throws IllegalArgumentException 当分析器类型不存在时抛出 / Thrown when analyzer type doesn't exist
     */
    public IAudioAnalyzer createAnalyzer(String analyzerType) throws IllegalArgumentException {
        Class<? extends IAudioAnalyzer> analyzerClass = analyzerRegistry.get(analyzerType.toLowerCase());
        if (analyzerClass == null) {
            throw new IllegalArgumentException("Unknown analyzer type: " + analyzerType);
        }
        
        try {
            return analyzerClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Failed to create analyzer: " + analyzerType, e);
        }
    }
    
    /**
     * 创建音频分析器（带参数） / Create audio analyzer (with parameters)
     *
     * @param analyzerType 分析器类型名称 / Analyzer type name
     * @param parameters 初始化参数 / Initialization parameters
     * @return 音频分析器实例 / Audio analyzer instance
     */
    public IAudioAnalyzer createAnalyzer(String analyzerType, Map<String, Object> parameters) {
        IAudioAnalyzer analyzer = createAnalyzer(analyzerType);
        
        if (parameters != null) {
            for (Map.Entry<String, Object> entry : parameters.entrySet()) {
                analyzer.setParameter(entry.getKey(), entry.getValue());
            }
        }
        
        return analyzer;
    }
    
    // ================ 音频滤波器创建方法 / Audio Filter Creation Methods ================

    /**
     * 创建音频滤波器 / Create audio filter
     *
     * @param filterType 滤波器类型 / Filter type
     * @return 音频滤波器实例 / Audio filter instance
     */
    public IBaseAudioFilter createFilter(IBaseAudioFilter.FilterType filterType) {
        return createFilter(filterType.name().toLowerCase());
    }

    /**
     * 创建音频滤波器 / Create audio filter
     *
     * @param filterType 滤波器类型名称 / Filter type name
     * @return 音频滤波器实例 / Audio filter instance
     * @throws IllegalArgumentException 当滤波器类型不存在时抛出 / Thrown when filter type doesn't exist
     */
    public IBaseAudioFilter createFilter(String filterType) throws IllegalArgumentException {
        Class<? extends IBaseAudioFilter> filterClass = filterRegistry.get(filterType.toLowerCase());
        if (filterClass == null) {
            throw new IllegalArgumentException("Unknown filter type: " + filterType);
        }

        try {
            return filterClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Failed to create filter: " + filterType, e);
        }
    }

    /**
     * 创建音频滤波器（带参数） / Create audio filter (with parameters)
     *
     * @param filterType 滤波器类型 / Filter type
     * @param cutoffFrequency 截止频率 / Cutoff frequency
     * @param order 滤波器阶数 / Filter order
     * @return 音频滤波器实例 / Audio filter instance
     */
    public IBaseAudioFilter createFilter(IBaseAudioFilter.FilterType filterType, double cutoffFrequency, int order) {
        IBaseAudioFilter filter = createFilter(filterType);
        filter.setCutoffFrequency(cutoffFrequency);
        filter.setOrder(order);
        filter.setFilterType(filterType);
        return filter;
    }
    
    // ================ 音频效果器创建方法 / Audio Effect Creation Methods ================
    
    /**
     * 创建音频效果器 / Create audio effect
     *
     * @param effectType 效果器类型 / Effect type
     * @return 音频效果器实例 / Audio effect instance
     */
    public IAudioEffect createEffect(IAudioEffect.EffectType effectType) {
        return createEffect(effectType.name().toLowerCase());
    }
    
    /**
     * 创建音频效果器 / Create audio effect
     *
     * @param effectType 效果器类型名称 / Effect type name
     * @return 音频效果器实例 / Audio effect instance
     * @throws IllegalArgumentException 当效果器类型不存在时抛出 / Thrown when effect type doesn't exist
     */
    public IAudioEffect createEffect(String effectType) throws IllegalArgumentException {
        Class<? extends IAudioEffect> effectClass = effectRegistry.get(effectType.toLowerCase());
        if (effectClass == null) {
            throw new IllegalArgumentException("Unknown effect type: " + effectType);
        }
        
        try {
            return effectClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Failed to create effect: " + effectType, e);
        }
    }
    
    // ================ 音频编解码器创建方法 / Audio Codec Creation Methods ================
    
    /**
     * 创建音频编解码器 / Create audio codec
     *
     * @param codecType 编解码器类型 / Codec type
     * @return 音频编解码器实例 / Audio codec instance
     */
    public IAudioCodec createCodec(IAudioCodec.CodecType codecType) {
        return createCodec(codecType.name().toLowerCase());
    }
    
    /**
     * 创建音频编解码器 / Create audio codec
     *
     * @param codecType 编解码器类型名称 / Codec type name
     * @return 音频编解码器实例 / Audio codec instance
     * @throws IllegalArgumentException 当编解码器类型不存在时抛出 / Thrown when codec type doesn't exist
     */
    public IAudioCodec createCodec(String codecType) throws IllegalArgumentException {
        Class<? extends IAudioCodec> codecClass = codecRegistry.get(codecType.toLowerCase());
        if (codecClass == null) {
            throw new IllegalArgumentException("Unknown codec type: " + codecType);
        }
        
        try {
            return codecClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Failed to create codec: " + codecType, e);
        }
    }
    
    // ================ 音频增强器创建方法 / Audio Enhancer Creation Methods ================
    
    /**
     * 创建音频增强器 / Create audio enhancer
     *
     * @param enhancerType 增强器类型名称 / Enhancer type name
     * @return 音频增强器实例 / Audio enhancer instance
     * @throws IllegalArgumentException 当增强器类型不存在时抛出 / Thrown when enhancer type doesn't exist
     */
    public IAudioEnhancer createEnhancer(String enhancerType) throws IllegalArgumentException {
        Class<? extends IAudioEnhancer> enhancerClass = enhancerRegistry.get(enhancerType.toLowerCase());
        if (enhancerClass == null) {
            throw new IllegalArgumentException("Unknown enhancer type: " + enhancerType);
        }
        
        try {
            return enhancerClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Failed to create enhancer: " + enhancerType, e);
        }
    }
    
    /**
     * 创建音频增强器（带参数） / Create audio enhancer (with parameters)
     *
     * @param enhancerType 增强器类型名称 / Enhancer type name
     * @param parameters 初始化参数 / Initialization parameters
     * @return 音频增强器实例 / Audio enhancer instance
     */
    public IAudioEnhancer createEnhancer(String enhancerType, Map<String, Object> parameters) {
        IAudioEnhancer enhancer = createEnhancer(enhancerType);
        
        if (parameters != null) {
            // Enhancers handle parameters differently than other components
            // This is kept for API consistency
        }
        
        return enhancer;
    }
    
    // ================ 组件注册和初始化方法 / Component Registration and Initialization Methods ================
    
    /**
     * 注册音频处理器 / Register audio processor
     *
     * @param name 处理器名称 / Processor name
     * @param processorClass 处理器类 / Processor class
     */
    public void registerProcessor(String name, Class<? extends IAdvancedAudioProcessor> processorClass) {
        processorRegistry.put(name.toLowerCase(), processorClass);
    }
    
    /**
     * 注册音频分析器 / Register audio analyzer
     *
     * @param name 分析器名称 / Analyzer name
     * @param analyzerClass 分析器类 / Analyzer class
     */
    public void registerAnalyzer(String name, Class<? extends IAudioAnalyzer> analyzerClass) {
        analyzerRegistry.put(name.toLowerCase(), analyzerClass);
    }
    
    /**
     * 注册音频滤波器 / Register audio filter
     *
     * @param name 滤波器名称 / Filter name
     * @param filterClass 滤波器类 / Filter class
     */
    public void registerFilter(String name, Class<? extends IBaseAudioFilter> filterClass) {
        filterRegistry.put(name.toLowerCase(), filterClass);
    }
    
    /**
     * 注册音频效果器 / Register audio effect
     *
     * @param name 效果器名称 / Effect name
     * @param effectClass 效果器类 / Effect class
     */
    public void registerEffect(String name, Class<? extends IAudioEffect> effectClass) {
        effectRegistry.put(name.toLowerCase(), effectClass);
    }
    
    /**
     * 注册音频编解码器 / Register audio codec
     *
     * @param name 编解码器名称 / Codec name
     * @param codecClass 编解码器类 / Codec class
     */
    public void registerCodec(String name, Class<? extends IAudioCodec> codecClass) {
        codecRegistry.put(name.toLowerCase(), codecClass);
    }
    
    /**
     * 注册音频特征提取器 / Register audio feature extractor
     *
     * @param name 特征提取器名称 / Feature extractor name
     * @param featureExtractorClass 特征提取器类 / Feature extractor class
     */
    public void registerFeatureExtractor(String name, Class<? extends IAudioFeatureExtractor> featureExtractorClass) {
        featureExtractorRegistry.put(name.toLowerCase(), featureExtractorClass);
    }
    
    /**
     * 注册音频增强器 / Register audio enhancer
     *
     * @param name 增强器名称 / Enhancer name
     * @param enhancerClass 增强器类 / Enhancer class
     */
    public void registerEnhancer(String name, Class<? extends IAudioEnhancer> enhancerClass) {
        enhancerRegistry.put(name.toLowerCase(), enhancerClass);
    }
    
    // ================ 音频特征提取器创建方法 / Audio Feature Extractor Creation Methods ================
    
    /**
     * 创建音频特征提取器 / Create audio feature extractor
     *
     * @param extractorType 特征提取器类型名称 / Feature extractor type name
     * @return 音频特征提取器实例 / Audio feature extractor instance
     * @throws IllegalArgumentException 当特征提取器类型不存在时抛出 / Thrown when feature extractor type doesn't exist
     */
    public IAudioFeatureExtractor createFeatureExtractor(String extractorType) throws IllegalArgumentException {
        Class<? extends IAudioFeatureExtractor> extractorClass = featureExtractorRegistry.get(extractorType.toLowerCase());
        if (extractorClass == null) {
            throw new IllegalArgumentException("Unknown feature extractor type: " + extractorType);
        }
        
        try {
            return extractorClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Failed to create feature extractor: " + extractorType, e);
        }
    }
    
    /**
     * 创建音频特征提取器（带参数） / Create audio feature extractor (with parameters)
     *
     * @param extractorType 特征提取器类型名称 / Feature extractor type name
     * @param parameters 初始化参数 / Initialization parameters
     * @return 音频特征提取器实例 / Audio feature extractor instance
     */
    public IAudioFeatureExtractor createFeatureExtractor(String extractorType, Map<String, Object> parameters) {
        IAudioFeatureExtractor extractor = createFeatureExtractor(extractorType);
        
        // Note: Feature extractors don't currently support parameter setting like other components
        // This is kept for API consistency
        return extractor;
    }
    
    /**
     * 初始化默认组件 / Initialize default components
     */
    private void initializeDefaultComponents() {
        // 注册默认的音频处理器 / Register default audio processors
        registerProcessor("channel", (Class<? extends IAdvancedAudioProcessor>) (Class<?>) ChannelProcessor.class);
        registerProcessor("normalize", (Class<? extends IAdvancedAudioProcessor>) (Class<?>) NormalizeProcessor.class);
        registerProcessor("volume", (Class<? extends IAdvancedAudioProcessor>) (Class<?>) VolumeProcessor.class);

        // 注册默认的音频分析器 / Register default audio analyzers
        registerAnalyzer("spectrum", (Class<? extends IAudioAnalyzer>) (Class<?>) SpectrumAnalyzer.class);
        registerAnalyzer("pitch", (Class<? extends IAudioAnalyzer>) (Class<?>) PitchDetector.class);
        registerAnalyzer("stft", (Class<? extends IAudioAnalyzer>) (Class<?>) STFTAnalyzer.class);

        // 注册默认的音频滤波器 / Register default audio filters
        registerFilter("lowpass", (Class<? extends IBaseAudioFilter>) (Class<?>) LowPassFilter.class);
        registerFilter("advanced_lowpass", (Class<? extends IBaseAudioFilter>) (Class<?>) AdvancedLowPassFilter.class);

        // 注册默认的音频效果器 / Register default audio effects
        registerEffect("reverb", (Class<? extends IAudioEffect>) (Class<?>) ReverbEffect.class);
        
        // 注册默认的音频特征提取器 / Register default audio feature extractors
        registerFeatureExtractor("default", (Class<? extends IAudioFeatureExtractor>) (Class<?>) AudioFeatureExtractorImpl.class);
        registerFeatureExtractor("standard", (Class<? extends IAudioFeatureExtractor>) (Class<?>) AudioFeatureExtractorImpl.class);
        
        // 注册默认的音频增强器 / Register default audio enhancers
        registerEnhancer("noise_reduction", (Class<? extends IAudioEnhancer>) (Class<?>) NoiseReductionEnhancer.class);
        registerEnhancer("equalizer", (Class<? extends IAudioEnhancer>) (Class<?>) EqualizerEnhancer.class);
        registerEnhancer("compressor", (Class<? extends IAudioEnhancer>) (Class<?>) CompressorEnhancer.class);
        registerEnhancer("reverb_enhancer", (Class<? extends IAudioEnhancer>) (Class<?>) ReverbEnhancer.class);
    }
    
    /**
     * 从类路径加载服务 / Load services from classpath
     * <p>
     * 使用ServiceLoader机制自动发现和注册组件。
     * Use ServiceLoader mechanism for automatic component discovery and registration.
     * </p>
     */
    @SuppressWarnings("unused")
    private void loadServicesFromClasspath() {
        // 使用ServiceLoader机制自动发现和注册组件 / Use ServiceLoader mechanism for automatic component discovery and registration
        ServiceLoader<IAdvancedAudioProcessor> processorLoader = ServiceLoader.load(IAdvancedAudioProcessor.class);
        for (IAdvancedAudioProcessor processor : processorLoader) {
            processorRegistry.put(processor.getName().toLowerCase(), (Class<? extends IAdvancedAudioProcessor>) processor.getClass());
        }

        ServiceLoader<IAudioAnalyzer> analyzerLoader = ServiceLoader.load(IAudioAnalyzer.class);
        for (IAudioAnalyzer analyzer : analyzerLoader) {
            analyzerRegistry.put(analyzer.getName().toLowerCase(), (Class<? extends IAudioAnalyzer>) analyzer.getClass());
        }

        ServiceLoader<IBaseAudioFilter> filterLoader = ServiceLoader.load(IBaseAudioFilter.class);
        for (IBaseAudioFilter filter : filterLoader) {
            filterRegistry.put(filter.getName().toLowerCase(), filter.getClass());
        }

        ServiceLoader<IAudioEffect> effectLoader = ServiceLoader.load(IAudioEffect.class);
        for (IAudioEffect effect : effectLoader) {
            effectRegistry.put(effect.getName().toLowerCase(), effect.getClass());
        }

        ServiceLoader<IAudioCodec> codecLoader = ServiceLoader.load(IAudioCodec.class);
        for (IAudioCodec codec : codecLoader) {
            codecRegistry.put(codec.getCodecInfo().toLowerCase(), codec.getClass());
        }
        
        ServiceLoader<IAudioFeatureExtractor> featureExtractorLoader = ServiceLoader.load(IAudioFeatureExtractor.class);
        for (IAudioFeatureExtractor featureExtractor : featureExtractorLoader) {
            featureExtractorRegistry.put(featureExtractor.getExtractorName().toLowerCase(), (Class<? extends IAudioFeatureExtractor>) featureExtractor.getClass());
        }
        
        ServiceLoader<IAudioEnhancer> enhancerLoader = ServiceLoader.load(IAudioEnhancer.class);
        for (IAudioEnhancer enhancer : enhancerLoader) {
            enhancerRegistry.put(enhancer.getName().toLowerCase(), (Class<? extends IAudioEnhancer>) enhancer.getClass());
        }
    }
}