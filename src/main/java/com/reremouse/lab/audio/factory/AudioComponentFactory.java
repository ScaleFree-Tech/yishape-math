package com.reremouse.lab.audio.factory;

import com.reremouse.lab.audio.core.*;

import java.util.HashMap;
import java.util.Map;
import java.util.ServiceLoader;

/**
 * 音频组件工厂类 / Audio Component Factory Class
 * <p>
 * 使用工厂模式创建各种音频处理组件，包括处理器、分析器、滤波器、效果器等。
 * 支持动态注册和服务发现机制。
 * </p>
 * <p>
 * Uses factory pattern to create various audio processing components including processors, analyzers, filters, effects, etc.
 * Supports dynamic registration and service discovery mechanism.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class AudioComponentFactory {
    
    private static AudioComponentFactory instance;
    
    // 组件注册表 / Component registries
    private final Map<String, Class<? extends IAudioProcessor>> processorRegistry = new HashMap<>();
    private final Map<String, Class<? extends IAudioAnalyzer>> analyzerRegistry = new HashMap<>();
    private final Map<String, Class<? extends IAudioFilter>> filterRegistry = new HashMap<>();
    private final Map<String, Class<? extends IAudioEffect>> effectRegistry = new HashMap<>();
    private final Map<String, Class<? extends IAudioCodec>> codecRegistry = new HashMap<>();
    
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
    public IAudioProcessor createProcessor(String processorType) throws IllegalArgumentException {
        Class<? extends IAudioProcessor> processorClass = processorRegistry.get(processorType.toLowerCase());
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
    public IAudioProcessor createProcessor(String processorType, Map<String, Object> parameters) throws IllegalArgumentException {
        IAudioProcessor processor = createProcessor(processorType);
        
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
    public IAudioFilter createFilter(IAudioFilter.FilterType filterType) {
        return createFilter(filterType.name().toLowerCase());
    }
    
    /**
     * 创建音频滤波器 / Create audio filter
     *
     * @param filterType 滤波器类型名称 / Filter type name
     * @return 音频滤波器实例 / Audio filter instance
     * @throws IllegalArgumentException 当滤波器类型不存在时抛出 / Thrown when filter type doesn't exist
     */
    public IAudioFilter createFilter(String filterType) throws IllegalArgumentException {
        Class<? extends IAudioFilter> filterClass = filterRegistry.get(filterType.toLowerCase());
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
    public IAudioFilter createFilter(IAudioFilter.FilterType filterType, double cutoffFrequency, int order) {
        IAudioFilter filter = createFilter(filterType);
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
    
    // ================ 音乐组件创建方法 / Music Component Creation Methods ================
    
    /**
     * 创建音乐分析器 / Create music analyzer
     *
     * @return 音乐分析器实例 / Music analyzer instance
     */
    public com.reremouse.lab.audio.core.IMusicAnalyzer createMusicAnalyzer() {
        return (com.reremouse.lab.audio.core.IMusicAnalyzer) createAnalyzer("music");
    }
    
    /**
     * 创建音乐分析器（带参数） / Create music analyzer (with parameters)
     *
     * @param parameters 初始化参数 / Initialization parameters
     * @return 音乐分析器实例 / Music analyzer instance
     */
    public com.reremouse.lab.audio.core.IMusicAnalyzer createMusicAnalyzer(Map<String, Object> parameters) {
        return (com.reremouse.lab.audio.core.IMusicAnalyzer) createAnalyzer("music", parameters);
    }
    
    /**
     * 创建音乐理论处理器 / Create music theory processor
     *
     * @return 音乐理论处理器实例 / Music theory processor instance
     */
    public com.reremouse.lab.audio.core.IMusicProcessor createMusicProcessor() {
        return (com.reremouse.lab.audio.core.IMusicProcessor) createProcessor("music_theory");
    }
    
    /**
     * 创建音乐理论处理器（带参数） / Create music theory processor (with parameters)
     *
     * @param parameters 初始化参数 / Initialization parameters
     * @return 音乐理论处理器实例 / Music theory processor instance
     */
    public com.reremouse.lab.audio.core.IMusicProcessor createMusicProcessor(Map<String, Object> parameters) {
        return (com.reremouse.lab.audio.core.IMusicProcessor) createProcessor("music_theory", parameters);
    }
    
    /**
     * 创建音乐处理流水线 / Create music processing pipeline
     * <p>
     * 创建包含音乐分析器和处理器的完整流水线。
     * Create complete pipeline including music analyzer and processor.
     * </p>
     *
     * @return 音乐处理流水线 / Music processing pipeline
     */
    public MusicProcessingPipeline createMusicPipeline() {
        com.reremouse.lab.audio.core.IMusicAnalyzer analyzer = createMusicAnalyzer();
        com.reremouse.lab.audio.core.IMusicProcessor processor = createMusicProcessor();
        return new MusicProcessingPipeline(analyzer, processor);
    }
    
    /**
     * 音乐处理流水线类 / Music Processing Pipeline Class
     */
    public static class MusicProcessingPipeline {
        private final com.reremouse.lab.audio.core.IMusicAnalyzer analyzer;
        private final com.reremouse.lab.audio.core.IMusicProcessor processor;
        
        public MusicProcessingPipeline(com.reremouse.lab.audio.core.IMusicAnalyzer analyzer, 
                                     com.reremouse.lab.audio.core.IMusicProcessor processor) {
            this.analyzer = analyzer;
            this.processor = processor;
        }
        
        public com.reremouse.lab.audio.core.IMusicAnalyzer getAnalyzer() { return analyzer; }
        public com.reremouse.lab.audio.core.IMusicProcessor getProcessor() { return processor; }
        
        /**
         * 分析音频特征 / Analyze audio features
         */
        public com.reremouse.lab.audio.MusicAnalyzer.MusicFeatures analyzeFeatures(com.reremouse.lab.audio.AudioData audioData) 
                throws com.reremouse.lab.audio.exception.AudioProcessingException {
            return analyzer.extractMusicFeatures(audioData);
        }
        
        /**
         * 生成音阶 / Generate scale
         */
        public com.reremouse.lab.audio.AudioData generateScale(int rootNote, 
                com.reremouse.lab.audio.MusicTheory.ScaleType scaleType, int octave, double duration) 
                throws com.reremouse.lab.audio.exception.AudioProcessingException {
            return processor.generateScale(rootNote, scaleType, octave, duration);
        }
        
        /**
         * 生成和弦 / Generate chord
         */
        public com.reremouse.lab.audio.AudioData generateChord(int rootNote, 
                com.reremouse.lab.audio.MusicTheory.ChordType chordType, int octave, double duration) 
                throws com.reremouse.lab.audio.exception.AudioProcessingException {
            return processor.generateChord(rootNote, chordType, octave, duration);
        }
    }
    
    // ================ 组件注册方法 / Component Registration Methods ================
    
    /**
     * 注册音频处理器 / Register audio processor
     *
     * @param name 处理器名称 / Processor name
     * @param processorClass 处理器类 / Processor class
     */
    public void registerProcessor(String name, Class<? extends IAudioProcessor> processorClass) {
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
    public void registerFilter(String name, Class<? extends IAudioFilter> filterClass) {
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
    
    // ================ 查询方法 / Query Methods ================
    
    /**
     * 获取已注册的处理器类型 / Get registered processor types
     *
     * @return 处理器类型数组 / Processor type array
     */
    public String[] getRegisteredProcessorTypes() {
        return processorRegistry.keySet().toArray(new String[0]);
    }
    
    /**
     * 获取已注册的分析器类型 / Get registered analyzer types
     *
     * @return 分析器类型数组 / Analyzer type array
     */
    public String[] getRegisteredAnalyzerTypes() {
        return analyzerRegistry.keySet().toArray(new String[0]);
    }
    
    /**
     * 获取已注册的滤波器类型 / Get registered filter types
     *
     * @return 滤波器类型数组 / Filter type array
     */
    public String[] getRegisteredFilterTypes() {
        return filterRegistry.keySet().toArray(new String[0]);
    }
    
    /**
     * 获取已注册的效果器类型 / Get registered effect types
     *
     * @return 效果器类型数组 / Effect type array
     */
    public String[] getRegisteredEffectTypes() {
        return effectRegistry.keySet().toArray(new String[0]);
    }
    
    /**
     * 获取已注册的编解码器类型 / Get registered codec types
     *
     * @return 编解码器类型数组 / Codec type array
     */
    public String[] getRegisteredCodecTypes() {
        return codecRegistry.keySet().toArray(new String[0]);
    }
    
    // ================ 私有辅助方法 / Private Helper Methods ================
    
    /**
     * 初始化默认组件 / Initialize default components
     */
    private void initializeDefaultComponents() {
        // 注册默认处理器 / Register default processors
        // registerProcessor("volume", VolumeProcessor.class);
        // registerProcessor("normalize", NormalizeProcessor.class);
        // registerProcessor("resample", ResampleProcessor.class);
        
        // 注册音乐处理器 / Register music processors
        registerProcessor("music_theory", com.reremouse.lab.audio.music.MusicTheoryProcessor.class);
        
        // 注册默认分析器 / Register default analyzers
        // registerAnalyzer("spectral", SpectralAnalyzer.class);
        // registerAnalyzer("mfcc", MFCCAnalyzer.class);
        // registerAnalyzer("chroma", ChromaAnalyzer.class);
        
        // 注册音乐分析器 / Register music analyzers
        registerAnalyzer("music", com.reremouse.lab.audio.music.IntegratedMusicAnalyzer.class);
        registerAnalyzer("music_integrated", com.reremouse.lab.audio.music.IntegratedMusicAnalyzer.class);
        
        // 注册默认滤波器 / Register default filters
        // registerFilter("low_pass", LowPassFilter.class);
        // registerFilter("high_pass", HighPassFilter.class);
        // registerFilter("band_pass", BandPassFilter.class);
        
        // 注册默认效果器 / Register default effects
        // registerEffect("reverb", ReverbEffect.class);
        // registerEffect("delay", DelayEffect.class);
        // registerEffect("chorus", ChorusEffect.class);
        
        // 注册默认编解码器 / Register default codecs
        // registerCodec("wav", WavCodec.class);
        // registerCodec("pcm", PCMCodec.class);
    }
    
    /**
     * 从类路径加载服务 / Load services from classpath
     */
    private void loadServicesFromClasspath() {
        // 使用ServiceLoader机制自动发现和注册组件 / Use ServiceLoader mechanism for automatic component discovery and registration
        ServiceLoader<IAudioProcessor> processorLoader = ServiceLoader.load(IAudioProcessor.class);
        for (IAudioProcessor processor : processorLoader) {
            processorRegistry.put(processor.getName().toLowerCase(), processor.getClass());
        }
        
        ServiceLoader<IAudioAnalyzer> analyzerLoader = ServiceLoader.load(IAudioAnalyzer.class);
        for (IAudioAnalyzer analyzer : analyzerLoader) {
            analyzerRegistry.put(analyzer.getName().toLowerCase(), analyzer.getClass());
        }
        
        ServiceLoader<IAudioFilter> filterLoader = ServiceLoader.load(IAudioFilter.class);
        for (IAudioFilter filter : filterLoader) {
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
    }
}