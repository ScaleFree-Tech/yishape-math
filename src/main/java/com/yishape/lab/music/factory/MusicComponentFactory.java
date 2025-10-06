package com.yishape.lab.music.factory;

import com.yishape.lab.audio.exception.AudioProcessingException;
import com.yishape.lab.music.analysis.IMusicAnalyzer;
import com.yishape.lab.music.analysis.BasicMusicAnalyzer;
import com.yishape.lab.music.analysis.ComprehensiveMusicAnalyzer;
import com.yishape.lab.music.analysis.basic.BeatAnalyzerImpl;
import com.yishape.lab.music.analysis.basic.KeyAnalyzerImpl;
import com.yishape.lab.music.analysis.basic.ChordAnalyzerImpl;
import com.yishape.lab.music.analysis.feature.IFeatureExtractor;
import com.yishape.lab.music.analysis.feature.FeatureExtractorImpl;
import com.yishape.lab.music.analysis.advanced.IAdvancedAnalyzer;
import com.yishape.lab.music.analysis.advanced.EmotionAnalyzer;
import com.yishape.lab.music.analysis.advanced.GenreAnalyzer;
import com.yishape.lab.music.analysis.advanced.ComplexityAnalyzer;
import com.yishape.lab.music.analysis.AdvancedMusicAnalyzer;
import com.yishape.lab.music.processing.IMusicProcessor;
import com.yishape.lab.music.processing.Harmonizer;
import com.yishape.lab.music.processing.MusicTheoryProcessor;
import com.yishape.lab.music.processing.Quantizer;
import com.yishape.lab.music.processing.Transposer;
import com.yishape.lab.music.generation.IMusicGenerator;
import com.yishape.lab.music.generation.ChordGenerator;
import com.yishape.lab.music.generation.IntervalGenerator;
import com.yishape.lab.music.generation.ScaleGenerator;
import com.yishape.lab.music.filter.IMusicFilter;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.lang.reflect.Constructor;
import java.util.ServiceLoader;

/**
 * 音乐组件工厂 / Music Component Factory
 * <p>
 * 工厂类用于创建各种音乐组件实例，包括分析器、处理器、生成器、滤波器等。
 * Factory class for creating various music component instances including analyzers, processors, generators, filters, etc.
 * </p>
 *
 * @author lteb2
 * @version 1.1
 * @since 1.0
 */
public class MusicComponentFactory {
    
    public enum ComponentType {
        ANALYZER,
        PROCESSOR,
        GENERATOR,
        FILTER
    }
    
    // 单例实例 / Singleton instance
    private static volatile MusicComponentFactory instance;
    
    // 组件注册表 / Component registries
    private final Map<String, AnalyzerInfo<?>> registeredAnalyzers;
    private final Map<String, Class<? extends IMusicProcessor>> processorRegistry = new HashMap<>();
    private final Map<String, Class<?>> generatorRegistry = new HashMap<>();
    private final Map<String, Class<? extends IMusicFilter>> filterRegistry = new HashMap<>();
    
    /**
     * 私有构造函数 / Private constructor
     */
    private MusicComponentFactory() {
        this.registeredAnalyzers = new HashMap<>();
        registerDefaultComponents();
        loadServicesFromClasspath();
    }
    
    /**
     * 获取工厂单例实例 / Get factory singleton instance
     *
     * @return 工厂实例 / Factory instance
     */
    public static MusicComponentFactory getInstance() {
        if (instance == null) {
            synchronized (MusicComponentFactory.class) {
                if (instance == null) {
                    instance = new MusicComponentFactory();
                }
            }
        }
        return instance;
    }
    
    // ================ 音乐分析器创建方法 / Music Analyzer Creation Methods ================
    
    /**
     * 创建音乐分析器 / Create music analyzer
     * <p>
     * 创建指定类型的音乐分析器实例
     * Create music analyzer instance of specified type.
     * </p>
     *
     * @param analyzerType 分析器类型 / Analyzer type
     * @return 音乐分析器实例 / Music analyzer instance
     * @throws AudioProcessingException 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     */
    public IMusicAnalyzer createMusicAnalyzer(String analyzerType) throws AudioProcessingException {
        switch (analyzerType.toLowerCase()) {
            case "basic":
                return createAnalyzer("basic_music_analyzer");
            case "comprehensive":
                return createAnalyzer("comprehensive_music_analyzer");
            default:
                throw new AudioProcessingException("未知的分析器类型：" + analyzerType);
        }
    }
    
    /**
     * 创建基础分析器 / Create basic analyzer
     * <p>
     * 创建基础音乐分析器实例
     * Create basic music analyzer instance.
     * </p>
     *
     * @param analyzerType 分析器类型("beat", "key", "chord") / Analyzer type ("beat", "key", "chord")
     * @return 基础分析器实例 / Basic analyzer instance
     * @throws AudioProcessingException 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     */
    public Object createBasicAnalyzer(String analyzerType) throws AudioProcessingException {
        switch (analyzerType.toLowerCase()) {
            case "beat":
                return createAnalyzer("beat_analyzer");
            case "key":
                return createAnalyzer("key_analyzer");
            case "chord":
                return createAnalyzer("chord_analyzer");
            default:
                throw new AudioProcessingException("未知的基础分析器类型：" + analyzerType);
        }
    }
    
    /**
     * 创建高级分析器 / Create advanced analyzer
     * <p>
     * 创建高级音乐分析器实例
     * Create advanced music analyzer instance.
     * </p>
     *
     * @param analyzerType 分析器类型("emotion", "genre", "complexity") / Analyzer type ("emotion", "genre", "complexity")
     * @return 高级分析器实例 / Advanced analyzer instance
     * @throws AudioProcessingException 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     */
    public IAdvancedAnalyzer createAdvancedAnalyzer(String analyzerType) throws AudioProcessingException {
        switch (analyzerType.toLowerCase()) {
            case "emotion":
                return createAnalyzer("emotion_analyzer");
            case "genre":
                return createAnalyzer("genre_analyzer");
            case "complexity":
                return createAnalyzer("complexity_analyzer");
            default:
                throw new AudioProcessingException("未知的高级分析器类型：" + analyzerType);
        }
    }
    
    /**
     * 创建特征提取器 / Create feature extractor
     * <p>
     * 创建音乐特征提取器实例
     * Create music feature extractor instance.
     * </p>
     *
     * @return 特征提取器实例 / Feature extractor instance
     * @throws AudioProcessingException 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     */
    public IFeatureExtractor createFeatureExtractor() throws AudioProcessingException {
        return createAnalyzer("feature_extractor");
    }
    
    /**
     * 创建高级音乐分析器 / Create advanced music analyzer
     * <p>
     * 创建高级音乐分析器实例，用于统一访问所有高级分析功能
     * Create advanced music analyzer instance for unified access to all advanced analysis features.
     * </p>
     *
     * @return 高级音乐分析器实例 / Advanced music analyzer instance
     * @throws AudioProcessingException 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     */
    public IAdvancedAnalyzer createAdvancedMusicAnalyzer() throws AudioProcessingException {
        return createAnalyzer("advanced_music_analyzer");
    }
    
    // ================ 音乐处理器创建方法 / Music Processor Creation Methods ================
    
    /**
     * 创建音乐处理器 / Create music processor
     * <p>
     * 根据处理器类型名称创建相应的音乐处理器实例。
     * Create music processor instance based on processor type name.
     * </p>
     *
     * @param processorType 处理器类型名称 / Processor type name
     * @return 音乐处理器实例 / Music processor instance
     * @throws AudioProcessingException 当处理器类型不存在时抛出 / Thrown when processor type doesn't exist
     */
    public IMusicProcessor createProcessor(String processorType) throws AudioProcessingException {
        Class<? extends IMusicProcessor> processorClass = processorRegistry.get(processorType.toLowerCase());
        if (processorClass == null) {
            throw new AudioProcessingException("未知的处理器类型：" + processorType);
        }
        
        try {
            return processorClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new AudioProcessingException("创建处理器失败：" + processorType, e);
        }
    }
    
    // ================ 音乐生成器创建方法 / Music Generator Creation Methods ================
    
    /**
     * 创建音乐生成器 / Create music generator
     * <p>
     * 根据生成器类型名称创建相应的音乐生成器实例。
     * Create music generator instance based on generator type name.
     * </p>
     *
     * @param generatorType 生成器类型名称 / Generator type name
     * @return 音乐生成器实例 / Music generator instance
     * @throws AudioProcessingException 当生成器类型不存在时抛出 / Thrown when generator type doesn't exist
     */
    public Object createGenerator(String generatorType) throws AudioProcessingException {
        Class<?> generatorClass = generatorRegistry.get(generatorType.toLowerCase());
        if (generatorClass == null) {
            throw new AudioProcessingException("未知的生成器类型：" + generatorType);
        }
        
        try {
            return generatorClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new AudioProcessingException("创建生成器失败：" + generatorType, e);
        }
    }
    
    // ================ 音乐滤波器创建方法 / Music Filter Creation Methods ================
    
    /**
     * 创建音乐滤波器 / Create music filter
     * <p>
     * 根据滤波器类型名称创建相应的音乐滤波器实例。
     * Create music filter instance based on filter type name.
     * </p>
     *
     * @param filterType 滤波器类型名称 / Filter type name
     * @return 音乐滤波器实例 / Music filter instance
     * @throws AudioProcessingException 当滤波器类型不存在时抛出 / Thrown when filter type doesn't exist
     */
    public IMusicFilter createFilter(String filterType) throws AudioProcessingException {
        Class<? extends IMusicFilter> filterClass = filterRegistry.get(filterType.toLowerCase());
        if (filterClass == null) {
            throw new AudioProcessingException("未知的滤波器类型：" + filterType);
        }
        
        try {
            return filterClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new AudioProcessingException("创建滤波器失败：" + filterType, e);
        }
    }
    
    // ================ 组件注册方法 / Component Registration Methods ================
    
    /**
     * 注册音乐处理器 / Register music processor
     *
     * @param name 处理器名称 / Processor name
     * @param processorClass 处理器类 / Processor class
     */
    public void registerProcessor(String name, Class<? extends IMusicProcessor> processorClass) {
        processorRegistry.put(name.toLowerCase(), processorClass);
    }
    
    /**
     * 注册音乐生成器 / Register music generator
     *
     * @param name 生成器名称 / Generator name
     * @param generatorClass 生成器类 / Generator class
     */
    public void registerGenerator(String name, Class<?> generatorClass) {
        // Note: The generator classes don't implement IMusicGenerator interface directly
        // We'll store them as Class<?> and handle instantiation differently
        generatorRegistry.put(name.toLowerCase(), (Class<? extends IMusicGenerator>) generatorClass);
    }

    /**
     * 注册音乐滤波器 / Register music filter
     *
     * @param name 滤波器名称 / Filter name
     * @param filterClass 滤波器类 / Filter class
     */
    public void registerFilter(String name, Class<? extends IMusicFilter> filterClass) {
        filterRegistry.put(name.toLowerCase(), filterClass);
    }
    
    /**
     * 获取已注册的分析器列表 / Get list of registered analyzers
     */
    public Set<String> getRegisteredAnalyzerNames() {
        return registeredAnalyzers.keySet();
    }
    
    /**
     * 获取指定分类的分析器列表 / Get list of analyzers in specified category
     */
    public Map<String, AnalyzerInfo<?>> getAnalyzersByCategory(AnalyzerCategory category) {
        Map<String, AnalyzerInfo<?>> result = new HashMap<>();
        for (Map.Entry<String, AnalyzerInfo<?>> entry : registeredAnalyzers.entrySet()) {
            if (entry.getValue().getCategory() == category) {
                result.put(entry.getKey(), entry.getValue());
            }
        }
        return result;
    }
    
    /**
     * 检查分析器是否已注册 / Check if analyzer is registered
     */
    public boolean isAnalyzerRegistered(String name) {
        return registeredAnalyzers.containsKey(name.toLowerCase());
    }
    
    /**
     * 获取分析器信息 / Get analyzer information
     */
    public AnalyzerInfo<?> getAnalyzerInfo(String name) {
        return registeredAnalyzers.get(name.toLowerCase());
    }
    
    /**
     * 注销分析器 / Unregister analyzer
     */
    public void unregisterAnalyzer(String name) {
        registeredAnalyzers.remove(name.toLowerCase());
    }
    
    /**
     * 注册分析器 / Register analyzer
     */
    private <T> void registerAnalyzer(String name, Class<T> analyzerClass,
                                     AnalyzerCategory category, String description, String version) {
        registeredAnalyzers.put(name.toLowerCase(), new AnalyzerInfo<>(analyzerClass, category, description, version));
    }
    
    /**
     * 创建分析器实例 / Create analyzer instance
     */
    @SuppressWarnings("unchecked")
    private <T> T createAnalyzer(String name) throws AudioProcessingException {
        AnalyzerInfo<T> info = (AnalyzerInfo<T>) registeredAnalyzers.get(name.toLowerCase());
        if (info == null) {
            throw new AudioProcessingException("未注册的分析器：" + name);
        }
        
        try {
            Class<T> clazz = info.getAnalyzerClass();
            Constructor<T> constructor = clazz.getDeclaredConstructor();
            return constructor.newInstance();
        } catch (Exception e) {
            throw new AudioProcessingException("创建分析器实例失败：" + name, e);
        }
    }
    
    /**
     * 注册默认组件 / Register default components
     * <p>
     * 注册内置的默认音乐组件
     * Register built-in default music components.
     * </p>
     */
    private void registerDefaultComponents() {
        // 注册主分析器 / Register main analyzers
        registerAnalyzer("basic_music_analyzer", BasicMusicAnalyzer.class,
                         AnalyzerCategory.COMPREHENSIVE, "基础音乐分析器", "1.0.0");
        registerAnalyzer("comprehensive_music_analyzer", ComprehensiveMusicAnalyzer.class,
                         AnalyzerCategory.COMPREHENSIVE, "综合音乐分析器", "1.0.0");
        
        // 注册基础分析器 / Register basic analyzers
        registerAnalyzer("beat_analyzer", BeatAnalyzerImpl.class,
                         AnalyzerCategory.BASIC, "节拍分析器", "1.0.0");
        registerAnalyzer("key_analyzer", KeyAnalyzerImpl.class,
                         AnalyzerCategory.BASIC, "调性分析器", "1.0.0");
        registerAnalyzer("chord_analyzer", ChordAnalyzerImpl.class,
                         AnalyzerCategory.BASIC, "和弦分析器", "1.0.0");
        
        // 注册特征提取器 / Register feature extractor
        registerAnalyzer("feature_extractor", FeatureExtractorImpl.class,
                         AnalyzerCategory.FEATURE, "特征提取器", "1.0.0");
        
        // 注册高级分析器 / Register advanced analyzers
        registerAnalyzer("emotion_analyzer", EmotionAnalyzer.class,
                         AnalyzerCategory.EMOTION, "情感分析器", "1.0.0");
        registerAnalyzer("genre_analyzer", GenreAnalyzer.class,
                         AnalyzerCategory.GENRE, "风格分析器", "1.0.0");
        registerAnalyzer("complexity_analyzer", ComplexityAnalyzer.class,
                         AnalyzerCategory.COMPLEXITY, "复杂度分析器", "1.0.0");
        registerAnalyzer("advanced_music_analyzer", AdvancedMusicAnalyzer.class,
                         AnalyzerCategory.ADVANCED, "高级音乐分析器", "1.0.0");
                         
        // 注册默认的音乐处理器 / Register default music processors
        registerProcessor("harmonizer", Harmonizer.class);
        registerProcessor("theory_processor", MusicTheoryProcessor.class);
        registerProcessor("quantizer", Quantizer.class);
        registerProcessor("transposer", Transposer.class);
        
        // 注册默认的音乐生成器 / Register default music generators
        // Note: The generator classes don't implement IMusicGenerator interface directly
        registerGenerator("chord_generator", ChordGenerator.class);
        registerGenerator("interval_generator", IntervalGenerator.class);
        registerGenerator("scale_generator", ScaleGenerator.class);
    }
    
    /**
     * 从类路径加载服务 / Load services from classpath
     * <p>
     * 使用ServiceLoader机制自动发现和注册组件。
     * Use ServiceLoader mechanism for automatic component discovery and registration.
     * </p>
     */
    private void loadServicesFromClasspath() {
        // 使用ServiceLoader机制自动发现和注册组件 / Use ServiceLoader mechanism for automatic component discovery and registration
        ServiceLoader<IMusicProcessor> processorLoader = ServiceLoader.load(IMusicProcessor.class);
        for (IMusicProcessor processor : processorLoader) {
            processorRegistry.put(processor.getName().toLowerCase(), processor.getClass());
        }

        ServiceLoader<IMusicGenerator> generatorLoader = ServiceLoader.load(IMusicGenerator.class);
        for (IMusicGenerator generator : generatorLoader) {
            generatorRegistry.put(generator.getGeneratorName().toLowerCase(), generator.getClass());
        }

        ServiceLoader<IMusicFilter> filterLoader = ServiceLoader.load(IMusicFilter.class);
        for (IMusicFilter filter : filterLoader) {
            // IMusicFilter doesn't have a getName() method, so we'll use a simple approach
            filterRegistry.put(filter.getClass().getSimpleName().toLowerCase(), filter.getClass());
        }
    }
    
    /**
     * 清空所有注册的分析器 / Clear all registered analyzers
     */
    public void clearAllAnalyzers() {
        registeredAnalyzers.clear();
    }
    
    /**
     * 获取工厂版本信息 / Get factory version information
     */
    public String getVersion() {
        return "1.1.0";
    }
    
    /**
     * 获取工厂描述信息 / Get factory description
     */
    public String getDescription() {
        return "YiShape-Math 音乐组件工厂 / YiShape-Math Music Component Factory";
    }
    
    /**
     * 分析器信息类 / Analyzer Information Class
     */
    private static class AnalyzerInfo<T> {
        private final Class<T> analyzerClass;
        private final AnalyzerCategory category;
        private final String description;
        private final String version;
        
        public AnalyzerInfo(Class<T> analyzerClass,
                           AnalyzerCategory category, String description, String version) {
            this.analyzerClass = analyzerClass;
            this.category = category;
            this.description = description;
            this.version = version;
        }
        
        public Class<T> getAnalyzerClass() { return analyzerClass; }
        public AnalyzerCategory getCategory() { return category; }
        public String getDescription() { return description; }
        public String getVersion() { return version; }
    }
    
    /**
     * 分析器分类枚举 / Analyzer Category Enum
     */
    public enum AnalyzerCategory {
        BASIC("基础", "Basic"),
        ADVANCED("高级", "Advanced"),
        COMPREHENSIVE("综合", "Comprehensive"),
        FEATURE("特征", "Feature"),
        EMOTION("情感", "Emotion"),
        GENRE("风格", "Genre"),
        COMPLEXITY("复杂度", "Complexity");
        
        private final String chineseName;
        private final String englishName;
        
        AnalyzerCategory(String chineseName, String englishName) {
            this.chineseName = chineseName;
            this.englishName = englishName;
        }
        
        public String getChineseName() { return chineseName; }
        public String getEnglishName() { return englishName; }
    }
}