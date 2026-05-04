package com.yishape.lab.math.signal.factory;

import com.yishape.lab.math.signal.core.ISignalProcessor;
import com.yishape.lab.math.signal.core.SignalProcessingException;
import com.yishape.lab.math.signal.transform.ISignalTransform;
import com.yishape.lab.math.signal.filter.ISignalFilter;
import com.yishape.lab.math.signal.generation.ISignalGenerator;
import com.yishape.lab.math.signal.analysis.ISignalAnalyzer;
import com.yishape.lab.math.signal.transform.ChirpZTransform;
import com.yishape.lab.math.signal.transform.WalshHadamardTransform;
import com.yishape.lab.math.signal.transform.ZTransform;
import com.yishape.lab.math.signal.filter.ButterworthFilter;
import com.yishape.lab.math.signal.filter.ChebyshevFilter;
import com.yishape.lab.math.signal.filter.EllipticFilter;
import com.yishape.lab.math.signal.filter.BesselFilter;
import com.yishape.lab.math.signal.filter.GaussianFilter;
import com.yishape.lab.math.signal.filter.MovingAverageFilter;
import com.yishape.lab.math.signal.filter.MedianFilter;
import com.yishape.lab.math.signal.filter.BandpassFilter;
import com.yishape.lab.math.signal.filter.KalmanFilter;
import com.yishape.lab.math.signal.filter.WienerFilter;
import com.yishape.lab.math.signal.filter.BandStopFilter;
import com.yishape.lab.math.signal.analysis.SpectrumAnalyzer;
import com.yishape.lab.math.signal.generation.SignalGenerator;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 信号处理器工厂类 / Signal Processor Factory Class
 * <p>
 * 使用工厂模式和单例模式管理所有信号处理器的创建。
 * 支持插件式注册新的处理器实现。
 * </p>
 * <p>
 * Uses Factory and Singleton patterns to manage creation of all signal processors.
 * Supports plugin-style registration of new processor implementations.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class SignalProcessorFactory {
    
    private static volatile SignalProcessorFactory instance;
    private final Map<String, ProcessorInfo<?>> registeredProcessors;
    
    /**
     * 处理器信息类 / Processor Information Class
     */
    private static class ProcessorInfo<T extends Number> {
        private final Class<? extends ISignalProcessor<T>> processorClass;
        private final ProcessorCategory category;
        private final String description;
        private final String version;
        
        public ProcessorInfo(Class<? extends ISignalProcessor<T>> processorClass, 
                           ProcessorCategory category, String description, String version) {
            this.processorClass = processorClass;
            this.category = category;
            this.description = description;
            this.version = version;
        }
        
        public Class<? extends ISignalProcessor<T>> getProcessorClass() { return processorClass; }
        public ProcessorCategory getCategory() { return category; }
        public String getDescription() { return description; }
        public String getVersion() { return version; }
    }
    
    /**
     * 处理器分类枚举 / Processor Category Enum
     */
    public enum ProcessorCategory {
        TRANSFORM("变换", "Transform"),
        FILTER("滤波", "Filter"),
        GENERATOR("生成", "Generator"),
        ANALYZER("分析", "Analyzer"),
        ADAPTIVE("自适应", "Adaptive"),
        DETECTION("检测", "Detection"),
        ESTIMATION("估计", "Estimation"),
        ENHANCEMENT("增强", "Enhancement");
        
        private final String chineseName;
        private final String englishName;
        
        ProcessorCategory(String chineseName, String englishName) {
            this.chineseName = chineseName;
            this.englishName = englishName;
        }
        
        public String getChineseName() { return chineseName; }
        public String getEnglishName() { return englishName; }
    }
    
    /**
     * 私有构造函数 / Private constructor
     */
    private SignalProcessorFactory() {
        this.registeredProcessors = new ConcurrentHashMap<>();
        registerDefaultProcessors();
    }
    
    /**
     * 获取单例实例 / Get singleton instance
     *
     * @return SignalProcessorFactory 单例实例 / Singleton instance
     */
    public static SignalProcessorFactory getInstance() {
        if (instance == null) {
            synchronized (SignalProcessorFactory.class) {
                if (instance == null) {
                    instance = new SignalProcessorFactory();
                }
            }
        }
        return instance;
    }
    
    /**
     * 注册信号处理器 / Register signal processor
     * <p>
     * 注册新的信号处理器实现。
     * Register new signal processor implementation.
     * </p>
     *
     * @param name 处理器名称 / Processor name
     * @param processorClass 处理器类 / Processor class
     * @param category 处理器分类 / Processor category
     * @param description 描述信息 / Description
     * @param version 版本信息 / Version
     * @param <T> 数据类型 / Data type
     */
    public <T extends Number> void registerProcessor(String name, 
                                                   Class<? extends ISignalProcessor<T>> processorClass,
                                                   ProcessorCategory category,
                                                   String description,
                                                   String version) {
        ProcessorInfo<T> info = new ProcessorInfo<>(processorClass, category, description, version);
        registeredProcessors.put(name.toLowerCase(), info);
    }
    
    /**
     * 创建信号处理器 / Create signal processor
     * <p>
     * 根据名称创建指定的信号处理器实例。
     * Create specified signal processor instance by name.
     * </p>
     *
     * @param name 处理器名称 / Processor name
     * @param <T> 数据类型 / Data type
     * @return 信号处理器实例 / Signal processor instance
     * @throws SignalProcessingException 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     */
    @SuppressWarnings("unchecked")
    public <T extends Number> ISignalProcessor<T> createProcessor(String name) throws SignalProcessingException {
        ProcessorInfo<T> info = (ProcessorInfo<T>) registeredProcessors.get(name.toLowerCase());
        if (info == null) {
            throw new SignalProcessingException("未找到处理器：" + name + " / Processor not found: " + name);
        }
        
        try {
            // For special cases that need parameters, we handle them explicitly
            Class<? extends ISignalProcessor<T>> processorClass = info.getProcessorClass();
            
            // Handle ButterworthFilter special case
            if (ButterworthFilter.class.isAssignableFrom(processorClass)) {
                // Create with default parameters (4th order, 100Hz cutoff, 1000Hz sampling rate)
                return (ISignalProcessor<T>) new ButterworthFilter(4, 100.0, 1000.0);
            }
            
            // Handle other filters that might need default parameters
            if (ChebyshevFilter.class.isAssignableFrom(processorClass)) {
                // Create with default parameters (ChebyshevType.TYPE_I, 4th order, 100Hz cutoff, 1000Hz sampling rate, 0.5dB ripple)
                return (ISignalProcessor<T>) new ChebyshevFilter(ChebyshevFilter.ChebyshevType.TYPE_I, 4, 100.0, 1000.0, 0.5);
            }
            
            if (EllipticFilter.class.isAssignableFrom(processorClass)) {
                // Create with default parameters (4th order, 100Hz cutoff, 1000Hz sampling rate, 0.5dB passband ripple, 40dB stopband ripple)
                return (ISignalProcessor<T>) new EllipticFilter(4, 100.0, 1000.0, 0.5, 40.0);
            }
            
            if (BesselFilter.class.isAssignableFrom(processorClass)) {
                // Create with default parameters (4th order, 100Hz cutoff, 1000Hz sampling rate)
                return (ISignalProcessor<T>) new BesselFilter(4, 100.0, 1000.0);
            }
            
            if (GaussianFilter.class.isAssignableFrom(processorClass)) {
                // Create with default parameters (sigma = 1.0)
                return (ISignalProcessor<T>) new GaussianFilter(1.0);
            }
            
            if (MovingAverageFilter.class.isAssignableFrom(processorClass)) {
                // Create with default parameters (window size = 5)
                return (ISignalProcessor<T>) new MovingAverageFilter(5);
            }
            
            if (MedianFilter.class.isAssignableFrom(processorClass)) {
                // Create with default parameters (window size = 5, must be odd)
                return (ISignalProcessor<T>) new MedianFilter(5);
            }
            
            if (BandpassFilter.class.isAssignableFrom(processorClass)) {
                // Create with default parameters (4th order, 50Hz low cutoff, 150Hz high cutoff, 1000Hz sampling rate)
                return (ISignalProcessor<T>) new BandpassFilter(4, 50.0, 150.0, 1000.0);
            }
            
            if (KalmanFilter.class.isAssignableFrom(processorClass)) {
                // Create with default parameters (process noise variance = 1.0, measurement noise variance = 1.0)
                return (ISignalProcessor<T>) new KalmanFilter(1.0, 1.0);
            }
            
            if (WienerFilter.class.isAssignableFrom(processorClass)) {
                // Create with default parameters (signal power = 1.0, noise power = 0.1, filter length = 10)
                return (ISignalProcessor<T>) new WienerFilter(1.0, 0.1, 10);
            }
            
            if (BandStopFilter.class.isAssignableFrom(processorClass)) {
                // Create with default parameters (4th order, 50Hz low cutoff, 150Hz high cutoff, 1000Hz sampling rate)
                return (ISignalProcessor<T>) new BandStopFilter(4, 50.0, 150.0, 1000.0);
            }
            
            // Default case - try to create with no-argument constructor
            return processorClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new SignalProcessingException("创建处理器失败 / Failed to create processor: " + name, e);
        }
    }
    
    /**
     * 创建变换器 / Create transformer
     * <p>
     * 创建信号变换器实例。
     * Create signal transformer instance.
     * </p>
     *
     * @param transformType 变换类型 / Transform type
     * @param <T> 输入数据类型 / Input data type
     * @param <R> 输出数据类型 / Output data type
     * @return 信号变换器实例 / Signal transformer instance
     * @throws SignalProcessingException 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     */
    @SuppressWarnings("unchecked")
    public <T extends Number, R> ISignalTransform<T, R> createTransform(String transformType) throws SignalProcessingException {
        ISignalProcessor<T> processor = createProcessor(transformType);
        if (!(processor instanceof ISignalTransform)) {
            throw new SignalProcessingException("处理器不是变换器类型 / Processor is not a transformer type: " + transformType);
        }
        return (ISignalTransform<T, R>) processor;
    }
    
    /**
     * 创建滤波器 / Create filter
     * <p>
     * 创建信号滤波器实例。
     * Create signal filter instance.
     * </p>
     *
     * @param filterType 滤波器类型 / Filter type
     * @param <T> 数据类型 / Data type
     * @return 信号滤波器实例 / Signal filter instance
     * @throws SignalProcessingException 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     */
    @SuppressWarnings("unchecked")
    public <T extends Number> ISignalFilter<T> createFilter(String filterType) throws SignalProcessingException {
        ISignalProcessor<T> processor = createProcessor(filterType);
        if (!(processor instanceof ISignalFilter)) {
            throw new SignalProcessingException("处理器不是滤波器类型 / Processor is not a filter type: " + filterType);
        }
        return (ISignalFilter<T>) processor;
    }
    
    /**
     * 创建生成器 / Create generator
     * <p>
     * 创建信号生成器实例。
     * Create signal generator instance.
     * </p>
     *
     * @param generatorType 生成器类型 / Generator type
     * @param <T> 数据类型 / Data type
     * @return 信号生成器实例 / Signal generator instance
     * @throws SignalProcessingException 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     */
    @SuppressWarnings("unchecked")
    public <T extends Number> ISignalGenerator<T> createGenerator(String generatorType) throws SignalProcessingException {
        // Check if it's registered as a processor
        ProcessorInfo<T> info = (ProcessorInfo<T>) registeredProcessors.get(generatorType.toLowerCase());
        if (info != null) {
            try {
                return (ISignalGenerator<T>) info.getProcessorClass().getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                throw new SignalProcessingException("创建生成器失败 / Failed to create generator: " + generatorType, e);
            }
        }
        
        // If not found as processor, throw exception
        throw new SignalProcessingException("未找到生成器：" + generatorType + " / Generator not found: " + generatorType);
    }
    
    /**
     * 创建分析器 / Create analyzer
     * <p>
     * 创建信号分析器实例。
     * Create signal analyzer instance.
     * </p>
     *
     * @param analyzerType 分析器类型 / Analyzer type
     * @param <T> 数据类型 / Data type
     * @return 信号分析器实例 / Signal analyzer instance
     * @throws SignalProcessingException 创建过程中发生错误时抛出 / Thrown when errors occur during creation
     */
    @SuppressWarnings("unchecked")
    public <T extends Number> ISignalAnalyzer<T> createAnalyzer(String analyzerType) throws SignalProcessingException {
        ISignalProcessor<T> processor = createProcessor(analyzerType);
        if (!(processor instanceof ISignalAnalyzer)) {
            throw new SignalProcessingException("处理器不是分析器类型 / Processor is not an analyzer type: " + analyzerType);
        }
        return (ISignalAnalyzer<T>) processor;
    }
    
    /**
     * 获取已注册的处理器列表 / Get list of registered processors
     *
     * @return 已注册处理器名称的集合 / Set of registered processor names
     */
    public Set<String> getRegisteredProcessorNames() {
        return registeredProcessors.keySet();
    }
    
    /**
     * 获取指定分类的处理器列表 / Get list of processors in specified category
     *
     * @param category 处理器分类 / Processor category
     * @return 指定分类的处理器映射 / Map of processors in specified category
     */
    public Map<String, ProcessorInfo<?>> getProcessorsByCategory(ProcessorCategory category) {
        Map<String, ProcessorInfo<?>> result = new HashMap<>();
        for (Map.Entry<String, ProcessorInfo<?>> entry : registeredProcessors.entrySet()) {
            if (entry.getValue().getCategory() == category) {
                result.put(entry.getKey(), entry.getValue());
            }
        }
        return result;
    }
    
    /**
     * 检查处理器是否已注册 / Check if processor is registered
     *
     * @param name 处理器名称 / Processor name
     * @return 是否已注册 / Whether registered
     */
    public boolean isProcessorRegistered(String name) {
        return registeredProcessors.containsKey(name.toLowerCase());
    }
    
    /**
     * 获取处理器信息 / Get processor information
     *
     * @param name 处理器名称 / Processor name
     * @return 处理器信息 / Processor information
     */
    public ProcessorInfo<?> getProcessorInfo(String name) {
        return registeredProcessors.get(name.toLowerCase());
    }
    
    /**
     * 注销处理器 / Unregister processor
     *
     * @param name 处理器名称 / Processor name
     */
    public void unregisterProcessor(String name) {
        registeredProcessors.remove(name.toLowerCase());
    }
    
    /**
     * 注册默认处理器 / Register default processors
     * <p>
     * 注册内置的默认信号处理器。
     * Register built-in default signal processors.
     * </p>
     */
    private void registerDefaultProcessors() {
        // 注册变换器 / Register transformers
        registerProcessor("fft", ChirpZTransform.class, ProcessorCategory.TRANSFORM, "Fast Fourier Transform", "1.0.0");
        registerProcessor("dct", ChirpZTransform.class, ProcessorCategory.TRANSFORM, "Discrete Cosine Transform", "1.0.0");
        registerProcessor("hilbert", ChirpZTransform.class, ProcessorCategory.TRANSFORM, "Hilbert Transform", "1.0.0");
        registerProcessor("wavelet", ChirpZTransform.class, ProcessorCategory.TRANSFORM, "Wavelet Transform", "1.0.0");
        registerProcessor("ztransform", ZTransform.class, ProcessorCategory.TRANSFORM, "Z Transform", "1.0.0");
        registerProcessor("chirpz", ChirpZTransform.class, ProcessorCategory.TRANSFORM, "Chirp-Z Transform", "1.0.0");
        registerProcessor("walsh", WalshHadamardTransform.class, ProcessorCategory.TRANSFORM, "Walsh-Hadamard Transform", "1.0.0");
        
        // 注册滤波器 / Register filters
        registerProcessor("butterworth", ButterworthFilter.class, ProcessorCategory.FILTER, "Butterworth Filter", "1.0.0");
        registerProcessor("chebyshev", ChebyshevFilter.class, ProcessorCategory.FILTER, "Chebyshev Filter", "1.0.0");
        registerProcessor("elliptic", EllipticFilter.class, ProcessorCategory.FILTER, "Elliptic Filter", "1.0.0");
        registerProcessor("bessel", BesselFilter.class, ProcessorCategory.FILTER, "Bessel Filter", "1.0.0");
        registerProcessor("gaussian", GaussianFilter.class, ProcessorCategory.FILTER, "Gaussian Filter", "1.0.0");
        registerProcessor("movingaverage", MovingAverageFilter.class, ProcessorCategory.FILTER, "Moving Average Filter", "1.0.0");
        registerProcessor("median", MedianFilter.class, ProcessorCategory.FILTER, "Median Filter", "1.0.0");
        registerProcessor("bandpass", BandpassFilter.class, ProcessorCategory.FILTER, "Bandpass Filter", "1.0.0");
        
        // 注册新滤波器 / Register new filters
        registerProcessor("kalman", KalmanFilter.class, ProcessorCategory.FILTER, "Kalman Filter", "1.0.0");
        registerProcessor("wiener", WienerFilter.class, ProcessorCategory.FILTER, "Wiener Filter", "1.0.0");
        registerProcessor("bandstop", BandStopFilter.class, ProcessorCategory.FILTER, "Band-stop Filter", "1.0.0");
        
        // 注册分析器 / Register analyzers
        registerProcessor("spectrum", SpectrumAnalyzer.class, ProcessorCategory.ANALYZER, "Spectrum Analyzer", "1.0.0");
        registerProcessor("psd", SpectrumAnalyzer.class, ProcessorCategory.ANALYZER, "Power Spectral Density Analyzer", "1.0.0");
        registerProcessor("autocorr", SpectrumAnalyzer.class, ProcessorCategory.ANALYZER, "Autocorrelation Analyzer", "1.0.0");
        registerProcessor("crosscorr", SpectrumAnalyzer.class, ProcessorCategory.ANALYZER, "Cross-correlation Analyzer", "1.0.0");
        registerProcessor("wavelet", SpectrumAnalyzer.class, ProcessorCategory.ANALYZER, "Wavelet Analyzer", "1.0.0");
        registerProcessor("envelope", SpectrumAnalyzer.class, ProcessorCategory.ANALYZER, "Envelope Analyzer", "1.0.0");
        registerProcessor("instantaneous", SpectrumAnalyzer.class, ProcessorCategory.ANALYZER, "Instantaneous Feature Analyzer", "1.0.0");
        
        // 注册生成器 / Register generators
        registerProcessor("sine", SignalGenerator.class, ProcessorCategory.GENERATOR, "Sine Wave Generator", "1.0.0");
        registerProcessor("cosine", SignalGenerator.class, ProcessorCategory.GENERATOR, "Cosine Wave Generator", "1.0.0");
        registerProcessor("square", SignalGenerator.class, ProcessorCategory.GENERATOR, "Square Wave Generator", "1.0.0");
        registerProcessor("triangle", SignalGenerator.class, ProcessorCategory.GENERATOR, "Triangle Wave Generator", "1.0.0");
        registerProcessor("sawtooth", SignalGenerator.class, ProcessorCategory.GENERATOR, "Sawtooth Wave Generator", "1.0.0");
        registerProcessor("noise", SignalGenerator.class, ProcessorCategory.GENERATOR, "Noise Generator", "1.0.0");
        registerProcessor("chirp", SignalGenerator.class, ProcessorCategory.GENERATOR, "Chirp Signal Generator", "1.0.0");
        registerProcessor("pulse", SignalGenerator.class, ProcessorCategory.GENERATOR, "Pulse Signal Generator", "1.0.0");
        
        // 注册通用信号生成器 / Register generic signal generator
        registerProcessor("signal", SignalGenerator.class, ProcessorCategory.GENERATOR, "Generic Signal Generator", "1.0.0");
    }
    
    /**
     * 清空所有注册的处理器 / Clear all registered processors
     */
    public void clearAllProcessors() {
        registeredProcessors.clear();
    }
    
    /**
     * 获取工厂版本信息 / Get factory version information
     *
     * @return 工厂版本 / Factory version
     */
    public String getVersion() {
        return "1.0.0";
    }
    
    /**
     * 获取工厂描述信息 / Get factory description
     *
     * @return 工厂描述 / Factory description
     */
    public String getDescription() {
        return "YiShape-Math 信号处理器工厂 / YiShape-Math Signal Processor Factory";
    }
}