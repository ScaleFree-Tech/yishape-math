package com.yishape.lab.image.factory;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.image.core.*;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * 图像组件工厂类 / Image Component Factory Class
 * <p>
 * 使用工厂模式和注册表模式创建各种图像处理组件。
 * 支持动态注册、延迟加载和依赖注入。
 * </p>
 * <p>
 * Uses Factory and Registry patterns to create various image processing components.
 * Supports dynamic registration, lazy loading, and dependency injection.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public class ImageComponentFactory {

    private static final Logger log = LoggerFactory.getLogger(ImageComponentFactory.class);

    
    /**
     * 组件类型枚举 / Component Type Enum
     */
    public enum ComponentType {
        PROCESSOR,              // 处理器 / Processor
        ANALYZER,               // 分析器 / Analyzer
        FILTER,                 // 滤波器 / Filter
        TRANSFORMER,            // 变换器 / Transformer
        SEGMENTER,              // 分割器 / Segmenter
        ENHANCER,               // 增强器 / Enhancer
        DETECTOR,               // 检测器 / Detector
        CLASSIFIER,             // 分类器 / Classifier
        RECONSTRUCTOR          // 重建器 / Reconstructor
    }
    
    /**
     * 组件配置类 / Component Configuration Class
     */
    public static class ComponentConfig {
        private String name;
        private String description;
        private String version;
        private ComponentType type;
        private Map<String, Object> defaultParameters;
        private boolean singleton;
        private Class<?> implementationClass;
        private Supplier<?> factory;
        
        public ComponentConfig(String name, ComponentType type, Class<?> implementationClass) {
            this.name = name;
            this.type = type;
            this.implementationClass = implementationClass;
            this.defaultParameters = new java.util.HashMap<>();
            this.singleton = false;
        }
        
        public ComponentConfig(String name, ComponentType type, Supplier<?> factory) {
            this.name = name;
            this.type = type;
            this.factory = factory;
            this.defaultParameters = new java.util.HashMap<>();
            this.singleton = false;
        }
        
        // Getters and Setters
        public String getName() { return name; }
        public ComponentConfig setName(String name) { this.name = name; return this; }
        
        public String getDescription() { return description; }
        public ComponentConfig setDescription(String description) { this.description = description; return this; }
        
        public String getVersion() { return version; }
        public ComponentConfig setVersion(String version) { this.version = version; return this; }
        
        public ComponentType getType() { return type; }
        public ComponentConfig setType(ComponentType type) { this.type = type; return this; }
        
        public Map<String, Object> getDefaultParameters() { return defaultParameters; }
        public ComponentConfig setDefaultParameters(Map<String, Object> parameters) { 
            this.defaultParameters = parameters; return this; 
        }
        
        public boolean isSingleton() { return singleton; }
        public ComponentConfig setSingleton(boolean singleton) { this.singleton = singleton; return this; }
        
        public Class<?> getImplementationClass() { return implementationClass; }
        public ComponentConfig setImplementationClass(Class<?> clazz) { 
            this.implementationClass = clazz; return this; 
        }
        
        public Supplier<?> getFactory() { return factory; }
        public ComponentConfig setFactory(Supplier<?> factory) { this.factory = factory; return this; }
    }
    
    // 单例实例 / Singleton instance
    private static volatile ImageComponentFactory instance;
    
    // 组件注册表 / Component registry
    private final Map<String, ComponentConfig> componentRegistry = new ConcurrentHashMap<>();
    
    // 单例组件缓存 / Singleton component cache
    private final Map<String, Object> singletonCache = new ConcurrentHashMap<>();
    
    // 类型别名映射 / Type alias mapping
    private final Map<String, String> typeAliases = new ConcurrentHashMap<>();
    
    // 组件依赖关系 / Component dependencies
    private final Map<String, java.util.Set<String>> dependencies = new ConcurrentHashMap<>();
    
    /**
     * 私有构造函数 / Private constructor
     */
    private ImageComponentFactory() {
        initializeDefaultComponents();
    }
    
    /**
     * 获取工厂实例 / Get Factory Instance
     * 
     * @return 工厂实例 / Factory instance
     */
    public static ImageComponentFactory getInstance() {
        if (instance == null) {
            synchronized (ImageComponentFactory.class) {
                if (instance == null) {
                    instance = new ImageComponentFactory();
                }
            }
        }
        return instance;
    }
    
    /**
     * 创建处理器 / Create Processor
     * 
     * @param name 处理器名称 / Processor name
     * @return 处理器实例 / Processor instance
     * @throws ImageProcessingException 创建失败 / Creation failed
     */
    public IImageProcessor createProcessor(String name) throws ImageProcessingException {
        return (IImageProcessor) createComponent(name, ComponentType.PROCESSOR);
    }
    
    /**
     * 创建处理器 / Create Processor
     * 
     * @param name 处理器名称 / Processor name
     * @param parameters 初始化参数 / Initialization parameters
     * @return 处理器实例 / Processor instance
     * @throws ImageProcessingException 创建失败 / Creation failed
     */
    public IImageProcessor createProcessor(String name, Map<String, Object> parameters) throws ImageProcessingException {
        return (IImageProcessor) createComponent(name, ComponentType.PROCESSOR, parameters);
    }
    
    /**
     * 创建分析器 / Create Analyzer
     * 
     * @param name 分析器名称 / Analyzer name
     * @return 分析器实例 / Analyzer instance
     * @throws ImageProcessingException 创建失败 / Creation failed
     */
    public IImageAnalyzer createAnalyzer(String name) throws ImageProcessingException {
        return (IImageAnalyzer) createComponent(name, ComponentType.ANALYZER);
    }
    
    /**
     * 创建分析器 / Create Analyzer
     * 
     * @param name 分析器名称 / Analyzer name
     * @param parameters 初始化参数 / Initialization parameters
     * @return 分析器实例 / Analyzer instance
     * @throws ImageProcessingException 创建失败 / Creation failed
     */
    public IImageAnalyzer createAnalyzer(String name, Map<String, Object> parameters) throws ImageProcessingException {
        return (IImageAnalyzer) createComponent(name, ComponentType.ANALYZER, parameters);
    }
    
    /**
     * 创建滤波器 / Create Filter
     * 
     * @param name 滤波器名称 / Filter name
     * @return 滤波器实例 / Filter instance
     * @throws ImageProcessingException 创建失败 / Creation failed
     */
    public IImageFilter createFilter(String name) throws ImageProcessingException {
        return (IImageFilter) createComponent(name, ComponentType.FILTER);
    }
    
    /**
     * 创建滤波器 / Create Filter
     * 
     * @param name 滤波器名称 / Filter name
     * @param parameters 初始化参数 / Initialization parameters
     * @return 滤波器实例 / Filter instance
     * @throws ImageProcessingException 创建失败 / Creation failed
     */
    public IImageFilter createFilter(String name, Map<String, Object> parameters) throws ImageProcessingException {
        return (IImageFilter) createComponent(name, ComponentType.FILTER, parameters);
    }
    
    /**
     * 创建变换器 / Create Transformer
     * 
     * @param name 变换器名称 / Transformer name
     * @return 变换器实例 / Transformer instance
     * @throws ImageProcessingException 创建失败 / Creation failed
     */
    public IImageTransformer createTransformer(String name) throws ImageProcessingException {
        return (IImageTransformer) createComponent(name, ComponentType.TRANSFORMER);
    }
    
    /**
     * 创建变换器 / Create Transformer
     * 
     * @param name 变换器名称 / Transformer name
     * @param parameters 初始化参数 / Initialization parameters
     * @return 变换器实例 / Transformer instance
     * @throws ImageProcessingException 创建失败 / Creation failed
     */
    public IImageTransformer createTransformer(String name, Map<String, Object> parameters) throws ImageProcessingException {
        return (IImageTransformer) createComponent(name, ComponentType.TRANSFORMER, parameters);
    }
    
    /**
     * 创建分割器 / Create Segmenter
     * 
     * @param name 分割器名称 / Segmenter name
     * @return 分割器实例 / Segmenter instance
     * @throws ImageProcessingException 创建失败 / Creation failed
     */
    public IImageSegmenter createSegmenter(String name) throws ImageProcessingException {
        return (IImageSegmenter) createComponent(name, ComponentType.SEGMENTER);
    }
    
    /**
     * 创建分割器 / Create Segmenter
     * 
     * @param name 分割器名称 / Segmenter name
     * @param parameters 初始化参数 / Initialization parameters
     * @return 分割器实例 / Segmenter instance
     * @throws ImageProcessingException 创建失败 / Creation failed
     */
    public IImageSegmenter createSegmenter(String name, Map<String, Object> parameters) throws ImageProcessingException {
        return (IImageSegmenter) createComponent(name, ComponentType.SEGMENTER, parameters);
    }
    
    /**
     * 创建通用组件 / Create Generic Component
     * 
     * @param name 组件名称 / Component name
     * @param type 组件类型 / Component type
     * @return 组件实例 / Component instance
     * @throws ImageProcessingException 创建失败 / Creation failed
     */
    @SuppressWarnings("unchecked")
    public <T> T createComponent(String name, ComponentType type) throws ImageProcessingException {
        return (T) createComponent(name, type, null);
    }
    
    /**
     * 创建通用组件 / Create Generic Component
     * 
     * @param name 组件名称 / Component name
     * @param type 组件类型 / Component type
     * @param parameters 初始化参数 / Initialization parameters
     * @return 组件实例 / Component instance
     * @throws ImageProcessingException 创建失败 / Creation failed
     */
    @SuppressWarnings("unchecked")
    public <T> T createComponent(String name, ComponentType type, Map<String, Object> parameters) 
            throws ImageProcessingException {
        
        // 解析别名 / Resolve alias
        String resolvedName = typeAliases.getOrDefault(name, name);
        
        // 获取组件配置 / Get component configuration
        ComponentConfig config = componentRegistry.get(resolvedName);
        if (config == null) {
            throw ImageProcessingException.invalidParameters("Unknown component: " + name);
        }
        
        // 验证类型 / Validate type
        if (config.getType() != type) {
            throw ImageProcessingException.invalidParameters(
                String.format("Component %s is not of type %s", name, type));
        }
        
        // 检查单例缓存 / Check singleton cache
        if (config.isSingleton()) {
            Object cached = singletonCache.get(resolvedName);
            if (cached != null) {
                return (T) cached;
            }
        }
        
        // 创建实例 / Create instance
        Object component = createInstance(config, parameters);
        
        // 缓存单例 / Cache singleton
        if (config.isSingleton()) {
            singletonCache.put(resolvedName, component);
        }
        
        return (T) component;
    }
    
    /**
     * 注册组件 / Register Component
     * 
     * @param config 组件配置 / Component configuration
     */
    public void registerComponent(ComponentConfig config) {
        componentRegistry.put(config.getName(), config);
    }
    
    /**
     * 注册组件 / Register Component
     * 
     * @param name 组件名称 / Component name
     * @param type 组件类型 / Component type
     * @param implementationClass 实现类 / Implementation class
     */
    public void registerComponent(String name, ComponentType type, Class<?> implementationClass) {
        ComponentConfig config = new ComponentConfig(name, type, implementationClass);
        registerComponent(config);
    }
    
    /**
     * 注册组件 / Register Component
     * 
     * @param name 组件名称 / Component name
     * @param type 组件类型 / Component type
     * @param factory 工厂函数 / Factory function
     */
    public void registerComponent(String name, ComponentType type, Supplier<?> factory) {
        ComponentConfig config = new ComponentConfig(name, type, factory);
        registerComponent(config);
    }
    
    /**
     * 注册别名 / Register Alias
     * 
     * @param alias 别名 / Alias
     * @param realName 真实名称 / Real name
     */
    public void registerAlias(String alias, String realName) {
        typeAliases.put(alias, realName);
    }
    
    /**
     * 取消注册组件 / Unregister Component
     * 
     * @param name 组件名称 / Component name
     */
    public void unregisterComponent(String name) {
        componentRegistry.remove(name);
        singletonCache.remove(name);
    }
    
    /**
     * 获取已注册的组件列表 / Get Registered Component List
     * 
     * @param type 组件类型 / Component type
     * @return 组件名称列表 / Component name list
     */
    public java.util.List<String> getRegisteredComponents(ComponentType type) {
        return componentRegistry.entrySet().stream()
                .filter(entry -> entry.getValue().getType() == type)
                .map(Map.Entry::getKey)
                .collect(java.util.stream.Collectors.toList());
    }
    
    /**
     * 获取所有已注册的组件列表 / Get All Registered Component List
     * 
     * @return 组件名称列表 / Component name list
     */
    public java.util.List<String> getAllRegisteredComponents() {
        return new java.util.ArrayList<>(componentRegistry.keySet());
    }
    
    /**
     * 获取组件配置 / Get Component Configuration
     * 
     * @param name 组件名称 / Component name
     * @return 组件配置 / Component configuration
     */
    public ComponentConfig getComponentConfig(String name) {
        return componentRegistry.get(name);
    }
    
    /**
     * 检查组件是否已注册 / Check if Component is Registered
     * 
     * @param name 组件名称 / Component name
     * @return 是否已注册 / Whether registered
     */
    public boolean isComponentRegistered(String name) {
        return componentRegistry.containsKey(name);
    }
    
    /**
     * 清理单例缓存 / Clear Singleton Cache
     */
    public void clearSingletonCache() {
        singletonCache.clear();
    }
    
    /**
     * 清理所有注册 / Clear All Registrations
     */
    public void clearAll() {
        componentRegistry.clear();
        singletonCache.clear();
        typeAliases.clear();
        dependencies.clear();
    }
    
    // ========== 私有方法 / Private Methods ==========
    
    /**
     * 创建实例 / Create Instance
     */
    private Object createInstance(ComponentConfig config, Map<String, Object> parameters) 
            throws ImageProcessingException {
        try {
            Object instance;
            
            if (config.getFactory() != null) {
                // 使用工厂函数创建 / Create using factory function
                instance = config.getFactory().get();
            } else if (config.getImplementationClass() != null) {
                // 使用反射创建 / Create using reflection
                instance = config.getImplementationClass().getDeclaredConstructor().newInstance();
            } else {
                throw ImageProcessingException.processingFailed("ComponentFactory", 
                    "No factory or implementation class configured for " + config.getName());
            }
            
            // 应用参数 / Apply parameters
            if (parameters != null && !parameters.isEmpty()) {
                applyParameters(instance, parameters);
            } else if (config.getDefaultParameters() != null && !config.getDefaultParameters().isEmpty()) {
                applyParameters(instance, config.getDefaultParameters());
            }
            
            return instance;
            
        } catch (Exception e) {
            throw ImageProcessingException.processingFailed("ComponentFactory", 
                "Failed to create component " + config.getName(), e);
        }
    }
    
    /**
     * 应用参数 / Apply Parameters
     */
    private void applyParameters(Object instance, Map<String, Object> parameters) {
        // 使用反射设置参数 / Use reflection to set parameters
        // 这里简化实现，实际应用中可以使用更复杂的参数注入机制
        // Simplified implementation here, more complex parameter injection can be used in practice
        
        for (Map.Entry<String, Object> entry : parameters.entrySet()) {
            try {
                String paramName = entry.getKey();
                Object paramValue = entry.getValue();
                
                // 尝试查找并调用setter方法 / Try to find and call setter method
                String setterName = "set" + Character.toUpperCase(paramName.charAt(0)) + paramName.substring(1);
                java.lang.reflect.Method[] methods = instance.getClass().getMethods();
                
                for (java.lang.reflect.Method method : methods) {
                    if (method.getName().equals(setterName) && method.getParameterCount() == 1) {
                        method.invoke(instance, paramValue);
                        break;
                    }
                }
            } catch (Exception e) {
                // 忽略参数设置错误 / Ignore parameter setting errors
                log.warn("Warning: Failed to set parameter " + entry.getKey() + ": " + e.getMessage());
            }
        }
    }
    
    /**
     * 初始化默认组件 / Initialize Default Components
     */
    private void initializeDefaultComponents() {
        // 这里注册默认的组件实现 / Register default component implementations here
        // 将在后续的实现中添加具体的组件 / Specific components will be added in subsequent implementations
        
        // 注册别名 / Register aliases
        registerAlias("gaussian", "GaussianFilter");
        registerAlias("sobel", "SobelFilter");
        registerAlias("canny", "CannyEdgeDetector");
        registerAlias("otsu", "OtsuSegmenter");
        registerAlias("kmeans", "KMeansSegmenter");
        registerAlias("fft", "FFTTransformer");
        registerAlias("dct", "DCTTransformer");
        registerAlias("harris", "HarrisCornerDetector");
        registerAlias("sift", "SIFTFeatureDetector");
        registerAlias("surf", "SURFFeatureDetector");
    }
}