package com.reremouse.lab.audio.pipeline;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.exception.AudioProcessingException;
import com.reremouse.lab.audio.factory.AudioComponentFactory;
import com.reremouse.lab.audio.filter.IBaseAudioFilter;
import com.reremouse.lab.audio.processing.IAdvancedAudioProcessor;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import com.reremouse.lab.audio.effect.IAudioEffect;
import com.reremouse.lab.audio.analysis.IAudioAnalyzer;

/**
 * 音频处理流水线构建器 / Audio Processing Pipeline Builder
 * <p>
 * 提供链式API来构建复杂的音频处理流水线。
 * Provides a fluent API to build complex audio processing pipelines.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class AudioPipelineBuilder {
    
    private final List<PipelineStep> steps;
    private final AudioComponentFactory factory;
    
    /**
     * 构造函数 / Constructor
     */
    public AudioPipelineBuilder() {
        this.steps = new ArrayList<>();
        this.factory = AudioComponentFactory.getInstance();
    }
    
    /**
     * 添加处理器步骤 / Add processor step
     * 
     * @param processorName 处理器名称 / Processor name
     * @return 构建器实例 / Builder instance
     */
    public AudioPipelineBuilder addProcessor(String processorName) {
        steps.add(new PipelineStep(AudioComponentFactory.ComponentType.PROCESSOR, processorName, null));
        return this;
    }
    
    /**
     * 添加带参数的处理器步骤 / Add processor step with parameters
     * 
     * @param processorName 处理器名称 / Processor name
     * @param parameters 参数映射 / Parameter map
     * @return 构建器实例 / Builder instance
     */
    public AudioPipelineBuilder addProcessor(String processorName, Map<String, Object> parameters) {
        steps.add(new PipelineStep(AudioComponentFactory.ComponentType.PROCESSOR, processorName, parameters));
        return this;
    }
    
    /**
     * 添加分析器步骤 / Add analyzer step
     * 
     * @param analyzerName 分析器名称 / Analyzer name
     * @return 构建器实例 / Builder instance
     */
    public AudioPipelineBuilder addAnalyzer(String analyzerName) {
        steps.add(new PipelineStep(AudioComponentFactory.ComponentType.ANALYZER, analyzerName, null));
        return this;
    }
    
    /**
     * 添加带参数的分析器步骤 / Add analyzer step with parameters
     * 
     * @param analyzerName 分析器名称 / Analyzer name
     * @param parameters 参数映射 / Parameter map
     * @return 构建器实例 / Builder instance
     */
    public AudioPipelineBuilder addAnalyzer(String analyzerName, Map<String, Object> parameters) {
        steps.add(new PipelineStep(AudioComponentFactory.ComponentType.ANALYZER, analyzerName, parameters));
        return this;
    }
    
    /**
     * 添加滤波器步骤 / Add filter step
     * 
     * @param filterName 滤波器名称 / Filter name
     * @return 构建器实例 / Builder instance
     */
    public AudioPipelineBuilder addFilter(String filterName) {
        steps.add(new PipelineStep(AudioComponentFactory.ComponentType.FILTER, filterName, null));
        return this;
    }
    
    /**
     * 添加带参数的滤波器步骤 / Add filter step with parameters
     * 
     * @param filterName 滤波器名称 / Filter name
     * @param parameters 参数映射 / Parameter map
     * @return 构建器实例 / Builder instance
     */
    public AudioPipelineBuilder addFilter(String filterName, Map<String, Object> parameters) {
        steps.add(new PipelineStep(AudioComponentFactory.ComponentType.FILTER, filterName, parameters));
        return this;
    }
    
    /**
     * 添加效果器步骤 / Add effect step
     * 
     * @param effectName 效果器名称 / Effect name
     * @return 构建器实例 / Builder instance
     */
    public AudioPipelineBuilder addEffect(String effectName) {
        steps.add(new PipelineStep(AudioComponentFactory.ComponentType.EFFECT, effectName, null));
        return this;
    }
    
    /**
     * 添加带参数的效果器步骤 / Add effect step with parameters
     * 
     * @param effectName 效果器名称 / Effect name
     * @param parameters 参数映射 / Parameter map
     * @return 构建器实例 / Builder instance
     */
    public AudioPipelineBuilder addEffect(String effectName, Map<String, Object> parameters) {
        steps.add(new PipelineStep(AudioComponentFactory.ComponentType.EFFECT, effectName, parameters));
        return this;
    }
    
    /**
     * 构建并执行流水线 / Build and execute pipeline
     * 
     * @param input 输入音频数据 / Input audio data
     * @return 处理后的音频数据 / Processed audio data
     * @throws AudioProcessingException 处理过程中发生错误 / Error occurred during processing
     */
    public AudioData buildAndExecute(AudioData input) throws AudioProcessingException {
        AudioData current = input;
        
        for (PipelineStep step : steps) {
            current = executeStep(current, step);
        }
        
        return current;
    }
    
    /**
     * 执行单个步骤 / Execute single step
     */
    @SuppressWarnings("unchecked")
    private AudioData executeStep(AudioData input, PipelineStep step) throws AudioProcessingException {
        switch (step.type) {
            case PROCESSOR:
                IAdvancedAudioProcessor processor = factory.createProcessor(step.componentName, step.parameters);
                return processor.process(input);
                
            case FILTER:
                IBaseAudioFilter filter = factory.createFilter(step.componentName);
                // Set parameters if provided
                if (step.parameters != null) {
                    // In a real implementation, you might need to handle filter-specific parameters
                }
                return filter.filter(input);
                
            case EFFECT:
                IAudioEffect effect = factory.createEffect(step.componentName);
                // Set parameters if provided
                if (step.parameters != null) {
                    // In a real implementation, you might need to handle effect-specific parameters
                }
                return effect.applyEffect(input);
                
            case ANALYZER:
                // 分析器通常不修改音频数据，但可能需要存储结果 / Analyzers typically don't modify audio data, but may need to store results
                IAudioAnalyzer analyzer = factory.createAnalyzer(step.componentName, step.parameters);
                // 执行分析但不修改输入 / Execute analysis but don't modify input
                analyzer.calculateSpectrum(input);
                return input;
                
            default:
                throw new AudioProcessingException("Unsupported component type: " + step.type);
        }
    }
    
    /**
     * 流水线步骤类 / Pipeline Step Class
     */
    private static class PipelineStep {
        final AudioComponentFactory.ComponentType type;
        final String componentName;
        final Map<String, Object> parameters;
        
        PipelineStep(AudioComponentFactory.ComponentType type, String componentName, Map<String, Object> parameters) {
            this.type = type;
            this.componentName = componentName;
            this.parameters = parameters != null ? new HashMap<>(parameters) : null;
        }
    }
    
    /**
     * 创建构建器实例 / Create builder instance
     * 
     * @return 构建器实例 / Builder instance
     */
    public static AudioPipelineBuilder create() {
        return new AudioPipelineBuilder();
    }
}