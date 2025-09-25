package com.reremouse.lab.image.factory;

import com.reremouse.lab.image.ImageData;
import com.reremouse.lab.image.core.*;
import java.util.Map;
import java.util.function.Function;

/**
 * 图像处理流水线构建器 / Image Processing Pipeline Builder
 * <p>
 * 使用建造者模式构建复杂的图像处理流水线。
 * 支持链式调用、条件分支、并行处理和错误处理。
 * </p>
 * <p>
 * Uses Builder pattern to construct complex image processing pipelines.
 * Supports method chaining, conditional branching, parallel processing and error handling.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public class ImagePipelineBuilder {
    
    /**
     * 流水线步骤接口 / Pipeline Step Interface
     */
    @FunctionalInterface
    public interface PipelineStep {
        /**
         * 执行步骤 / Execute Step
         * 
         * @param input 输入图像 / Input image
         * @param context 执行上下文 / Execution context
         * @return 输出图像 / Output image
         * @throws ImageProcessingException 处理异常 / Processing exception
         */
        ImageData execute(ImageData input, PipelineContext context) throws ImageProcessingException;
    }
    
    /**
     * 流水线上下文类 / Pipeline Context Class
     */
    public static class PipelineContext {
        private final Map<String, Object> parameters = new java.util.concurrent.ConcurrentHashMap<>();
        private final Map<String, ImageData> intermediateResults = new java.util.concurrent.ConcurrentHashMap<>();
        private final java.util.List<String> executionLog = new java.util.concurrent.CopyOnWriteArrayList<>();
        private boolean debugMode = false;
        private long startTime;
        
        public PipelineContext() {
            this.startTime = System.currentTimeMillis();
        }
        
        public void setParameter(String key, Object value) {
            parameters.put(key, value);
        }
        
        @SuppressWarnings("unchecked")
        public <T> T getParameter(String key) {
            return (T) parameters.get(key);
        }
        
        public <T> T getParameter(String key, T defaultValue) {
            @SuppressWarnings("unchecked")
            T value = (T) parameters.get(key);
            return value != null ? value : defaultValue;
        }
        
        public void storeIntermediateResult(String stepName, ImageData result) {
            intermediateResults.put(stepName, result);
        }
        
        public ImageData getIntermediateResult(String stepName) {
            return intermediateResults.get(stepName);
        }
        
        public void log(String message) {
            String logEntry = String.format("[%d] %s", System.currentTimeMillis() - startTime, message);
            executionLog.add(logEntry);
            if (debugMode) {
                System.out.println(logEntry);
            }
        }
        
        public java.util.List<String> getExecutionLog() {
            return new java.util.ArrayList<>(executionLog);
        }
        
        public boolean isDebugMode() {
            return debugMode;
        }
        
        public void setDebugMode(boolean debugMode) {
            this.debugMode = debugMode;
        }
        
        public long getElapsedTime() {
            return System.currentTimeMillis() - startTime;
        }
    }
    
    /**
     * 可执行流水线类 / Executable Pipeline Class
     */
    public static class ImagePipeline {
        private final java.util.List<PipelineStep> steps;
        private final String name;
        private final String description;
        private boolean parallel;
        private int maxThreads;
        private long timeoutMs;
        
        public ImagePipeline(String name, java.util.List<PipelineStep> steps) {
            this.name = name;
            this.steps = new java.util.ArrayList<>(steps);
            this.description = "";
            this.parallel = false;
            this.maxThreads = Runtime.getRuntime().availableProcessors();
            this.timeoutMs = 30000; // 30 seconds default
        }
        
        /**
         * 执行流水线 / Execute Pipeline
         * 
         * @param input 输入图像 / Input image
         * @return 输出图像 / Output image
         * @throws ImageProcessingException 处理异常 / Processing exception
         */
        public ImageData execute(ImageData input) throws ImageProcessingException {
            return execute(input, new PipelineContext());
        }
        
        /**
         * 执行流水线 / Execute Pipeline
         * 
         * @param input 输入图像 / Input image
         * @param context 执行上下文 / Execution context
         * @return 输出图像 / Output image
         * @throws ImageProcessingException 处理异常 / Processing exception
         */
        public ImageData execute(ImageData input, PipelineContext context) throws ImageProcessingException {
            context.log("Starting pipeline: " + name);
            ImageData current = input;
            
            for (int i = 0; i < steps.size(); i++) {
                PipelineStep step = steps.get(i);
                String stepName = "Step" + (i + 1);
                
                try {
                    context.log("Executing " + stepName);
                    long stepStart = System.currentTimeMillis();
                    
                    current = step.execute(current, context);
                    
                    long stepTime = System.currentTimeMillis() - stepStart;
                    context.log(stepName + " completed in " + stepTime + " ms");
                    context.storeIntermediateResult(stepName, current);
                    
                } catch (Exception e) {
                    context.log("Error in " + stepName + ": " + e.getMessage());
                    throw ImageProcessingException.processingFailed("Pipeline", 
                        "Step " + (i + 1) + " failed in pipeline " + name, e);
                }
            }
            
            context.log("Pipeline completed in " + context.getElapsedTime() + " ms");
            return current;
        }
        
        public String getName() { return name; }
        public String getDescription() { return description; }
        public int getStepCount() { return steps.size(); }
        public boolean isParallel() { return parallel; }
        public void setParallel(boolean parallel) { this.parallel = parallel; }
        public void setMaxThreads(int maxThreads) { this.maxThreads = maxThreads; }
        public void setTimeoutMs(long timeoutMs) { this.timeoutMs = timeoutMs; }
    }
    
    // 流水线步骤列表 / Pipeline step list
    private final java.util.List<PipelineStep> steps = new java.util.ArrayList<>();
    private final ImageComponentFactory factory = ImageComponentFactory.getInstance();
    
    // 流水线配置 / Pipeline configuration
    private String pipelineName = "ImagePipeline";
    private String pipelineDescription = "";
    private boolean enableDebug = false;
    private boolean enableParallel = false;
    
    /**
     * 创建新的流水线构建器 / Create New Pipeline Builder
     * 
     * @return 流水线构建器 / Pipeline builder
     */
    public static ImagePipelineBuilder create() {
        return new ImagePipelineBuilder();
    }
    
    /**
     * 创建命名的流水线构建器 / Create Named Pipeline Builder
     * 
     * @param name 流水线名称 / Pipeline name
     * @return 流水线构建器 / Pipeline builder
     */
    public static ImagePipelineBuilder create(String name) {
        return new ImagePipelineBuilder().name(name);
    }
    
    /**
     * 设置流水线名称 / Set Pipeline Name
     * 
     * @param name 流水线名称 / Pipeline name
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder name(String name) {
        this.pipelineName = name;
        return this;
    }
    
    /**
     * 设置流水线描述 / Set Pipeline Description
     * 
     * @param description 流水线描述 / Pipeline description
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder description(String description) {
        this.pipelineDescription = description;
        return this;
    }
    
    /**
     * 启用调试模式 / Enable Debug Mode
     * 
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder debug() {
        this.enableDebug = true;
        return this;
    }
    
    /**
     * 启用并行处理 / Enable Parallel Processing
     * 
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder parallel() {
        this.enableParallel = true;
        return this;
    }
    
    /**
     * 添加处理器步骤 / Add Processor Step
     * 
     * @param processorName 处理器名称 / Processor name
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder addProcessor(String processorName) {
        return addStep((input, context) -> {
            IImageProcessor processor = factory.createProcessor(processorName);
            return processor.process(input);
        });
    }
    
    /**
     * 添加处理器步骤 / Add Processor Step
     * 
     * @param processorName 处理器名称 / Processor name
     * @param parameters 处理参数 / Processing parameters
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder addProcessor(String processorName, Map<String, Object> parameters) {
        return addStep((input, context) -> {
            IImageProcessor processor = factory.createProcessor(processorName, parameters);
            return processor.process(input);
        });
    }
    
    /**
     * 添加滤波器步骤 / Add Filter Step
     * 
     * @param filterName 滤波器名称 / Filter name
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder addFilter(String filterName) {
        return addStep((input, context) -> {
            IImageFilter filter = factory.createFilter(filterName);
            return filter.process(input);
        });
    }
    
    /**
     * 添加滤波器步骤 / Add Filter Step
     * 
     * @param filterName 滤波器名称 / Filter name
     * @param parameters 滤波参数 / Filter parameters
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder addFilter(String filterName, Map<String, Object> parameters) {
        return addStep((input, context) -> {
            IImageFilter filter = factory.createFilter(filterName, parameters);
            return filter.process(input, parameters);
        });
    }
    
    /**
     * 添加变换器步骤 / Add Transformer Step
     * 
     * @param transformerName 变换器名称 / Transformer name
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder addTransformer(String transformerName) {
        return addStep((input, context) -> {
            IImageTransformer transformer = factory.createTransformer(transformerName);
            return transformer.process(input);
        });
    }
    
    /**
     * 添加变换器步骤 / Add Transformer Step
     * 
     * @param transformerName 变换器名称 / Transformer name
     * @param parameters 变换参数 / Transform parameters
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder addTransformer(String transformerName, Map<String, Object> parameters) {
        return addStep((input, context) -> {
            IImageTransformer transformer = factory.createTransformer(transformerName, parameters);
            return transformer.process(input, parameters);
        });
    }
    
    /**
     * 添加分割器步骤 / Add Segmenter Step
     * 
     * @param segmenterName 分割器名称 / Segmenter name
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder addSegmenter(String segmenterName) {
        return addStep((input, context) -> {
            IImageSegmenter segmenter = factory.createSegmenter(segmenterName);
            return segmenter.process(input);
        });
    }
    
    /**
     * 添加分割器步骤 / Add Segmenter Step
     * 
     * @param segmenterName 分割器名称 / Segmenter name
     * @param parameters 分割参数 / Segmentation parameters
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder addSegmenter(String segmenterName, Map<String, Object> parameters) {
        return addStep((input, context) -> {
            IImageSegmenter segmenter = factory.createSegmenter(segmenterName, parameters);
            return segmenter.process(input, parameters);
        });
    }
    
    /**
     * 添加自定义步骤 / Add Custom Step
     * 
     * @param step 自定义步骤 / Custom step
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder addStep(PipelineStep step) {
        this.steps.add(step);
        return this;
    }
    
    /**
     * 添加函数步骤 / Add Function Step
     * 
     * @param function 处理函数 / Processing function
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder addFunction(Function<ImageData, ImageData> function) {
        return addStep((input, context) -> function.apply(input));
    }
    
    /**
     * 添加条件步骤 / Add Conditional Step
     * 
     * @param condition 条件函数 / Condition function
     * @param thenStep 条件为真时的步骤 / Step when condition is true
     * @param elseStep 条件为假时的步骤 / Step when condition is false
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder addConditional(Function<ImageData, Boolean> condition,
                                              PipelineStep thenStep, PipelineStep elseStep) {
        return addStep((input, context) -> {
            if (condition.apply(input)) {
                return thenStep.execute(input, context);
            } else {
                return elseStep.execute(input, context);
            }
        });
    }
    
    /**
     * 添加分支步骤 / Add Branch Step
     * 
     * @param branchName 分支名称 / Branch name
     * @param branch 分支流水线 / Branch pipeline
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder addBranch(String branchName, ImagePipeline branch) {
        return addStep((input, context) -> {
            context.log("Entering branch: " + branchName);
            ImageData result = branch.execute(input, context);
            context.log("Exiting branch: " + branchName);
            return result;
        });
    }
    
    /**
     * 添加循环步骤 / Add Loop Step
     * 
     * @param maxIterations 最大迭代次数 / Maximum iterations
     * @param condition 循环条件 / Loop condition
     * @param loopStep 循环步骤 / Loop step
     * @return 构建器实例 / Builder instance
     */
    public ImagePipelineBuilder addLoop(int maxIterations, Function<ImageData, Boolean> condition, 
                                       PipelineStep loopStep) {
        return addStep((input, context) -> {
            ImageData current = input;
            int iteration = 0;
            
            while (iteration < maxIterations && condition.apply(current)) {
                context.log("Loop iteration: " + (iteration + 1));
                current = loopStep.execute(current, context);
                iteration++;
            }
            
            context.log("Loop completed after " + iteration + " iterations");
            return current;
        });
    }
    
    /**
     * 构建流水线 / Build Pipeline
     * 
     * @return 可执行的流水线 / Executable pipeline
     */
    public ImagePipeline build() {
        ImagePipeline pipeline = new ImagePipeline(pipelineName, steps);
        pipeline.setParallel(enableParallel);
        return pipeline;
    }
    
    /**
     * 构建并执行流水线 / Build and Execute Pipeline
     * 
     * @param input 输入图像 / Input image
     * @return 输出图像 / Output image
     * @throws ImageProcessingException 处理异常 / Processing exception
     */
    public ImageData execute(ImageData input) throws ImageProcessingException {
        ImagePipeline pipeline = build();
        PipelineContext context = new PipelineContext();
        context.setDebugMode(enableDebug);
        return pipeline.execute(input, context);
    }
    
    /**
     * 构建并执行流水线 / Build and Execute Pipeline
     * 
     * @param input 输入图像 / Input image
     * @param parameters 全局参数 / Global parameters
     * @return 输出图像 / Output image
     * @throws ImageProcessingException 处理异常 / Processing exception
     */
    public ImageData execute(ImageData input, Map<String, Object> parameters) throws ImageProcessingException {
        ImagePipeline pipeline = build();
        PipelineContext context = new PipelineContext();
        context.setDebugMode(enableDebug);
        
        // 设置全局参数 / Set global parameters
        for (Map.Entry<String, Object> entry : parameters.entrySet()) {
            context.setParameter(entry.getKey(), entry.getValue());
        }
        
        return pipeline.execute(input, context);
    }
}