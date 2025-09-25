package com.reremouse.lab.image.core;

import com.reremouse.lab.image.ImageData;
import java.util.Map;

/**
 * 图像处理器接口 / Image Processor Interface
 * <p>
 * 定义图像处理操作的统一接口，支持策略模式和命令模式。
 * 所有图像处理算法都应实现此接口以确保一致的行为和可扩展性。
 * </p>
 * <p>
 * Defines unified interface for image processing operations, supporting Strategy and Command patterns.
 * All image processing algorithms should implement this interface to ensure consistent behavior and extensibility.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public interface IImageProcessor {
    
    /**
     * 处理图像 / Process Image
     * <p>
     * 对输入图像执行特定的处理操作。
     * Executes specific processing operation on input image.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @return 处理后的图像 / Processed image
     * @throws ImageProcessingException 处理过程中发生错误 / Error occurred during processing
     */
    ImageData process(ImageData input) throws ImageProcessingException;
    
    /**
     * 使用参数处理图像 / Process Image with Parameters
     * <p>
     * 对输入图像执行处理操作，使用指定的参数。
     * Executes processing operation on input image with specified parameters.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @param parameters 处理参数 / Processing parameters
     * @return 处理后的图像 / Processed image
     * @throws ImageProcessingException 处理过程中发生错误 / Error occurred during processing
     */
    ImageData process(ImageData input, Map<String, Object> parameters) throws ImageProcessingException;
    
    /**
     * 获取处理器名称 / Get Processor Name
     * 
     * @return 处理器名称 / Processor name
     */
    String getName();
    
    /**
     * 获取处理器描述 / Get Processor Description
     * 
     * @return 处理器描述 / Processor description
     */
    String getDescription();
    
    /**
     * 获取支持的参数 / Get Supported Parameters
     * 
     * @return 支持的参数名称列表 / List of supported parameter names
     */
    java.util.Set<String> getSupportedParameters();
    
    /**
     * 获取默认参数 / Get Default Parameters
     * 
     * @return 默认参数映射 / Default parameter mapping
     */
    Map<String, Object> getDefaultParameters();
    
    /**
     * 验证输入图像 / Validate Input Image
     * <p>
     * 验证输入图像是否符合处理器的要求。
     * Validates if input image meets processor requirements.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @return 验证结果 / Validation result
     */
    boolean validateInput(ImageData input);
    
    /**
     * 验证参数 / Validate Parameters
     * <p>
     * 验证参数是否有效。
     * Validates if parameters are valid.
     * </p>
     * 
     * @param parameters 参数映射 / Parameter mapping
     * @return 验证结果 / Validation result
     */
    boolean validateParameters(Map<String, Object> parameters);
    
    /**
     * 克隆处理器 / Clone Processor
     * <p>
     * 创建处理器的副本，用于并行处理。
     * Creates a copy of processor for parallel processing.
     * </p>
     * 
     * @return 处理器副本 / Processor copy
     */
    IImageProcessor clone();
    
    /**
     * 获取处理器版本 / Get Processor Version
     * 
     * @return 版本字符串 / Version string
     */
    default String getVersion() {
        return "1.0";
    }
    
    /**
     * 是否支持并行处理 / Supports Parallel Processing
     * 
     * @return 是否支持并行处理 / Whether parallel processing is supported
     */
    default boolean supportsParallel() {
        return true;
    }
    
    /**
     * 是否支持GPU加速 / Supports GPU Acceleration
     * 
     * @return 是否支持GPU加速 / Whether GPU acceleration is supported
     */
    default boolean supportsGPU() {
        return false;
    }
    
    /**
     * 获取估计的处理时间复杂度 / Get Estimated Time Complexity
     * 
     * @param imageSize 图像尺寸 / Image size
     * @return 时间复杂度描述 / Time complexity description
     */
    default String getTimeComplexity(int imageSize) {
        return "O(n)";
    }
    
    /**
     * 获取内存使用估计 / Get Memory Usage Estimate
     * 
     * @param imageSize 图像尺寸 / Image size
     * @return 内存使用估计（字节） / Memory usage estimate in bytes
     */
    default long getMemoryUsage(int imageSize) {
        return imageSize * 8L; // Default: 8 bytes per pixel
    }
}