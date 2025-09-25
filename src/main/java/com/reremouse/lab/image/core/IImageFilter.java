package com.reremouse.lab.image.core;

import com.reremouse.lab.image.ImageData;
import java.util.Map;

/**
 * 图像滤波器接口 / Image Filter Interface
 * <p>
 * 定义图像滤波操作的统一接口，支持各种类型的图像滤波算法。
 * 继承自IImageProcessor接口，提供滤波特有的功能。
 * </p>
 * <p>
 * Defines unified interface for image filtering operations, supporting various types of filtering algorithms.
 * Extends IImageProcessor interface and provides filter-specific functionality.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public interface IImageFilter extends IImageProcessor {
    
    /**
     * 滤波类型枚举 / Filter Type Enum
     */
    enum FilterType {
        LOW_PASS,               // 低通滤波 / Low-pass filter
        HIGH_PASS,              // 高通滤波 / High-pass filter
        BAND_PASS,              // 带通滤波 / Band-pass filter
        BAND_STOP,              // 带阻滤波 / Band-stop filter
        SPATIAL,                // 空间滤波 / Spatial filter
        FREQUENCY,              // 频域滤波 / Frequency domain filter
        MORPHOLOGICAL,          // 形态学滤波 / Morphological filter
        EDGE_DETECTION,         // 边缘检测 / Edge detection
        NOISE_REDUCTION,        // 噪声抑制 / Noise reduction
        SHARPENING,             // 锐化 / Sharpening
        SMOOTHING,              // 平滑 / Smoothing
        ADAPTIVE,               // 自适应滤波 / Adaptive filtering
        NON_LINEAR,             // 非线性滤波 / Non-linear filter
        STATISTICAL             // 统计滤波 / Statistical filter
    }
    
    /**
     * 滤波结果接口 / Filter Result Interface
     */
    interface FilterResult {
        /**
         * 获取滤波后的图像 / Get Filtered Image
         */
        ImageData getFilteredImage();
        
        /**
         * 获取滤波核 / Get Filter Kernel
         */
        Object getKernel();
        
        /**
         * 获取滤波参数 / Get Filter Parameters
         */
        Map<String, Object> getParameters();
        
        /**
         * 获取滤波质量指标 / Get Filter Quality Metrics
         */
        Map<String, Double> getQualityMetrics();
        
        /**
         * 获取处理时间 / Get Processing Time
         */
        long getProcessingTime();
    }
    
    /**
     * 执行滤波操作 / Execute Filtering Operation
     * <p>
     * 对输入图像执行滤波操作，返回详细的滤波结果。
     * Executes filtering operation on input image and returns detailed filter result.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @return 滤波结果 / Filter result
     * @throws ImageProcessingException 滤波过程中发生错误 / Error occurred during filtering
     */
    FilterResult filter(ImageData input) throws ImageProcessingException;
    
    /**
     * 使用参数执行滤波操作 / Execute Filtering with Parameters
     * <p>
     * 对输入图像执行滤波操作，使用指定的参数。
     * Executes filtering operation on input image with specified parameters.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @param parameters 滤波参数 / Filter parameters
     * @return 滤波结果 / Filter result
     * @throws ImageProcessingException 滤波过程中发生错误 / Error occurred during filtering
     */
    FilterResult filter(ImageData input, Map<String, Object> parameters) throws ImageProcessingException;
    
    /**
     * 级联滤波 / Cascade Filtering
     * <p>
     * 将当前滤波器与其他滤波器级联。
     * Cascades current filter with other filters.
     * </p>
     * 
     * @param other 其他滤波器 / Other filter
     * @return 级联滤波器 / Cascaded filter
     */
    IImageFilter cascade(IImageFilter other);
    
    /**
     * 获取滤波器类型 / Get Filter Type
     * 
     * @return 滤波器类型 / Filter type
     */
    FilterType getFilterType();
    
    /**
     * 获取滤波核大小 / Get Filter Kernel Size
     * 
     * @return 滤波核大小 / Filter kernel size
     */
    int[] getKernelSize();
    
    /**
     * 设置滤波核大小 / Set Filter Kernel Size
     * 
     * @param kernelSize 滤波核大小 / Filter kernel size
     */
    void setKernelSize(int[] kernelSize);
    
    /**
     * 获取边界处理模式 / Get Border Handling Mode
     * 
     * @return 边界处理模式 / Border handling mode
     */
    String getBorderMode();
    
    /**
     * 设置边界处理模式 / Set Border Handling Mode
     * 
     * @param borderMode 边界处理模式 / Border handling mode
     */
    void setBorderMode(String borderMode);
    
    /**
     * 是否支持可分离滤波 / Supports Separable Filtering
     * 
     * @return 是否支持可分离滤波 / Whether separable filtering is supported
     */
    default boolean supportsSeparable() {
        return false;
    }
    
    /**
     * 是否为线性滤波器 / Is Linear Filter
     * 
     * @return 是否为线性滤波器 / Whether it's a linear filter
     */
    default boolean isLinear() {
        return true;
    }
    
    /**
     * 获取频率响应 / Get Frequency Response
     * <p>
     * 获取滤波器的频率响应特性。
     * Gets frequency response characteristics of the filter.
     * </p>
     * 
     * @param frequencies 频率点 / Frequency points
     * @return 频率响应 / Frequency response
     */
    default double[] getFrequencyResponse(double[] frequencies) {
        throw new UnsupportedOperationException("Frequency response not implemented");
    }
    
    /**
     * 预处理图像 / Preprocess Image
     * <p>
     * 在滤波前对图像进行预处理。
     * Preprocesses image before filtering.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @return 预处理后的图像 / Preprocessed image
     */
    default ImageData preprocess(ImageData input) {
        return input;
    }
    
    /**
     * 后处理图像 / Postprocess Image
     * <p>
     * 在滤波后对图像进行后处理。
     * Postprocesses image after filtering.
     * </p>
     * 
     * @param filtered 滤波后的图像 / Filtered image
     * @return 后处理后的图像 / Postprocessed image
     */
    default ImageData postprocess(ImageData filtered) {
        return filtered;
    }
    
    /**
     * 获取滤波器稳定性 / Get Filter Stability
     * 
     * @return 稳定性描述 / Stability description
     */
    default String getStability() {
        return "STABLE";
    }
    
    /**
     * 优化滤波器参数 / Optimize Filter Parameters
     * <p>
     * 根据输入图像特性优化滤波器参数。
     * Optimizes filter parameters based on input image characteristics.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @return 优化后的参数 / Optimized parameters
     */
    default Map<String, Object> optimizeParameters(ImageData input) {
        return getDefaultParameters();
    }
    
    @Override
    default ImageData process(ImageData input) throws ImageProcessingException {
        return filter(input).getFilteredImage();
    }
    
    @Override
    default ImageData process(ImageData input, Map<String, Object> parameters) throws ImageProcessingException {
        return filter(input, parameters).getFilteredImage();
    }
}