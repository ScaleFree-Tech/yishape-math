package com.reremouse.lab.image.core;

import com.reremouse.lab.image.ImageData;
import com.reremouse.lab.math.linalg.IMatrix;
import java.util.Map;

/**
 * 图像分割器接口 / Image Segmenter Interface
 * <p>
 * 定义图像分割操作的统一接口，支持各种类型的图像分割算法。
 * 继承自IImageProcessor接口，提供分割特有的功能。
 * </p>
 * <p>
 * Defines unified interface for image segmentation operations, supporting various types of segmentation algorithms.
 * Extends IImageProcessor interface and provides segmentation-specific functionality.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public interface IImageSegmenter extends IImageProcessor {
    
    /**
     * 分割类型枚举 / Segmentation Type Enum
     */
    enum SegmentationType {
        THRESHOLD,              // 阈值分割 / Threshold segmentation
        REGION_GROWING,         // 区域生长 / Region growing
        WATERSHED,              // 分水岭算法 / Watershed algorithm
        KMEANS,                 // K-means聚类 / K-means clustering
        MEAN_SHIFT,             // Mean Shift算法 / Mean Shift algorithm
        GRAPH_CUT,              // 图割算法 / Graph cut algorithm
        ACTIVE_CONTOUR,         // 活动轮廓 / Active contour
        LEVEL_SET,              // 水平集方法 / Level set method
        SUPERPIXEL,             // 超像素分割 / Superpixel segmentation
        SEMANTIC,               // 语义分割 / Semantic segmentation
        INSTANCE,               // 实例分割 / Instance segmentation
        PANOPTIC,               // 全景分割 / Panoptic segmentation
        EDGE_BASED,             // 基于边缘的分割 / Edge-based segmentation
        TEXTURE_BASED          // 基于纹理的分割 / Texture-based segmentation
    }
    
    /**
     * 分割结果接口 / Segmentation Result Interface
     */
    interface SegmentationResult {
        /**
         * 获取分割后的图像 / Get Segmented Image
         */
        ImageData getSegmentedImage();
        
        /**
         * 获取分割标签图 / Get Segmentation Label Map
         */
        IMatrix<Integer> getLabelMap();
        
        /**
         * 获取区域数量 / Get Number of Regions
         */
        int getNumRegions();
        
        /**
         * 获取区域信息 / Get Region Information
         */
        java.util.List<RegionInfo> getRegions();
        
        /**
         * 获取分割参数 / Get Segmentation Parameters
         */
        Map<String, Object> getParameters();
        
        /**
         * 获取分割质量指标 / Get Segmentation Quality Metrics
         */
        Map<String, Double> getQualityMetrics();
        
        /**
         * 获取置信度图 / Get Confidence Map
         */
        IMatrix<Double> getConfidenceMap();
        
        /**
         * 获取处理时间 / Get Processing Time
         */
        long getProcessingTime();
        
        /**
         * 合并区域 / Merge Regions
         */
        SegmentationResult mergeRegions(int[] regionIds);
        
        /**
         * 分割区域 / Split Region
         */
        SegmentationResult splitRegion(int regionId, Map<String, Object> splitParams);
        
        /**
         * 过滤小区域 / Filter Small Regions
         */
        SegmentationResult filterSmallRegions(int minSize);
    }
    
    /**
     * 区域信息类 / Region Information Class
     */
    interface RegionInfo {
        /**
         * 获取区域ID / Get Region ID
         */
        int getRegionId();
        
        /**
         * 获取区域面积 / Get Region Area
         */
        int getArea();
        
        /**
         * 获取区域周长 / Get Region Perimeter
         */
        double getPerimeter();
        
        /**
         * 获取区域质心 / Get Region Centroid
         */
        double[] getCentroid();
        
        /**
         * 获取边界框 / Get Bounding Box
         */
        int[] getBoundingBox();
        
        /**
         * 获取区域像素坐标 / Get Region Pixel Coordinates
         */
        java.util.List<int[]> getPixelCoordinates();
        
        /**
         * 获取区域统计信息 / Get Region Statistics
         */
        Map<String, Double> getStatistics();
        
        /**
         * 获取区域形状特征 / Get Region Shape Features
         */
        Map<String, Double> getShapeFeatures();
        
        /**
         * 是否为边界区域 / Is Border Region
         */
        boolean isBorderRegion();
    }
    
    /**
     * 执行分割操作 / Execute Segmentation Operation
     * <p>
     * 对输入图像执行分割操作，返回详细的分割结果。
     * Executes segmentation operation on input image and returns detailed segmentation result.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @return 分割结果 / Segmentation result
     * @throws ImageProcessingException 分割过程中发生错误 / Error occurred during segmentation
     */
    SegmentationResult segment(ImageData input) throws ImageProcessingException;
    
    /**
     * 使用参数执行分割操作 / Execute Segmentation with Parameters
     * <p>
     * 对输入图像执行分割操作，使用指定的参数。
     * Executes segmentation operation on input image with specified parameters.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @param parameters 分割参数 / Segmentation parameters
     * @return 分割结果 / Segmentation result
     * @throws ImageProcessingException 分割过程中发生错误 / Error occurred during segmentation
     */
    SegmentationResult segment(ImageData input, Map<String, Object> parameters) throws ImageProcessingException;
    
    /**
     * 交互式分割 / Interactive Segmentation
     * <p>
     * 使用用户提供的种子点或标记进行交互式分割。
     * Performs interactive segmentation using user-provided seed points or markers.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @param seedPoints 种子点 / Seed points
     * @param parameters 分割参数 / Segmentation parameters
     * @return 分割结果 / Segmentation result
     * @throws ImageProcessingException 分割过程中发生错误 / Error occurred during segmentation
     */
    SegmentationResult interactiveSegment(ImageData input, java.util.List<int[]> seedPoints, 
                                        Map<String, Object> parameters) throws ImageProcessingException;
    
    /**
     * 增量分割 / Incremental Segmentation
     * <p>
     * 在现有分割结果基础上进行增量分割。
     * Performs incremental segmentation based on existing segmentation result.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @param existingResult 现有分割结果 / Existing segmentation result
     * @param parameters 分割参数 / Segmentation parameters
     * @return 更新的分割结果 / Updated segmentation result
     * @throws ImageProcessingException 分割过程中发生错误 / Error occurred during segmentation
     */
    SegmentationResult incrementalSegment(ImageData input, SegmentationResult existingResult, 
                                        Map<String, Object> parameters) throws ImageProcessingException;
    
    /**
     * 获取分割类型 / Get Segmentation Type
     * 
     * @return 分割类型 / Segmentation type
     */
    SegmentationType getSegmentationType();
    
    /**
     * 获取最小区域尺寸 / Get Minimum Region Size
     * 
     * @return 最小区域尺寸 / Minimum region size
     */
    int getMinRegionSize();
    
    /**
     * 设置最小区域尺寸 / Set Minimum Region Size
     * 
     * @param minSize 最小区域尺寸 / Minimum region size
     */
    void setMinRegionSize(int minSize);
    
    /**
     * 获取最大区域数量 / Get Maximum Number of Regions
     * 
     * @return 最大区域数量 / Maximum number of regions
     */
    int getMaxNumRegions();
    
    /**
     * 设置最大区域数量 / Set Maximum Number of Regions
     * 
     * @param maxRegions 最大区域数量 / Maximum number of regions
     */
    void setMaxNumRegions(int maxRegions);
    
    /**
     * 是否支持交互式分割 / Supports Interactive Segmentation
     * 
     * @return 是否支持交互式分割 / Whether interactive segmentation is supported
     */
    default boolean supportsInteractive() {
        return false;
    }
    
    /**
     * 是否支持增量分割 / Supports Incremental Segmentation
     * 
     * @return 是否支持增量分割 / Whether incremental segmentation is supported
     */
    default boolean supportsIncremental() {
        return false;
    }
    
    /**
     * 是否支持多尺度分割 / Supports Multi-scale Segmentation
     * 
     * @return 是否支持多尺度分割 / Whether multi-scale segmentation is supported
     */
    default boolean supportsMultiScale() {
        return false;
    }
    
    /**
     * 是否支持3D分割 / Supports 3D Segmentation
     * 
     * @return 是否支持3D分割 / Whether 3D segmentation is supported
     */
    default boolean supports3D() {
        return false;
    }
    
    /**
     * 评估分割质量 / Evaluate Segmentation Quality
     * <p>
     * 使用标准指标评估分割质量。
     * Evaluates segmentation quality using standard metrics.
     * </p>
     * 
     * @param segmentation 分割结果 / Segmentation result
     * @param groundTruth 真实标签 / Ground truth
     * @return 质量指标 / Quality metrics
     */
    default Map<String, Double> evaluateQuality(SegmentationResult segmentation, IMatrix<Integer> groundTruth) {
        Map<String, Double> metrics = new java.util.HashMap<>();
        // 基本实现：计算准确率 / Basic implementation: calculate accuracy
        int correct = 0;
        int total = 0;
        IMatrix<Integer> predicted = segmentation.getLabelMap();
        
        for (int y = 0; y < predicted.getRowNum(); y++) {
            for (int x = 0; x < predicted.getColNum(); x++) {
                if (predicted.get(y, x).equals(groundTruth.get(y, x))) {
                    correct++;
                }
                total++;
            }
        }
        
        metrics.put("accuracy", (double) correct / total);
        return metrics;
    }
    
    /**
     * 可视化分割结果 / Visualize Segmentation Result
     * <p>
     * 创建分割结果的可视化图像。
     * Creates visualization image of segmentation result.
     * </p>
     * 
     * @param result 分割结果 / Segmentation result
     * @param colorMap 颜色映射 / Color map
     * @return 可视化图像 / Visualization image
     */
    default ImageData visualizeSegmentation(SegmentationResult result, Map<Integer, double[]> colorMap) {
        IMatrix<Integer> labelMap = result.getLabelMap();
        int height = labelMap.getRowNum();
        int width = labelMap.getColNum();
        
        // 创建RGB图像 / Create RGB image
        double[][][] rgbData = new double[height][width][3];
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int label = labelMap.get(y, x);
                double[] color = colorMap.getOrDefault(label, new double[]{0.0, 0.0, 0.0});
                rgbData[y][x][0] = color[0];
                rgbData[y][x][1] = color[1];
                rgbData[y][x][2] = color[2];
            }
        }
        
        // 转换为ImageData / Convert to ImageData
        // 此处需要实现具体的转换逻辑 / Implementation of conversion logic needed here
        return null; // 占位符 / Placeholder
    }
    
    @Override
    default ImageData process(ImageData input) throws ImageProcessingException {
        return segment(input).getSegmentedImage();
    }
    
    @Override
    default ImageData process(ImageData input, Map<String, Object> parameters) throws ImageProcessingException {
        return segment(input, parameters).getSegmentedImage();
    }
}