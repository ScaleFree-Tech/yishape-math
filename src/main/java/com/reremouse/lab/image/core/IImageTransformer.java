package com.reremouse.lab.image.core;

import com.reremouse.lab.image.ImageData;
import com.reremouse.lab.math.linalg.IMatrix;
import java.util.Map;

/**
 * 图像变换器接口 / Image Transformer Interface
 * <p>
 * 定义图像几何变换和数学变换操作的统一接口。
 * 支持各种类型的图像变换，包括几何变换、频域变换、小波变换等。
 * </p>
 * <p>
 * Defines unified interface for image geometric and mathematical transformation operations.
 * Supports various types of image transformations including geometric, frequency domain, wavelet transforms, etc.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public interface IImageTransformer extends IImageProcessor {
    
    /**
     * 变换类型枚举 / Transform Type Enum
     */
    enum TransformType {
        GEOMETRIC,              // 几何变换 / Geometric transform
        AFFINE,                 // 仿射变换 / Affine transform
        PERSPECTIVE,            // 透视变换 / Perspective transform
        FOURIER,                // 傅里叶变换 / Fourier transform
        WAVELET,                // 小波变换 / Wavelet transform
        DCT,                    // 离散余弦变换 / Discrete Cosine Transform
        HADAMARD,               // 哈达玛变换 / Hadamard transform
        ROTATION,               // 旋转变换 / Rotation transform
        SCALING,                // 缩放变换 / Scaling transform
        TRANSLATION,            // 平移变换 / Translation transform
        SHEARING,               // 切变变换 / Shearing transform
        REFLECTION,             // 反射变换 / Reflection transform
        POLAR,                  // 极坐标变换 / Polar transform
        LOG_POLAR              // 对数极坐标变换 / Log-polar transform
    }
    
    /**
     * 变换结果接口 / Transform Result Interface
     */
    interface TransformResult {
        /**
         * 获取变换后的图像 / Get Transformed Image
         */
        ImageData getTransformedImage();
        
        /**
         * 获取变换矩阵 / Get Transform Matrix
         */
        IMatrix<Double> getTransformMatrix();
        
        /**
         * 获取逆变换矩阵 / Get Inverse Transform Matrix
         */
        IMatrix<Double> getInverseMatrix();
        
        /**
         * 获取变换参数 / Get Transform Parameters
         */
        Map<String, Object> getParameters();
        
        /**
         * 获取变换质量指标 / Get Transform Quality Metrics
         */
        Map<String, Double> getQualityMetrics();
        
        /**
         * 是否可逆 / Is Invertible
         */
        boolean isInvertible();
        
        /**
         * 获取处理时间 / Get Processing Time
         */
        long getProcessingTime();
    }
    
    /**
     * 执行变换操作 / Execute Transform Operation
     * <p>
     * 对输入图像执行变换操作，返回详细的变换结果。
     * Executes transform operation on input image and returns detailed transform result.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @return 变换结果 / Transform result
     * @throws ImageProcessingException 变换过程中发生错误 / Error occurred during transformation
     */
    TransformResult transform(ImageData input) throws ImageProcessingException;
    
    /**
     * 使用参数执行变换操作 / Execute Transform with Parameters
     * <p>
     * 对输入图像执行变换操作，使用指定的参数。
     * Executes transform operation on input image with specified parameters.
     * </p>
     * 
     * @param input 输入图像 / Input image
     * @param parameters 变换参数 / Transform parameters
     * @return 变换结果 / Transform result
     * @throws ImageProcessingException 变换过程中发生错误 / Error occurred during transformation
     */
    TransformResult transform(ImageData input, Map<String, Object> parameters) throws ImageProcessingException;
    
    /**
     * 执行逆变换操作 / Execute Inverse Transform Operation
     * <p>
     * 对变换结果执行逆变换，恢复原始图像。
     * Executes inverse transform on transform result to restore original image.
     * </p>
     * 
     * @param transformResult 变换结果 / Transform result
     * @return 逆变换后的图像 / Inverse transformed image
     * @throws ImageProcessingException 逆变换过程中发生错误 / Error occurred during inverse transformation
     */
    ImageData inverseTransform(TransformResult transformResult) throws ImageProcessingException;
    
    /**
     * 组合变换 / Compose Transforms
     * <p>
     * 将当前变换与其他变换组合。
     * Composes current transform with other transforms.
     * </p>
     * 
     * @param other 其他变换器 / Other transformer
     * @return 组合变换器 / Composed transformer
     */
    IImageTransformer compose(IImageTransformer other);
    
    /**
     * 获取变换类型 / Get Transform Type
     * 
     * @return 变换类型 / Transform type
     */
    TransformType getTransformType();
    
    /**
     * 是否可逆 / Is Invertible
     * 
     * @return 是否可逆 / Whether invertible
     */
    boolean isInvertible();
    
    /**
     * 是否保持面积 / Preserves Area
     * 
     * @return 是否保持面积 / Whether area is preserved
     */
    default boolean preservesArea() {
        return false;
    }
    
    /**
     * 是否保持角度 / Preserves Angles
     * 
     * @return 是否保持角度 / Whether angles are preserved
     */
    default boolean preservesAngles() {
        return false;
    }
    
    /**
     * 是否保持距离 / Preserves Distances
     * 
     * @return 是否保持距离 / Whether distances are preserved
     */
    default boolean preservesDistances() {
        return false;
    }
    
    /**
     * 获取变换矩阵 / Get Transform Matrix
     * <p>
     * 获取当前变换的数学表示矩阵。
     * Gets mathematical representation matrix of current transform.
     * </p>
     * 
     * @return 变换矩阵 / Transform matrix
     */
    IMatrix<Double> getTransformMatrix();
    
    /**
     * 设置变换矩阵 / Set Transform Matrix
     * <p>
     * 设置变换的数学表示矩阵。
     * Sets mathematical representation matrix of transform.
     * </p>
     * 
     * @param matrix 变换矩阵 / Transform matrix
     */
    void setTransformMatrix(IMatrix<Double> matrix);
    
    /**
     * 获取插值方法 / Get Interpolation Method
     * 
     * @return 插值方法 / Interpolation method
     */
    String getInterpolationMethod();
    
    /**
     * 设置插值方法 / Set Interpolation Method
     * 
     * @param method 插值方法 / Interpolation method
     */
    void setInterpolationMethod(String method);
    
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
     * 计算输出图像尺寸 / Calculate Output Image Size
     * <p>
     * 根据变换参数计算输出图像的尺寸。
     * Calculates output image size based on transform parameters.
     * </p>
     * 
     * @param inputWidth 输入图像宽度 / Input image width
     * @param inputHeight 输入图像高度 / Input image height
     * @return 输出图像尺寸 [宽度, 高度] / Output image size [width, height]
     */
    int[] calculateOutputSize(int inputWidth, int inputHeight);
    
    /**
     * 变换坐标点 / Transform Coordinate Point
     * <p>
     * 将输入坐标点通过变换映射到输出坐标。
     * Maps input coordinate point to output coordinate through transform.
     * </p>
     * 
     * @param x 输入X坐标 / Input X coordinate
     * @param y 输入Y坐标 / Input Y coordinate
     * @return 输出坐标 [X, Y] / Output coordinates [X, Y]
     */
    double[] transformPoint(double x, double y);
    
    /**
     * 逆变换坐标点 / Inverse Transform Coordinate Point
     * <p>
     * 将输出坐标点通过逆变换映射到输入坐标。
     * Maps output coordinate point to input coordinate through inverse transform.
     * </p>
     * 
     * @param x 输出X坐标 / Output X coordinate
     * @param y 输出Y坐标 / Output Y coordinate
     * @return 输入坐标 [X, Y] / Input coordinates [X, Y]
     */
    double[] inverseTransformPoint(double x, double y);
    
    /**
     * 优化变换参数 / Optimize Transform Parameters
     * <p>
     * 根据输入图像特性优化变换参数。
     * Optimizes transform parameters based on input image characteristics.
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
        return transform(input).getTransformedImage();
    }
    
    @Override
    default ImageData process(ImageData input, Map<String, Object> parameters) throws ImageProcessingException {
        return transform(input, parameters).getTransformedImage();
    }
}