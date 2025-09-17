package com.reremouse.lab.math.image;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;

/**
 * 图像滤波类 / Image Filtering Class
 * <p>
 * 提供各种图像滤波功能，包括空间域滤波、频域滤波、边缘检测等。
 * 充分利用现有的信号处理功能和线性代数功能。
 * </p>
 * <p>
 * Provides various image filtering functionality including spatial domain filtering,
 * frequency domain filtering, edge detection, etc. Fully utilizes existing signal
 * processing and linear algebra functionality.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ImageFilter {
    
    /**
     * 滤波类型枚举 / Filter Type Enum
     */
    public enum FilterType {
        GAUSSIAN,          // 高斯滤波 / Gaussian filter
        MEAN,              // 均值滤波 / Mean filter
        MEDIAN,            // 中值滤波 / Median filter
        SOBEL,             // Sobel边缘检测 / Sobel edge detection
        PREWITT,           // Prewitt边缘检测 / Prewitt edge detection
        LAPLACIAN,         // 拉普拉斯滤波 / Laplacian filter
        SHARPEN,           // 锐化滤波 / Sharpening filter
        EMBOSS,            // 浮雕滤波 / Emboss filter
        BOX,               // 盒式滤波 / Box filter
        BILATERAL          // 双边滤波 / Bilateral filter
    }
    
    /**
     * 边缘检测方向枚举 / Edge Detection Direction Enum
     */
    public enum EdgeDirection {
        HORIZONTAL,        // 水平方向 / Horizontal direction
        VERTICAL,          // 垂直方向 / Vertical direction
        DIAGONAL_45,       // 45度对角线 / 45-degree diagonal
        DIAGONAL_135,      // 135度对角线 / 135-degree diagonal
        ALL_DIRECTIONS     // 所有方向 / All directions
    }
    
    /**
     * 高斯滤波 / Gaussian Filter
     * <p>
     * 使用高斯核进行图像平滑，有效去除噪声同时保持边缘。
     * Uses Gaussian kernel for image smoothing, effectively removing noise while preserving edges.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param sigma 高斯核标准差 / Gaussian kernel standard deviation
     * @param kernelSize 核大小（如果为0则自动计算） / Kernel size (auto-calculated if 0)
     * @return 滤波后的图像 / Filtered image
     */
    public static ImageData gaussianFilter(ImageData image, double sigma, int kernelSize) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (sigma <= 0) {
            throw new IllegalArgumentException("标准差必须大于0 / Standard deviation must be greater than 0");
        }
        
        ImageData result = image.copy();
        
        // 对每个通道分别滤波 / Filter each channel separately
        for (int c = 0; c < image.getChannels(); c++) {
            IMatrix<Double> channel = image.getChannel(c);
            IMatrix<Double> filteredChannel = applyGaussianFilter2D(channel, sigma, kernelSize);
            result.setChannel(c, filteredChannel);
        }
        
        return result;
    }
    
    /**
     * 均值滤波 / Mean Filter
     * <p>
     * 使用均值核进行图像平滑，简单但有效的去噪方法。
     * Uses mean kernel for image smoothing, simple but effective denoising method.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param kernelSize 核大小 / Kernel size
     * @return 滤波后的图像 / Filtered image
     */
    public static ImageData meanFilter(ImageData image, int kernelSize) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (kernelSize <= 0 || kernelSize % 2 == 0) {
            throw new IllegalArgumentException("核大小必须为正奇数 / Kernel size must be positive odd number");
        }
        
        ImageData result = image.copy();
        
        // 对每个通道分别滤波 / Filter each channel separately
        for (int c = 0; c < image.getChannels(); c++) {
            IMatrix<Double> channel = image.getChannel(c);
            IMatrix<Double> filteredChannel = applyMeanFilter2D(channel, kernelSize);
            result.setChannel(c, filteredChannel);
        }
        
        return result;
    }
    
    /**
     * 中值滤波 / Median Filter
     * <p>
     * 使用中值核进行图像滤波，有效去除椒盐噪声。
     * Uses median kernel for image filtering, effectively removing salt-and-pepper noise.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param kernelSize 核大小 / Kernel size
     * @return 滤波后的图像 / Filtered image
     */
    public static ImageData medianFilter(ImageData image, int kernelSize) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (kernelSize <= 0 || kernelSize % 2 == 0) {
            throw new IllegalArgumentException("核大小必须为正奇数 / Kernel size must be positive odd number");
        }
        
        ImageData result = image.copy();
        
        // 对每个通道分别滤波 / Filter each channel separately
        for (int c = 0; c < image.getChannels(); c++) {
            IMatrix<Double> channel = image.getChannel(c);
            IMatrix<Double> filteredChannel = applyMedianFilter2D(channel, kernelSize);
            result.setChannel(c, filteredChannel);
        }
        
        return result;
    }
    
    /**
     * Sobel边缘检测 / Sobel Edge Detection
     * <p>
     * 使用Sobel算子检测图像边缘，对噪声具有较好的鲁棒性。
     * Uses Sobel operator to detect image edges with good robustness to noise.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param direction 边缘检测方向 / Edge detection direction
     * @return 边缘检测结果 / Edge detection result
     */
    public static ImageData sobelEdgeDetection(ImageData image, EdgeDirection direction) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        
        IMatrix<Double> edgeImage;
        switch (direction) {
            case HORIZONTAL:
                edgeImage = applySobelHorizontal(channel);
                break;
            case VERTICAL:
                edgeImage = applySobelVertical(channel);
                break;
            case ALL_DIRECTIONS:
                IMatrix<Double> horizontal = applySobelHorizontal(channel);
                IMatrix<Double> vertical = applySobelVertical(channel);
                edgeImage = combineEdgeImages(horizontal, vertical);
                break;
            default:
                throw new UnsupportedOperationException("不支持的边缘检测方向 / Unsupported edge detection direction");
        }
        
        return new ImageData(edgeImage, grayscale.getWidth(), grayscale.getHeight(), 1, 
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * 拉普拉斯滤波 / Laplacian Filter
     * <p>
     * 使用拉普拉斯算子进行边缘检测和图像锐化。
     * Uses Laplacian operator for edge detection and image sharpening.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param sharpen 是否用于锐化 / Whether used for sharpening
     * @return 滤波后的图像 / Filtered image
     */
    public static ImageData laplacianFilter(ImageData image, boolean sharpen) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        ImageData result = image.copy();
        
        // 对每个通道分别滤波 / Filter each channel separately
        for (int c = 0; c < image.getChannels(); c++) {
            IMatrix<Double> channel = image.getChannel(c);
            IMatrix<Double> filteredChannel = applyLaplacianFilter2D(channel, sharpen);
            result.setChannel(c, filteredChannel);
        }
        
        return result;
    }
    
    /**
     * 图像锐化 / Image Sharpening
     * <p>
     * 通过增强高频成分来锐化图像。
     * Sharpens image by enhancing high-frequency components.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param strength 锐化强度 / Sharpening strength
     * @return 锐化后的图像 / Sharpened image
     */
    public static ImageData sharpen(ImageData image, double strength) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (strength < 0) {
            throw new IllegalArgumentException("锐化强度不能为负数 / Sharpening strength cannot be negative");
        }
        
        ImageData result = image.copy();
        
        // 对每个通道分别锐化 / Sharpen each channel separately
        for (int c = 0; c < image.getChannels(); c++) {
            IMatrix<Double> channel = image.getChannel(c);
            IMatrix<Double> sharpenedChannel = applySharpeningFilter2D(channel, strength);
            result.setChannel(c, sharpenedChannel);
        }
        
        return result;
    }
    
    /**
     * 双边滤波 / Bilateral Filter
     * <p>
     * 保持边缘的同时进行平滑滤波，结合空间和像素值相似性。
     * Performs smoothing while preserving edges, combining spatial and pixel value similarity.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param spatialSigma 空间标准差 / Spatial standard deviation
     * @param intensitySigma 强度标准差 / Intensity standard deviation
     * @param kernelSize 核大小 / Kernel size
     * @return 滤波后的图像 / Filtered image
     */
    public static ImageData bilateralFilter(ImageData image, double spatialSigma, double intensitySigma, int kernelSize) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (spatialSigma <= 0 || intensitySigma <= 0) {
            throw new IllegalArgumentException("标准差必须大于0 / Standard deviations must be greater than 0");
        }
        if (kernelSize <= 0 || kernelSize % 2 == 0) {
            throw new IllegalArgumentException("核大小必须为正奇数 / Kernel size must be positive odd number");
        }
        
        ImageData result = image.copy();
        
        // 对每个通道分别滤波 / Filter each channel separately
        for (int c = 0; c < image.getChannels(); c++) {
            IMatrix<Double> channel = image.getChannel(c);
            IMatrix<Double> filteredChannel = applyBilateralFilter2D(channel, spatialSigma, intensitySigma, kernelSize);
            result.setChannel(c, filteredChannel);
        }
        
        return result;
    }
    
    // ========== 辅助方法 / Helper Methods ==========
    
    /**
     * 应用2D高斯滤波 / Apply 2D Gaussian filter
     */
    private static IMatrix<Double> applyGaussianFilter2D(IMatrix<Double> image, double sigma, int kernelSize) {
        // 自动计算核大小 / Auto-calculate kernel size
        if (kernelSize <= 0) {
            kernelSize = (int) (6 * sigma) + 1;
            if (kernelSize % 2 == 0) kernelSize++;
        }
        
        // 创建高斯核 / Create Gaussian kernel
        IMatrix<Double> kernel = createGaussianKernel2D(kernelSize, sigma);
        
        // 应用卷积 / Apply convolution
        return convolve2D(image, kernel);
    }
    
    /**
     * 应用2D均值滤波 / Apply 2D mean filter
     */
    private static IMatrix<Double> applyMeanFilter2D(IMatrix<Double> image, int kernelSize) {
        int height = image.getRowNum();
        int width = image.getColNum();
        int halfKernel = kernelSize / 2;
        
        IMatrix<Double> result = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0.0;
                int count = 0;
                
                for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                    for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                        int ny = y + ky;
                        int nx = x + kx;
                        
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            sum += image.get(ny, nx);
                            count++;
                        }
                    }
                }
                
                result.set(y, x, sum / count);
            }
        }
        
        return result;
    }
    
    /**
     * 应用2D中值滤波 / Apply 2D median filter
     */
    private static IMatrix<Double> applyMedianFilter2D(IMatrix<Double> image, int kernelSize) {
        int height = image.getRowNum();
        int width = image.getColNum();
        int halfKernel = kernelSize / 2;
        
        IMatrix<Double> result = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 收集邻域像素 / Collect neighborhood pixels
                java.util.List<Double> neighborhood = new java.util.ArrayList<>();
                
                for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                    for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                        int ny = y + ky;
                        int nx = x + kx;
                        
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            neighborhood.add(image.get(ny, nx));
                        }
                    }
                }
                
                // 排序并取中值 / Sort and take median
                neighborhood.sort(null);
                int medianIndex = neighborhood.size() / 2;
                result.set(y, x, neighborhood.get(medianIndex));
            }
        }
        
        return result;
    }
    
    /**
     * 应用Sobel水平边缘检测 / Apply Sobel horizontal edge detection
     */
    private static IMatrix<Double> applySobelHorizontal(IMatrix<Double> image) {
        double[][] sobelX = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        
        IMatrix<Double> kernel = Linalg.matrix(sobelX);
        return convolve2D(image, kernel);
    }
    
    /**
     * 应用Sobel垂直边缘检测 / Apply Sobel vertical edge detection
     */
    private static IMatrix<Double> applySobelVertical(IMatrix<Double> image) {
        double[][] sobelY = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };
        
        IMatrix<Double> kernel = Linalg.matrix(sobelY);
        return convolve2D(image, kernel);
    }
    
    /**
     * 应用拉普拉斯滤波 / Apply Laplacian filter
     */
    private static IMatrix<Double> applyLaplacianFilter2D(IMatrix<Double> image, boolean sharpen) {
        double[][] laplacian = {
            { 0, -1,  0},
            {-1,  4, -1},
            { 0, -1,  0}
        };
        
        if (sharpen) {
            // 锐化：原图像 + 拉普拉斯滤波结果 / Sharpening: original + Laplacian result
            IMatrix<Double> kernel = Linalg.matrix(laplacian);
            IMatrix<Double> laplacianResult = convolve2D(image, kernel);
            return image.add(laplacianResult);
        } else {
            // 边缘检测：直接使用拉普拉斯滤波 / Edge detection: use Laplacian filter directly
            IMatrix<Double> kernel = Linalg.matrix(laplacian);
            return convolve2D(image, kernel);
        }
    }
    
    /**
     * 应用锐化滤波 / Apply sharpening filter
     */
    private static IMatrix<Double> applySharpeningFilter2D(IMatrix<Double> image, double strength) {
        double[][] sharpen = {
            { 0, -strength,  0},
            {-strength, 1 + 4*strength, -strength},
            { 0, -strength,  0}
        };
        
        IMatrix<Double> kernel = Linalg.matrix(sharpen);
        return convolve2D(image, kernel);
    }
    
    /**
     * 应用双边滤波 / Apply bilateral filter
     */
    private static IMatrix<Double> applyBilateralFilter2D(IMatrix<Double> image, double spatialSigma, 
                                                        double intensitySigma, int kernelSize) {
        int height = image.getRowNum();
        int width = image.getColNum();
        int halfKernel = kernelSize / 2;
        
        IMatrix<Double> result = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double weightSum = 0.0;
                double valueSum = 0.0;
                double centerValue = image.get(y, x);
                
                for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                    for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                        int ny = y + ky;
                        int nx = x + kx;
                        
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            double neighborValue = image.get(ny, nx);
                            
                            // 空间权重 / Spatial weight
                            double spatialWeight = Math.exp(-(ky*ky + kx*kx) / (2 * spatialSigma * spatialSigma));
                            
                            // 强度权重 / Intensity weight
                            double intensityWeight = Math.exp(-Math.pow(neighborValue - centerValue, 2) / 
                                                            (2 * intensitySigma * intensitySigma));
                            
                            double weight = spatialWeight * intensityWeight;
                            weightSum += weight;
                            valueSum += weight * neighborValue;
                        }
                    }
                }
                
                result.set(y, x, valueSum / weightSum);
            }
        }
        
        return result;
    }
    
    /**
     * 创建2D高斯核 / Create 2D Gaussian kernel
     */
    private static IMatrix<Double> createGaussianKernel2D(int size, double sigma) {
        IMatrix<Double> kernel = Linalg.zeros(size, size);
        int center = size / 2;
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double dx = x - center;
                double dy = y - center;
                double value = Math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma));
                kernel.set(y, x, value);
            }
        }
        
        // 归一化 / Normalize
        double sum = kernel.sum();
        return kernel.multiplyScalar(1.0 / sum);
    }
    
    /**
     * 2D卷积 / 2D Convolution
     */
    private static IMatrix<Double> convolve2D(IMatrix<Double> image, IMatrix<Double> kernel) {
        int imageHeight = image.getRowNum();
        int imageWidth = image.getColNum();
        int kernelHeight = kernel.getRowNum();
        int kernelWidth = kernel.getColNum();
        
        int padHeight = kernelHeight / 2;
        int padWidth = kernelWidth / 2;
        
        IMatrix<Double> result = Linalg.zeros(imageHeight, imageWidth);
        
        for (int y = 0; y < imageHeight; y++) {
            for (int x = 0; x < imageWidth; x++) {
                double sum = 0.0;
                
                for (int ky = 0; ky < kernelHeight; ky++) {
                    for (int kx = 0; kx < kernelWidth; kx++) {
                        int imageY = y + ky - padHeight;
                        int imageX = x + kx - padWidth;
                        
                        if (imageY >= 0 && imageY < imageHeight && imageX >= 0 && imageX < imageWidth) {
                            sum += image.get(imageY, imageX) * kernel.get(ky, kx);
                        }
                    }
                }
                
                result.set(y, x, sum);
            }
        }
        
        return result;
    }
    
    /**
     * 合并边缘图像 / Combine edge images
     */
    private static IMatrix<Double> combineEdgeImages(IMatrix<Double> horizontal, IMatrix<Double> vertical) {
        int height = horizontal.getRowNum();
        int width = horizontal.getColNum();
        
        IMatrix<Double> result = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double h = horizontal.get(y, x);
                double v = vertical.get(y, x);
                double magnitude = Math.sqrt(h*h + v*v);
                result.set(y, x, magnitude);
            }
        }
        
        return result;
    }
}
