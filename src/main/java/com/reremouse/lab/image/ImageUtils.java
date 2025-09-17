package com.reremouse.lab.image;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.Stats;

/**
 * 图像处理工具类 / Image Processing Utilities Class
 * <p>
 * 提供各种图像处理工具函数，包括图像格式转换、尺寸调整、颜色空间转换、图像质量评估等。
 * 充分利用现有的线性代数和统计学功能。
 * </p>
 * <p>
 * Provides various image processing utility functions including format conversion, resizing,
 * color space conversion, image quality assessment, etc. Fully utilizes existing linear algebra and statistical functionality.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ImageUtils {
    
    /**
     * 插值方法枚举 / Interpolation Method Enum
     */
    public enum InterpolationMethod {
        NEAREST_NEIGHBOR,   // 最近邻插值 / Nearest neighbor
        BILINEAR,           // 双线性插值 / Bilinear
        BICUBIC,            // 双三次插值 / Bicubic
        LANCZOS             // Lanczos插值 / Lanczos
    }
    
    /**
     * 颜色空间枚举 / Color Space Enum
     */
    public enum ColorSpace {
        RGB,                // RGB颜色空间 / RGB color space
        HSV,                // HSV颜色空间 / HSV color space
        LAB,                // LAB颜色空间 / LAB color space
        YUV,                // YUV颜色空间 / YUV color space
        GRAYSCALE           // 灰度 / Grayscale
    }
    
    /**
     * 图像质量指标类 / Image Quality Metrics Class
     */
    public static class ImageQualityMetrics {
        private double mse;         // 均方误差 / Mean Squared Error
        private double psnr;        // 峰值信噪比 / Peak Signal-to-Noise Ratio
        private double ssim;        // 结构相似性指数 / Structural Similarity Index
        private double mae;         // 平均绝对误差 / Mean Absolute Error
        private double snr;         // 信噪比 / Signal-to-Noise Ratio
        
        public ImageQualityMetrics(double mse, double psnr, double ssim, double mae, double snr) {
            this.mse = mse;
            this.psnr = psnr;
            this.ssim = ssim;
            this.mae = mae;
            this.snr = snr;
        }
        
        public double getMSE() { return mse; }
        public double getPSNR() { return psnr; }
        public double getSSIM() { return ssim; }
        public double getMAE() { return mae; }
        public double getSNR() { return snr; }
    }
    
    /**
     * 调整图像尺寸 / Resize Image
     * <p>
     * 使用指定的插值方法调整图像尺寸。
     * Resizes image using specified interpolation method.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param newWidth 新宽度 / New width
     * @param newHeight 新高度 / New height
     * @param method 插值方法 / Interpolation method
     * @return 调整后的图像 / Resized image
     */
    public static ImageData resize(ImageData image, int newWidth, int newHeight, InterpolationMethod method) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (newWidth <= 0 || newHeight <= 0) {
            throw new IllegalArgumentException("新尺寸必须大于0 / New dimensions must be greater than 0");
        }
        
        int channels = image.getChannels();
        
        @SuppressWarnings("unchecked")
        IMatrix<Double>[] resizedChannels = new IMatrix[channels];
        
        for (int c = 0; c < channels; c++) {
            IMatrix<Double> channel = image.getChannel(c);
            resizedChannels[c] = resizeChannel(channel, newWidth, newHeight, method);
        }
        
        return new ImageData(resizedChannels, newWidth, newHeight, channels, 
                           image.getImageType(), image.getPixelFormat());
    }
    
    /**
     * 裁剪图像 / Crop Image
     * <p>
     * 从图像中裁剪指定区域。
     * Crops specified region from image.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param x 起始X坐标 / Start X coordinate
     * @param y 起始Y坐标 / Start Y coordinate
     * @param width 裁剪宽度 / Crop width
     * @param height 裁剪高度 / Crop height
     * @return 裁剪后的图像 / Cropped image
     */
    public static ImageData crop(ImageData image, int x, int y, int width, int height) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (x < 0 || y < 0 || width <= 0 || height <= 0) {
            throw new IllegalArgumentException("裁剪参数无效 / Invalid crop parameters");
        }
        if (x + width > image.getWidth() || y + height > image.getHeight()) {
            throw new IllegalArgumentException("裁剪区域超出图像边界 / Crop region exceeds image boundaries");
        }
        
        int channels = image.getChannels();
        @SuppressWarnings("unchecked")
        IMatrix<Double>[] croppedChannels = new IMatrix[channels];
        
        for (int c = 0; c < channels; c++) {
            IMatrix<Double> channel = image.getChannel(c);
            croppedChannels[c] = cropChannel(channel, x, y, width, height);
        }
        
        return new ImageData(croppedChannels, width, height, channels, 
                           image.getImageType(), image.getPixelFormat());
    }
    
    /**
     * 旋转图像 / Rotate Image
     * <p>
     * 按指定角度旋转图像。
     * Rotates image by specified angle.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param angle 旋转角度（弧度） / Rotation angle (radians)
     * @param method 插值方法 / Interpolation method
     * @return 旋转后的图像 / Rotated image
     */
    public static ImageData rotate(ImageData image, double angle, InterpolationMethod method) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        int width = image.getWidth();
        int height = image.getHeight();
        int channels = image.getChannels();
        
        // 计算旋转后的尺寸 / Calculate rotated dimensions
        double cos = Math.cos(angle);
        double sin = Math.sin(angle);
        
        int newWidth = (int) Math.ceil(Math.abs(width * cos) + Math.abs(height * sin));
        int newHeight = (int) Math.ceil(Math.abs(width * sin) + Math.abs(height * cos));
        
        @SuppressWarnings("unchecked")
        IMatrix<Double>[] rotatedChannels = new IMatrix[channels];
        
        for (int c = 0; c < channels; c++) {
            IMatrix<Double> channel = image.getChannel(c);
            rotatedChannels[c] = rotateChannel(channel, angle, newWidth, newHeight, method);
        }
        
        return new ImageData(rotatedChannels, newWidth, newHeight, channels, 
                           image.getImageType(), image.getPixelFormat());
    }
    
    /**
     * 翻转图像 / Flip Image
     * <p>
     * 水平或垂直翻转图像。
     * Flips image horizontally or vertically.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param horizontal 是否水平翻转 / Whether to flip horizontally
     * @param vertical 是否垂直翻转 / Whether to flip vertically
     * @return 翻转后的图像 / Flipped image
     */
    public static ImageData flip(ImageData image, boolean horizontal, boolean vertical) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        int channels = image.getChannels();
        @SuppressWarnings("unchecked")
        IMatrix<Double>[] flippedChannels = new IMatrix[channels];
        
        for (int c = 0; c < channels; c++) {
            IMatrix<Double> channel = image.getChannel(c);
            IMatrix<Double> flipped = channel.copy();
            
            if (horizontal) {
                flipped = flipHorizontal(flipped);
            }
            if (vertical) {
                flipped = flipVertical(flipped);
            }
            
            flippedChannels[c] = flipped;
        }
        
        return new ImageData(flippedChannels, image.getWidth(), image.getHeight(), channels, 
                           image.getImageType(), image.getPixelFormat());
    }
    
    /**
     * 颜色空间转换 / Color Space Conversion
     * <p>
     * 在RGB和其他颜色空间之间转换。
     * Converts between RGB and other color spaces.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param fromSpace 源颜色空间 / Source color space
     * @param toSpace 目标颜色空间 / Target color space
     * @return 转换后的图像 / Converted image
     */
    public static ImageData convertColorSpace(ImageData image, ColorSpace fromSpace, ColorSpace toSpace) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (fromSpace == toSpace) {
            return image.copy();
        }
        
        if (fromSpace == ColorSpace.RGB && toSpace == ColorSpace.HSV) {
            return rgbToHsv(image);
        } else if (fromSpace == ColorSpace.HSV && toSpace == ColorSpace.RGB) {
            return hsvToRgb(image);
        } else if (fromSpace == ColorSpace.RGB && toSpace == ColorSpace.LAB) {
            return rgbToLab(image);
        } else if (fromSpace == ColorSpace.LAB && toSpace == ColorSpace.RGB) {
            return labToRgb(image);
        } else if (fromSpace == ColorSpace.RGB && toSpace == ColorSpace.YUV) {
            return rgbToYuv(image);
        } else if (fromSpace == ColorSpace.YUV && toSpace == ColorSpace.RGB) {
            return yuvToRgb(image);
        } else if (toSpace == ColorSpace.GRAYSCALE) {
            return image.toGrayscale();
        } else {
            throw new IllegalArgumentException("不支持的颜色空间转换 / Unsupported color space conversion");
        }
    }
    
    /**
     * 图像质量评估 / Image Quality Assessment
     * <p>
     * 计算两个图像之间的质量指标。
     * Calculates quality metrics between two images.
     * </p>
     * 
     * @param original 原始图像 / Original image
     * @param processed 处理后的图像 / Processed image
     * @return 质量指标 / Quality metrics
     */
    public static ImageQualityMetrics calculateQualityMetrics(ImageData original, ImageData processed) {
        if (original == null || processed == null) {
            throw new IllegalArgumentException("图像不能为null / Images cannot be null");
        }
        
        // 确保尺寸相同 / Ensure same dimensions
        if (original.getWidth() != processed.getWidth() || original.getHeight() != processed.getHeight()) {
            processed = resize(processed, original.getWidth(), original.getHeight(), InterpolationMethod.BILINEAR);
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData origGray = original.toGrayscale();
        ImageData procGray = processed.toGrayscale();
        
        IMatrix<Double> origChannel = origGray.getChannel(0);
        IMatrix<Double> procChannel = procGray.getChannel(0);
        
        // 计算MSE / Calculate MSE
        IMatrix<Double> diff = origChannel.sub(procChannel);
        IMatrix<Double> squaredDiff = elementWiseMultiply(diff, diff);
        double mse = squaredDiff.mean();
        
        // 计算PSNR / Calculate PSNR
        double maxValue = 1.0; // 假设像素值范围[0,1] / Assume pixel value range [0,1]
        double psnr = 20 * Math.log10(maxValue / Math.sqrt(mse));
        
        // 计算MAE / Calculate MAE
        IMatrix<Double> absDiff = diff.abs();
        double mae = absDiff.mean();
        
        // 计算SNR / Calculate SNR
        double signalPower = elementWiseMultiply(origChannel, origChannel).mean();
        double noisePower = mse;
        double snr = 10 * Math.log10(signalPower / noisePower);
        
        // 计算SSIM / Calculate SSIM
        double ssim = calculateSSIM(origChannel, procChannel);
        
        return new ImageQualityMetrics(mse, psnr, ssim, mae, snr);
    }
    
    /**
     * 图像直方图均衡化 / Histogram Equalization
     * <p>
     * 对图像进行直方图均衡化以增强对比度。
     * Performs histogram equalization on image to enhance contrast.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @return 均衡化后的图像 / Equalized image
     */
    public static ImageData histogramEqualization(ImageData image) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        
        // 计算直方图 / Calculate histogram
        int bins = 256;
        IVector<Double> histogram = Linalg.zeros(bins);
        
        double min = channel.min();
        double max = channel.max();
        double range = max - min;
        
        if (range > 0) {
            for (int y = 0; y < channel.getRowNum(); y++) {
                for (int x = 0; x < channel.getColNum(); x++) {
                    double normalized = (channel.get(y, x) - min) / range;
                    int bin = Math.min((int) (normalized * (bins - 1)), bins - 1);
                    histogram.set(bin, histogram.get(bin) + 1);
                }
            }
        }
        
        // 计算累积分布函数 / Calculate cumulative distribution function
        IVector<Double> cdf = Linalg.zeros(bins);
        cdf.set(0, histogram.get(0));
        for (int i = 1; i < bins; i++) {
            cdf.set(i, cdf.get(i-1) + histogram.get(i));
        }
        
        // 归一化CDF / Normalize CDF
        double totalPixels = cdf.get(bins - 1);
        if (totalPixels > 0) {
            cdf = cdf.multiplyScalar(1.0 / totalPixels);
        }
        
        // 应用均衡化 / Apply equalization
        IMatrix<Double> equalized = channel.copy();
        for (int y = 0; y < channel.getRowNum(); y++) {
            for (int x = 0; x < channel.getColNum(); x++) {
                double normalized = (channel.get(y, x) - min) / range;
                int bin = Math.min((int) (normalized * (bins - 1)), bins - 1);
                double newValue = cdf.get(bin) * range + min;
                equalized.set(y, x, Math.max(0, Math.min(1, newValue)));
            }
        }
        
        return new ImageData(equalized, image.getWidth(), image.getHeight(), 1,
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * 图像噪声添加 / Add Noise to Image
     * <p>
     * 向图像添加指定类型的噪声。
     * Adds specified type of noise to image.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param noiseType 噪声类型 / Noise type
     * @param intensity 噪声强度 / Noise intensity
     * @return 添加噪声后的图像 / Image with noise
     */
    public static ImageData addNoise(ImageData image, String noiseType, double intensity) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        int channels = image.getChannels();
        @SuppressWarnings("unchecked")
        IMatrix<Double>[] noisyChannels = new IMatrix[channels];
        
        for (int c = 0; c < channels; c++) {
            IMatrix<Double> channel = image.getChannel(c);
            IMatrix<Double> noise = generateNoise(channel.getRowNum(), channel.getColNum(), noiseType, intensity);
            noisyChannels[c] = channel.add(noise);
        }
        
        return new ImageData(noisyChannels, image.getWidth(), image.getHeight(), channels, 
                           image.getImageType(), image.getPixelFormat());
    }
    
    // ========== 辅助方法 / Helper Methods ==========
    
    /**
     * 元素级乘法 / Element-wise multiplication
     */
    private static IMatrix<Double> elementWiseMultiply(IMatrix<Double> a, IMatrix<Double> b) {
        if (a.getRowNum() != b.getRowNum() || a.getColNum() != b.getColNum()) {
            throw new IllegalArgumentException("矩阵维度必须相同 / Matrix dimensions must be the same");
        }
        
        IMatrix<Double> result = Linalg.zeros(a.getRowNum(), a.getColNum());
        for (int y = 0; y < a.getRowNum(); y++) {
            for (int x = 0; x < a.getColNum(); x++) {
                result.set(y, x, a.get(y, x) * b.get(y, x));
            }
        }
        return result;
    }
    
    /**
     * 裁剪通道 / Crop channel
     */
    private static IMatrix<Double> cropChannel(IMatrix<Double> channel, int x, int y, int width, int height) {
        IMatrix<Double> cropped = Linalg.zeros(height, width);
        
        for (int dy = 0; dy < height; dy++) {
            for (int dx = 0; dx < width; dx++) {
                cropped.set(dy, dx, channel.get(y + dy, x + dx));
            }
        }
        
        return cropped;
    }
    
    /**
     * 调整通道尺寸 / Resize channel
     */
    private static IMatrix<Double> resizeChannel(IMatrix<Double> channel, int newWidth, int newHeight, InterpolationMethod method) {
        int originalHeight = channel.getRowNum();
        int originalWidth = channel.getColNum();
        
        IMatrix<Double> resized = Linalg.zeros(newHeight, newWidth);
        
        double scaleX = (double) originalWidth / newWidth;
        double scaleY = (double) originalHeight / newHeight;
        
        for (int y = 0; y < newHeight; y++) {
            for (int x = 0; x < newWidth; x++) {
                double srcX = x * scaleX;
                double srcY = y * scaleY;
                
                double value = interpolate(channel, srcX, srcY, method);
                resized.set(y, x, value);
            }
        }
        
        return resized;
    }
    
    /**
     * 插值计算 / Interpolation calculation
     */
    private static double interpolate(IMatrix<Double> channel, double x, double y, InterpolationMethod method) {
        int height = channel.getRowNum();
        int width = channel.getColNum();
        
        switch (method) {
            case NEAREST_NEIGHBOR:
                int nearestX = (int) Math.round(x);
                int nearestY = (int) Math.round(y);
                nearestX = Math.max(0, Math.min(width - 1, nearestX));
                nearestY = Math.max(0, Math.min(height - 1, nearestY));
                return channel.get(nearestY, nearestX);
                
            case BILINEAR:
                int x1 = (int) Math.floor(x);
                int y1 = (int) Math.floor(y);
                int x2 = Math.min(width - 1, x1 + 1);
                int y2 = Math.min(height - 1, y1 + 1);
                
                double fx = x - x1;
                double fy = y - y1;
                
                double v1 = channel.get(y1, x1);
                double v2 = channel.get(y1, x2);
                double v3 = channel.get(y2, x1);
                double v4 = channel.get(y2, x2);
                
                double i1 = v1 * (1 - fx) + v2 * fx;
                double i2 = v3 * (1 - fx) + v4 * fx;
                
                return i1 * (1 - fy) + i2 * fy;
                
            default:
                return interpolate(channel, x, y, InterpolationMethod.BILINEAR);
        }
    }
    
    /**
     * 旋转通道 / Rotate channel
     */
    private static IMatrix<Double> rotateChannel(IMatrix<Double> channel, double angle, int newWidth, int newHeight, InterpolationMethod method) {
        IMatrix<Double> rotated = Linalg.zeros(newHeight, newWidth);
        
        double cos = Math.cos(-angle);
        double sin = Math.sin(-angle);
        double centerX = channel.getColNum() / 2.0;
        double centerY = channel.getRowNum() / 2.0;
        double newCenterX = newWidth / 2.0;
        double newCenterY = newHeight / 2.0;
        
        for (int y = 0; y < newHeight; y++) {
            for (int x = 0; x < newWidth; x++) {
                double srcX = (x - newCenterX) * cos - (y - newCenterY) * sin + centerX;
                double srcY = (x - newCenterX) * sin + (y - newCenterY) * cos + centerY;
                
                if (srcX >= 0 && srcX < channel.getColNum() && srcY >= 0 && srcY < channel.getRowNum()) {
                    double value = interpolate(channel, srcX, srcY, method);
                    rotated.set(y, x, value);
                }
            }
        }
        
        return rotated;
    }
    
    /**
     * 水平翻转 / Flip horizontally
     */
    private static IMatrix<Double> flipHorizontal(IMatrix<Double> channel) {
        int height = channel.getRowNum();
        int width = channel.getColNum();
        IMatrix<Double> flipped = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                flipped.set(y, x, channel.get(y, width - 1 - x));
            }
        }
        
        return flipped;
    }
    
    /**
     * 垂直翻转 / Flip vertically
     */
    private static IMatrix<Double> flipVertical(IMatrix<Double> channel) {
        int height = channel.getRowNum();
        int width = channel.getColNum();
        IMatrix<Double> flipped = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                flipped.set(y, x, channel.get(height - 1 - y, x));
            }
        }
        
        return flipped;
    }
    
    /**
     * RGB到HSV转换 / RGB to HSV conversion
     */
    private static ImageData rgbToHsv(ImageData image) {
        if (image.getChannels() != 3) {
            throw new IllegalArgumentException("RGB图像必须有3个通道 / RGB image must have 3 channels");
        }
        
        IMatrix<Double> r = image.getChannel(0);
        IMatrix<Double> g = image.getChannel(1);
        IMatrix<Double> b = image.getChannel(2);
        
        int height = r.getRowNum();
        int width = r.getColNum();
        
        IMatrix<Double> h = Linalg.zeros(height, width);
        IMatrix<Double> s = Linalg.zeros(height, width);
        IMatrix<Double> v = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double rVal = r.get(y, x);
                double gVal = g.get(y, x);
                double bVal = b.get(y, x);
                
                double max = Math.max(Math.max(rVal, gVal), bVal);
                double min = Math.min(Math.min(rVal, gVal), bVal);
                double delta = max - min;
                
                // V / V
                v.set(y, x, max);
                
                // S / S
                if (max > 0) {
                    s.set(y, x, delta / max);
                } else {
                    s.set(y, x, 0.0);
                }
                
                // H / H
                if (delta > 0) {
                    if (max == rVal) {
                        h.set(y, x, ((gVal - bVal) / delta) % 6);
                    } else if (max == gVal) {
                        h.set(y, x, (bVal - rVal) / delta + 2);
                    } else {
                        h.set(y, x, (rVal - gVal) / delta + 4);
                    }
                    h.set(y, x, h.get(y, x) / 6.0);
                } else {
                    h.set(y, x, 0.0);
                }
            }
        }
        
        @SuppressWarnings("unchecked")
        IMatrix<Double>[] hsvChannels = new IMatrix[]{h, s, v};
        return new ImageData(hsvChannels, image.getWidth(), image.getHeight(), 3, 
                           ImageData.ImageType.RGB, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * HSV到RGB转换 / HSV to RGB conversion
     */
    private static ImageData hsvToRgb(ImageData image) {
        if (image.getChannels() != 3) {
            throw new IllegalArgumentException("HSV图像必须有3个通道 / HSV image must have 3 channels");
        }
        
        IMatrix<Double> h = image.getChannel(0);
        IMatrix<Double> s = image.getChannel(1);
        IMatrix<Double> v = image.getChannel(2);
        
        int height = h.getRowNum();
        int width = h.getColNum();
        
        IMatrix<Double> r = Linalg.zeros(height, width);
        IMatrix<Double> g = Linalg.zeros(height, width);
        IMatrix<Double> b = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double hVal = h.get(y, x) * 6.0;
                double sVal = s.get(y, x);
                double vVal = v.get(y, x);
                
                int i = (int) Math.floor(hVal);
                double f = hVal - i;
                double p = vVal * (1 - sVal);
                double q = vVal * (1 - sVal * f);
                double t = vVal * (1 - sVal * (1 - f));
                
                double rVal, gVal, bVal;
                switch (i % 6) {
                    case 0: rVal = vVal; gVal = t; bVal = p; break;
                    case 1: rVal = q; gVal = vVal; bVal = p; break;
                    case 2: rVal = p; gVal = vVal; bVal = t; break;
                    case 3: rVal = p; gVal = q; bVal = vVal; break;
                    case 4: rVal = t; gVal = p; bVal = vVal; break;
                    default: rVal = vVal; gVal = p; bVal = q; break;
                }
                
                r.set(y, x, rVal);
                g.set(y, x, gVal);
                b.set(y, x, bVal);
            }
        }
        
        @SuppressWarnings("unchecked")
        IMatrix<Double>[] rgbChannels = new IMatrix[]{r, g, b};
        return new ImageData(rgbChannels, image.getWidth(), image.getHeight(), 3, 
                           ImageData.ImageType.RGB, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * RGB到LAB转换 / RGB to LAB conversion
     */
    private static ImageData rgbToLab(ImageData image) {
        // 简化的RGB到LAB转换 / Simplified RGB to LAB conversion
        // 这里使用线性近似，实际应用中需要更精确的转换 / Uses linear approximation here, more precise conversion needed in practice
        return image.copy(); // 占位符实现 / Placeholder implementation
    }
    
    /**
     * LAB到RGB转换 / LAB to RGB conversion
     */
    private static ImageData labToRgb(ImageData image) {
        // 简化的LAB到RGB转换 / Simplified LAB to RGB conversion
        return image.copy(); // 占位符实现 / Placeholder implementation
    }
    
    /**
     * RGB到YUV转换 / RGB to YUV conversion
     */
    private static ImageData rgbToYuv(ImageData image) {
        if (image.getChannels() != 3) {
            throw new IllegalArgumentException("RGB图像必须有3个通道 / RGB image must have 3 channels");
        }
        
        IMatrix<Double> r = image.getChannel(0);
        IMatrix<Double> g = image.getChannel(1);
        IMatrix<Double> b = image.getChannel(2);
        
        int height = r.getRowNum();
        int width = r.getColNum();
        
        IMatrix<Double> y = Linalg.zeros(height, width);
        IMatrix<Double> u = Linalg.zeros(height, width);
        IMatrix<Double> v = Linalg.zeros(height, width);
        
        for (int yIdx = 0; yIdx < height; yIdx++) {
            for (int x = 0; x < width; x++) {
                double rVal = r.get(yIdx, x);
                double gVal = g.get(yIdx, x);
                double bVal = b.get(yIdx, x);
                
                y.set(yIdx, x, 0.299 * rVal + 0.587 * gVal + 0.114 * bVal);
                u.set(yIdx, x, -0.147 * rVal - 0.289 * gVal + 0.436 * bVal);
                v.set(yIdx, x, 0.615 * rVal - 0.515 * gVal - 0.100 * bVal);
            }
        }
        
        @SuppressWarnings("unchecked")
        IMatrix<Double>[] yuvChannels = new IMatrix[]{y, u, v};
        return new ImageData(yuvChannels, image.getWidth(), image.getHeight(), 3, 
                           ImageData.ImageType.RGB, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * YUV到RGB转换 / YUV to RGB conversion
     */
    private static ImageData yuvToRgb(ImageData image) {
        if (image.getChannels() != 3) {
            throw new IllegalArgumentException("YUV图像必须有3个通道 / YUV image must have 3 channels");
        }
        
        IMatrix<Double> y = image.getChannel(0);
        IMatrix<Double> u = image.getChannel(1);
        IMatrix<Double> v = image.getChannel(2);
        
        int height = y.getRowNum();
        int width = y.getColNum();
        
        IMatrix<Double> r = Linalg.zeros(height, width);
        IMatrix<Double> g = Linalg.zeros(height, width);
        IMatrix<Double> b = Linalg.zeros(height, width);
        
        for (int yIdx = 0; yIdx < height; yIdx++) {
            for (int x = 0; x < width; x++) {
                double yVal = y.get(yIdx, x);
                double uVal = u.get(yIdx, x);
                double vVal = v.get(yIdx, x);
                
                r.set(yIdx, x, yVal + 1.140 * vVal);
                g.set(yIdx, x, yVal - 0.394 * uVal - 0.581 * vVal);
                b.set(yIdx, x, yVal + 2.032 * uVal);
            }
        }
        
        @SuppressWarnings("unchecked")
        IMatrix<Double>[] rgbChannels = new IMatrix[]{r, g, b};
        return new ImageData(rgbChannels, image.getWidth(), image.getHeight(), 3, 
                           ImageData.ImageType.RGB, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * 计算SSIM / Calculate SSIM
     */
    private static double calculateSSIM(IMatrix<Double> img1, IMatrix<Double> img2) {
        // 简化的SSIM计算 / Simplified SSIM calculation
        double mu1 = img1.mean();
        double mu2 = img2.mean();
        double sigma1 = img1.std();
        double sigma2 = img2.std();
        
        IMatrix<Double> diff = img1.sub(img2);
        double sigma12 = diff.mean();
        
        double c1 = 0.01;
        double c2 = 0.03;
        
        double ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / 
                     ((mu1 * mu1 + mu2 * mu2 + c1) * (sigma1 * sigma1 + sigma2 * sigma2 + c2));
        
        return ssim;
    }
    
    /**
     * 生成噪声 / Generate noise
     */
    private static IMatrix<Double> generateNoise(int height, int width, String noiseType, double intensity) {
        IMatrix<Double> noise = Linalg.zeros(height, width);
        
        switch (noiseType.toLowerCase()) {
            case "gaussian":
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        noise.set(y, x, Stats.norm(0, intensity).sample());
                    }
                }
                break;
                
            case "uniform":
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        noise.set(y, x, Stats.uniform(-intensity, intensity).sample());
                    }
                }
                break;
                
            case "salt_pepper":
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        double rand = Math.random();
                        if (rand < intensity / 2) {
                            noise.set(y, x, -intensity);
                        } else if (rand < intensity) {
                            noise.set(y, x, intensity);
                        } else {
                            noise.set(y, x, 0.0);
                        }
                    }
                }
                break;
                
            default:
                throw new IllegalArgumentException("不支持的噪声类型 / Unsupported noise type: " + noiseType);
        }
        
        return noise;
    }
}
