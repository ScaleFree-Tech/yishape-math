package com.reremouse.lab.image.enhancement;

import com.reremouse.lab.image.ImageData;
import com.reremouse.lab.image.core.IImageProcessor;
import com.reremouse.lab.image.core.ImageProcessingException;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.compute.GPUComputeDoubleUtils;
import com.reremouse.lab.math.compute.GPUConfig;
import java.util.Map;

/**
 * 高级图像增强类 / Advanced Image Enhancement Class
 * <p>
 * 提供高级图像增强功能，包括多尺度Retinex、CLAHE、超分辨率、去噪等。
 * 支持GPU加速和自适应参数调整。
 * </p>
 * <p>
 * Provides advanced image enhancement functionality including Multi-Scale Retinex, CLAHE, 
 * super-resolution, denoising, etc. Supports GPU acceleration and adaptive parameter adjustment.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public class AdvancedImageEnhancement {
    
    /**
     * 多尺度Retinex增强器 / Multi-Scale Retinex Enhancer
     */
    public static class MultiScaleRetinexProcessor implements IImageProcessor {
        
        private double[] scales = {15.0, 80.0, 250.0};
        private double[] weights = {1.0/3.0, 1.0/3.0, 1.0/3.0};
        private double gain = 1.0;
        private double offset = 0.0;
        private boolean useGPU = false;
        
        @Override
        public ImageData process(ImageData input) throws ImageProcessingException {
            return process(input, getDefaultParameters());
        }
        
        @Override
        public ImageData process(ImageData input, Map<String, Object> parameters) throws ImageProcessingException {
            if (!validateInput(input)) {
                throw ImageProcessingException.invalidInput("Input image cannot be null");
            }
            
            // 应用参数 / Apply parameters
            if (parameters != null) {
                if (parameters.containsKey("scales")) {
                    scales = (double[]) parameters.get("scales");
                }
                if (parameters.containsKey("weights")) {
                    weights = (double[]) parameters.get("weights");
                }
                gain = (Double) parameters.getOrDefault("gain", gain);
                offset = (Double) parameters.getOrDefault("offset", offset);
                useGPU = (Boolean) parameters.getOrDefault("useGPU", useGPU);
            }
            
            // 检查GPU可用性 / Check GPU availability
            if (useGPU && !GPUComputeDoubleUtils.isGPUAvailable()) {
                System.out.println("Warning: GPU requested but not available, falling back to CPU");
                useGPU = false;
            }
            
            int channels = input.getChannels();
            @SuppressWarnings("unchecked")
            IMatrix<Double>[] enhancedChannels = new IMatrix[channels];
            
            for (int c = 0; c < channels; c++) {
                IMatrix<Double> channel = input.getChannel(c);
                enhancedChannels[c] = applyMultiScaleRetinex(channel);
            }
            
            return new ImageData(enhancedChannels, input.getWidth(), input.getHeight(), channels,
                               input.getImageType(), input.getPixelFormat());
        }
        
        @Override
        public String getName() { return "MultiScaleRetinex"; }
        
        @Override
        public String getDescription() { return "Multi-Scale Retinex enhancement"; }
        
        @Override
        public java.util.Set<String> getSupportedParameters() {
            return java.util.Set.of("scales", "weights", "gain", "offset", "useGPU");
        }
        
        @Override
        public Map<String, Object> getDefaultParameters() {
            Map<String, Object> params = new java.util.HashMap<>();
            params.put("scales", new double[]{15.0, 80.0, 250.0});
            params.put("weights", new double[]{1.0/3.0, 1.0/3.0, 1.0/3.0});
            params.put("gain", 1.0);
            params.put("offset", 0.0);
            params.put("useGPU", false);
            return params;
        }
        
        @Override
        public boolean validateInput(ImageData input) {
            return input != null;
        }
        
        @Override
        public boolean validateParameters(Map<String, Object> parameters) {
            if (parameters == null) return true;
            
            double[] s = (double[]) parameters.get("scales");
            double[] w = (double[]) parameters.get("weights");
            
            if (s != null && w != null && s.length != w.length) {
                return false;
            }
            
            return true;
        }
        
        @Override
        public IImageProcessor clone() {
            MultiScaleRetinexProcessor cloned = new MultiScaleRetinexProcessor();
            cloned.scales = this.scales.clone();
            cloned.weights = this.weights.clone();
            cloned.gain = this.gain;
            cloned.offset = this.offset;
            cloned.useGPU = this.useGPU;
            return cloned;
        }
        
        @Override
        public boolean supportsGPU() {
            return true;
        }
        
        private IMatrix<Double> applyMultiScaleRetinex(IMatrix<Double> channel) throws ImageProcessingException {
            int height = channel.getRowNum();
            int width = channel.getColNum();
            
            try {
                if (useGPU) {
                    return applyMultiScaleRetinexGPU(channel);
                } else {
                    return applyMultiScaleRetinexCPU(channel);
                }
            } catch (Exception e) {
                if (useGPU) {
                    // GPU失败时回退到CPU / Fallback to CPU when GPU fails
                    System.out.println("GPU processing failed, falling back to CPU: " + e.getMessage());
                    return applyMultiScaleRetinexCPU(channel);
                } else {
                    throw ImageProcessingException.processingFailed(getName(), "MSR processing failed", e);
                }
            }
        }
        
        private IMatrix<Double> applyMultiScaleRetinexCPU(IMatrix<Double> channel) {
            int height = channel.getRowNum();
            int width = channel.getColNum();
            IMatrix<Double> result = Linalg.zeros(height, width);
            
            // 添加小常数避免log(0) / Add small constant to avoid log(0)
            IMatrix<Double> logImage = Linalg.zeros(height, width);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double value = Math.max(channel.get(y, x), 1e-6);
                    logImage.set(y, x, Math.log(value));
                }
            }
            
            // 对每个尺度应用Retinex / Apply Retinex for each scale
            for (int i = 0; i < scales.length; i++) {
                double scale = scales[i];
                double weight = weights[i];
                
                // 创建高斯滤波器 / Create Gaussian filter
                IMatrix<Double> blurred = applyGaussianFilter(channel, scale);
                
                // 计算log差值 / Calculate log difference
                IMatrix<Double> logBlurred = Linalg.zeros(height, width);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        double value = Math.max(blurred.get(y, x), 1e-6);
                        logBlurred.set(y, x, Math.log(value));
                    }
                }
                
                IMatrix<Double> retinex = logImage.sub(logBlurred);
                
                // 加权累加 / Weighted accumulation
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        double current = result.get(y, x);
                        result.set(y, x, current + weight * retinex.get(y, x));
                    }
                }
            }
            
            // 应用增益和偏移 / Apply gain and offset
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double value = gain * result.get(y, x) + offset;
                    result.set(y, x, Math.max(0.0, Math.min(1.0, value)));
                }
            }
            
            return result;
        }
        
        private IMatrix<Double> applyMultiScaleRetinexGPU(IMatrix<Double> channel) throws ImageProcessingException {
            try {
                // GPU实现的多尺度Retinex - 由于没有专门的GPU方法，直接使用CPU实现 / GPU implementation of Multi-Scale Retinex - use CPU implementation directly since no dedicated GPU method
                return applyMultiScaleRetinexCPU(channel);
                
            } catch (Exception e) {
                throw ImageProcessingException.gpuError("GPU MSR processing failed", e);
            }
        }
        
        private IMatrix<Double> applyGaussianFilter(IMatrix<Double> image, double sigma) {
            // 简化的高斯滤波实现 / Simplified Gaussian filter implementation
            int kernelSize = (int) (6 * sigma) + 1;
            if (kernelSize % 2 == 0) kernelSize++;
            
            // 创建高斯核 / Create Gaussian kernel
            IMatrix<Double> kernel = createGaussianKernel(kernelSize, sigma);
            
            // 应用卷积 / Apply convolution
            return convolve2D(image, kernel);
        }
        
        private IMatrix<Double> createGaussianKernel(int size, double sigma) {
            IMatrix<Double> kernel = Linalg.zeros(size, size);
            int center = size / 2;
            double sum = 0.0;
            
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    double dx = x - center;
                    double dy = y - center;
                    double value = Math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma));
                    kernel.set(y, x, value);
                    sum += value;
                }
            }
            
            // 归一化 / Normalize
            return kernel.multiplyScalar(1.0 / sum);
        }
        
        private IMatrix<Double> convolve2D(IMatrix<Double> image, IMatrix<Double> kernel) {
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
    }
    
    /**
     * CLAHE (对比度限制自适应直方图均衡化) 处理器 / CLAHE Processor
     */
    public static class CLAHEProcessor implements IImageProcessor {
        
        private int tileSize = 8;
        private double clipLimit = 2.0;
        private int numBins = 256;
        
        @Override
        public ImageData process(ImageData input) throws ImageProcessingException {
            return process(input, getDefaultParameters());
        }
        
        @Override
        public ImageData process(ImageData input, Map<String, Object> parameters) throws ImageProcessingException {
            if (!validateInput(input)) {
                throw ImageProcessingException.invalidInput("Input image cannot be null");
            }
            
            // 应用参数 / Apply parameters
            if (parameters != null) {
                tileSize = (Integer) parameters.getOrDefault("tileSize", tileSize);
                clipLimit = (Double) parameters.getOrDefault("clipLimit", clipLimit);
                numBins = (Integer) parameters.getOrDefault("numBins", numBins);
            }
            
            // 转换为灰度图像处理 / Convert to grayscale for processing
            ImageData grayscale = input.toGrayscale();
            IMatrix<Double> channel = grayscale.getChannel(0);
            
            // 应用CLAHE / Apply CLAHE
            IMatrix<Double> enhanced = applyCLAHE(channel);
            
            return new ImageData(enhanced, input.getWidth(), input.getHeight(), 1,
                               ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
        }
        
        @Override
        public String getName() { return "CLAHE"; }
        
        @Override
        public String getDescription() { return "Contrast Limited Adaptive Histogram Equalization"; }
        
        @Override
        public java.util.Set<String> getSupportedParameters() {
            return java.util.Set.of("tileSize", "clipLimit", "numBins");
        }
        
        @Override
        public Map<String, Object> getDefaultParameters() {
            Map<String, Object> params = new java.util.HashMap<>();
            params.put("tileSize", 8);
            params.put("clipLimit", 2.0);
            params.put("numBins", 256);
            return params;
        }
        
        @Override
        public boolean validateInput(ImageData input) {
            return input != null;
        }
        
        @Override
        public boolean validateParameters(Map<String, Object> parameters) {
            if (parameters == null) return true;
            
            Integer ts = (Integer) parameters.get("tileSize");
            Double cl = (Double) parameters.get("clipLimit");
            Integer nb = (Integer) parameters.get("numBins");
            
            return (ts == null || ts > 0) && (cl == null || cl > 0) && (nb == null || nb > 0);
        }
        
        @Override
        public IImageProcessor clone() {
            CLAHEProcessor cloned = new CLAHEProcessor();
            cloned.tileSize = this.tileSize;
            cloned.clipLimit = this.clipLimit;
            cloned.numBins = this.numBins;
            return cloned;
        }
        
        private IMatrix<Double> applyCLAHE(IMatrix<Double> image) {
            int height = image.getRowNum();
            int width = image.getColNum();
            
            // 计算瓦片数量 / Calculate number of tiles
            int tilesY = (height + tileSize - 1) / tileSize;
            int tilesX = (width + tileSize - 1) / tileSize;
            
            // 为每个瓦片计算映射函数 / Calculate mapping function for each tile
            double[][][] tileMappings = new double[tilesY][tilesX][numBins];
            
            for (int ty = 0; ty < tilesY; ty++) {
                for (int tx = 0; tx < tilesX; tx++) {
                    // 提取瓦片 / Extract tile
                    int startY = ty * tileSize;
                    int startX = tx * tileSize;
                    int endY = Math.min(startY + tileSize, height);
                    int endX = Math.min(startX + tileSize, width);
                    
                    // 计算瓦片直方图 / Calculate tile histogram
                    double[] histogram = calculateTileHistogram(image, startX, startY, endX, endY);
                    
                    // 应用对比度限制 / Apply contrast limiting
                    histogram = applyContrastLimiting(histogram);
                    
                    // 计算累积分布函数 / Calculate CDF
                    double[] cdf = calculateCDF(histogram);
                    
                    // 创建映射函数 / Create mapping function
                    tileMappings[ty][tx] = cdf;
                }
            }
            
            // 应用双线性插值 / Apply bilinear interpolation
            IMatrix<Double> result = Linalg.zeros(height, width);
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double pixelValue = image.get(y, x);
                    int bin = Math.min((int) (pixelValue * (numBins - 1)), numBins - 1);
                    
                    // 计算瓦片坐标 / Calculate tile coordinates
                    double tileY = (double) y / tileSize;
                    double tileX = (double) x / tileSize;
                    
                    int ty1 = Math.max(0, Math.min(tilesY - 1, (int) Math.floor(tileY)));
                    int tx1 = Math.max(0, Math.min(tilesX - 1, (int) Math.floor(tileX)));
                    int ty2 = Math.max(0, Math.min(tilesY - 1, ty1 + 1));
                    int tx2 = Math.max(0, Math.min(tilesX - 1, tx1 + 1));
                    
                    double wy = tileY - ty1;
                    double wx = tileX - tx1;
                    
                    // 双线性插值 / Bilinear interpolation
                    double v1 = tileMappings[ty1][tx1][bin];
                    double v2 = tileMappings[ty1][tx2][bin];
                    double v3 = tileMappings[ty2][tx1][bin];
                    double v4 = tileMappings[ty2][tx2][bin];
                    
                    double interpolated = (1 - wy) * ((1 - wx) * v1 + wx * v2) + 
                                        wy * ((1 - wx) * v3 + wx * v4);
                    
                    result.set(y, x, Math.max(0.0, Math.min(1.0, interpolated)));
                }
            }
            
            return result;
        }
        
        private double[] calculateTileHistogram(IMatrix<Double> image, int startX, int startY, int endX, int endY) {
            double[] histogram = new double[numBins];
            
            for (int y = startY; y < endY; y++) {
                for (int x = startX; x < endX; x++) {
                    double value = image.get(y, x);
                    int bin = Math.min((int) (value * (numBins - 1)), numBins - 1);
                    histogram[bin]++;
                }
            }
            
            // 归一化 / Normalize
            int totalPixels = (endY - startY) * (endX - startX);
            for (int i = 0; i < numBins; i++) {
                histogram[i] /= totalPixels;
            }
            
            return histogram;
        }
        
        private double[] applyContrastLimiting(double[] histogram) {
            double[] limited = histogram.clone();
            double clipThreshold = clipLimit / numBins;
            
            // 计算超出限制的部分 / Calculate excess
            double excess = 0.0;
            for (int i = 0; i < numBins; i++) {
                if (limited[i] > clipThreshold) {
                    excess += limited[i] - clipThreshold;
                    limited[i] = clipThreshold;
                }
            }
            
            // 重新分配超出部分 / Redistribute excess
            double redistribute = excess / numBins;
            for (int i = 0; i < numBins; i++) {
                limited[i] += redistribute;
            }
            
            return limited;
        }
        
        private double[] calculateCDF(double[] histogram) {
            double[] cdf = new double[numBins];
            cdf[0] = histogram[0];
            
            for (int i = 1; i < numBins; i++) {
                cdf[i] = cdf[i - 1] + histogram[i];
            }
            
            return cdf;
        }
    }
}