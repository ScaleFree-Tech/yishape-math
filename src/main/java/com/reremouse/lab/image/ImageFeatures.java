package com.reremouse.lab.image;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;

/**
 * 图像特征提取类 / Image Feature Extraction Class
 * <p>
 * 提供各种图像特征提取功能，包括统计特征、纹理特征、形状特征等。
 * 充分利用现有的统计学功能和线性代数功能。
 * </p>
 * <p>
 * Provides various image feature extraction functionality including statistical features,
 * texture features, shape features, etc. Fully utilizes existing statistical and linear algebra functionality.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ImageFeatures {
    
    /**
     * 特征类型枚举 / Feature Type Enum
     */
    public enum FeatureType {
        STATISTICAL,        // 统计特征 / Statistical features
        TEXTURE,           // 纹理特征 / Texture features
        SHAPE,             // 形状特征 / Shape features
        COLOR,             // 颜色特征 / Color features
        FREQUENCY,         // 频域特征 / Frequency domain features
        WAVELET,           // 小波特征 / Wavelet features
        GRADIENT,          // 梯度特征 / Gradient features
        HISTOGRAM          // 直方图特征 / Histogram features
    }
    
    /**
     * 纹理特征结果类 / Texture Feature Result Class
     */
    public static class TextureFeatures {
        public double energy;           // 能量 / Energy
        public double contrast;         // 对比度 / Contrast
        public double correlation;      // 相关性 / Correlation
        public double homogeneity;      // 同质性 / Homogeneity
        public double entropy;          // 熵 / Entropy
        public double variance;         // 方差 / Variance
        public double mean;             // 均值 / Mean
        public double standardDeviation; // 标准差 / Standard deviation
        
        public TextureFeatures(double energy, double contrast, double correlation, double homogeneity,
                              double entropy, double variance, double mean, double standardDeviation) {
            this.energy = energy;
            this.contrast = contrast;
            this.correlation = correlation;
            this.homogeneity = homogeneity;
            this.entropy = entropy;
            this.variance = variance;
            this.mean = mean;
            this.standardDeviation = standardDeviation;
        }
        
        @Override
        public String toString() {
            return String.format("TextureFeatures[energy=%.3f, contrast=%.3f, correlation=%.3f, homogeneity=%.3f, entropy=%.3f, variance=%.3f, mean=%.3f, std=%.3f]",
                               energy, contrast, correlation, homogeneity, entropy, variance, mean, standardDeviation);
        }
    }
    
    /**
     * 形状特征结果类 / Shape Feature Result Class
     */
    public static class ShapeFeatures {
        public double area;             // 面积 / Area
        public double perimeter;        // 周长 / Perimeter
        public double compactness;      // 紧致度 / Compactness
        public double aspectRatio;      // 长宽比 / Aspect ratio
        public double circularity;      // 圆形度 / Circularity
        public double eccentricity;     // 偏心率 / Eccentricity
        public double solidity;         // 实心度 / Solidity
        public double extent;           // 范围 / Extent
        
        public ShapeFeatures(double area, double perimeter, double compactness, double aspectRatio,
                           double circularity, double eccentricity, double solidity, double extent) {
            this.area = area;
            this.perimeter = perimeter;
            this.compactness = compactness;
            this.aspectRatio = aspectRatio;
            this.circularity = circularity;
            this.eccentricity = eccentricity;
            this.solidity = solidity;
            this.extent = extent;
        }
        
        @Override
        public String toString() {
            return String.format("ShapeFeatures[area=%.3f, perimeter=%.3f, compactness=%.3f, aspectRatio=%.3f, circularity=%.3f, eccentricity=%.3f, solidity=%.3f, extent=%.3f]",
                               area, perimeter, compactness, aspectRatio, circularity, eccentricity, solidity, extent);
        }
    }
    
    /**
     * 颜色特征结果类 / Color Feature Result Class
     */
    public static class ColorFeatures {
        public double[] meanRGB;        // RGB均值 / RGB mean
        public double[] stdRGB;         // RGB标准差 / RGB standard deviation
        public double[] skewnessRGB;    // RGB偏度 / RGB skewness
        public double[] kurtosisRGB;    // RGB峰度 / RGB kurtosis
        public double meanHue;          // 色调均值 / Hue mean
        public double meanSaturation;   // 饱和度均值 / Saturation mean
        public double meanValue;        // 明度均值 / Value mean
        public double colorVariance;    // 颜色方差 / Color variance
        
        public ColorFeatures(double[] meanRGB, double[] stdRGB, double[] skewnessRGB, double[] kurtosisRGB,
                           double meanHue, double meanSaturation, double meanValue, double colorVariance) {
            this.meanRGB = meanRGB;
            this.stdRGB = stdRGB;
            this.skewnessRGB = skewnessRGB;
            this.kurtosisRGB = kurtosisRGB;
            this.meanHue = meanHue;
            this.meanSaturation = meanSaturation;
            this.meanValue = meanValue;
            this.colorVariance = colorVariance;
        }
        
        @Override
        public String toString() {
            return String.format("ColorFeatures[meanRGB=[%.3f,%.3f,%.3f], stdRGB=[%.3f,%.3f,%.3f], meanHSV=[%.3f,%.3f,%.3f], colorVar=%.3f]",
                               meanRGB[0], meanRGB[1], meanRGB[2], stdRGB[0], stdRGB[1], stdRGB[2],
                               meanHue, meanSaturation, meanValue, colorVariance);
        }
    }
    
    /**
     * 提取统计特征 / Extract Statistical Features
     * <p>
     * 提取图像的基本统计特征，包括均值、方差、偏度、峰度等。
     * Extracts basic statistical features of image including mean, variance, skewness, kurtosis, etc.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @return 统计特征向量 / Statistical feature vector
     */
    public static IVector<Double> extractStatisticalFeatures(ImageData image) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        java.util.List<Double> features = new java.util.ArrayList<>();
        
        // 对每个通道提取统计特征 / Extract statistical features for each channel
        for (int c = 0; c < image.getChannels(); c++) {
            IMatrix<Double> channel = image.getChannel(c);
            IVector<Double> channelVector = channel.flatten();
            
            // 基本统计量 / Basic statistics
            features.add(channelVector.mean());
            features.add(channelVector.std());
            features.add(channelVector.min());
            features.add(channelVector.max());
            features.add(channelVector.median());
            features.add(channelVector.var());
            features.add(channelVector.skewness());
            features.add(channelVector.kurtosis());
        }
        
        // 整体图像统计特征 / Overall image statistical features
        IVector<Double> allPixels = image.getData().flatten();
        features.add(allPixels.mean());
        features.add(allPixels.std());
        features.add(allPixels.min());
        features.add(allPixels.max());
        features.add(allPixels.median());
        features.add(allPixels.var());
        features.add(allPixels.skewness());
        features.add(allPixels.kurtosis());
        
        // 转换为向量 / Convert to vector
        double[] featureArray = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            featureArray[i] = features.get(i);
        }
        
        return Linalg.vector(featureArray);
    }
    
    /**
     * 提取纹理特征 / Extract Texture Features
     * <p>
     * 使用灰度共生矩阵(GLCM)提取纹理特征。
     * Extracts texture features using Gray-Level Co-occurrence Matrix (GLCM).
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param distance 距离参数 / Distance parameter
     * @param angle 角度参数 / Angle parameter
     * @return 纹理特征 / Texture features
     */
    public static TextureFeatures extractTextureFeatures(ImageData image, int distance, double angle) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        
        // 量化到较少的灰度级 / Quantize to fewer gray levels
        IMatrix<Double> quantized = quantizeImage(channel, 32);
        
        // 计算灰度共生矩阵 / Calculate GLCM
        IMatrix<Double> glcm = calculateGLCM(quantized, distance, angle);
        
        // 归一化GLCM / Normalize GLCM
        double sum = glcm.sum();
        if (sum > 0) {
            glcm = glcm.multiplyScalar(1.0 / sum);
        }
        
        // 计算纹理特征 / Calculate texture features
        double energy = calculateEnergy(glcm);
        double contrast = calculateContrast(glcm);
        double correlation = calculateCorrelation(glcm);
        double homogeneity = calculateHomogeneity(glcm);
        double entropy = calculateEntropy(glcm);
        
        // 计算基本统计特征 / Calculate basic statistical features
        IVector<Double> channelVector = channel.flatten();
        double variance = channelVector.var();
        double mean = channelVector.mean();
        double standardDeviation = channelVector.std();
        
        return new TextureFeatures(energy, contrast, correlation, homogeneity, entropy, 
                                 variance, mean, standardDeviation);
    }
    
    /**
     * 提取形状特征 / Extract Shape Features
     * <p>
     * 从二值图像中提取形状特征。
     * Extracts shape features from binary image.
     * </p>
     * 
     * @param binaryImage 二值图像 / Binary image
     * @return 形状特征 / Shape features
     */
    public static ShapeFeatures extractShapeFeatures(ImageData binaryImage) {
        if (binaryImage == null) {
            throw new IllegalArgumentException("二值图像不能为null / Binary image cannot be null");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = binaryImage.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        
        // 二值化 / Binarize
        IMatrix<Double> binary = binarizeImage(channel, 0.5);
        
        // 计算基本形状特征 / Calculate basic shape features
        double area = calculateArea(binary);
        double perimeter = calculatePerimeter(binary);
        double compactness = calculateCompactness(area, perimeter);
        double aspectRatio = calculateAspectRatio(binary);
        double circularity = calculateCircularity(area, perimeter);
        double eccentricity = calculateEccentricity(binary);
        double solidity = calculateSolidity(binary, area);
        double extent = calculateExtent(binary, area);
        
        return new ShapeFeatures(area, perimeter, compactness, aspectRatio, circularity, 
                               eccentricity, solidity, extent);
    }
    
    /**
     * 提取颜色特征 / Extract Color Features
     * <p>
     * 从彩色图像中提取颜色特征。
     * Extracts color features from color image.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @return 颜色特征 / Color features
     */
    public static ColorFeatures extractColorFeatures(ImageData image) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        if (image.getChannels() < 3) {
            throw new IllegalArgumentException("需要彩色图像 / Color image required");
        }
        
        // 提取RGB通道 / Extract RGB channels
        IMatrix<Double> rChannel = image.getChannel(0);
        IMatrix<Double> gChannel = image.getChannel(1);
        IMatrix<Double> bChannel = image.getChannel(2);
        
        // 计算RGB统计特征 / Calculate RGB statistical features
        double[] meanRGB = new double[3];
        double[] stdRGB = new double[3];
        double[] skewnessRGB = new double[3];
        double[] kurtosisRGB = new double[3];
        
        IVector<Double> rVector = rChannel.flatten();
        IVector<Double> gVector = gChannel.flatten();
        IVector<Double> bVector = bChannel.flatten();
        
        meanRGB[0] = rVector.mean();
        meanRGB[1] = gVector.mean();
        meanRGB[2] = bVector.mean();
        
        stdRGB[0] = rVector.std();
        stdRGB[1] = gVector.std();
        stdRGB[2] = bVector.std();
        
        skewnessRGB[0] = rVector.skewness();
        skewnessRGB[1] = gVector.skewness();
        skewnessRGB[2] = bVector.skewness();
        
        kurtosisRGB[0] = rVector.kurtosis();
        kurtosisRGB[1] = gVector.kurtosis();
        kurtosisRGB[2] = bVector.kurtosis();
        
        // 转换为HSV并计算特征 / Convert to HSV and calculate features
        double meanHue = 0.0, meanSaturation = 0.0, meanValue = 0.0;
        double colorVariance = 0.0;
        
        if (image.getChannels() >= 3) {
            // 简化的HSV转换 / Simplified HSV conversion
            int pixelCount = 0;
            double hueSum = 0.0, saturationSum = 0.0, valueSum = 0.0;
            
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    double r = rChannel.get(y, x);
                    double g = gChannel.get(y, x);
                    double b = bChannel.get(y, x);
                    
                    double[] hsv = rgbToHsv(r, g, b);
                    hueSum += hsv[0];
                    saturationSum += hsv[1];
                    valueSum += hsv[2];
                    pixelCount++;
                }
            }
            
            meanHue = hueSum / pixelCount;
            meanSaturation = saturationSum / pixelCount;
            meanValue = valueSum / pixelCount;
            
            // 计算颜色方差 / Calculate color variance
            double colorSum = 0.0;
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    double r = rChannel.get(y, x);
                    double g = gChannel.get(y, x);
                    double b = bChannel.get(y, x);
                    double colorDistance = Math.sqrt(r*r + g*g + b*b);
                    colorSum += Math.pow(colorDistance - Math.sqrt(meanRGB[0]*meanRGB[0] + meanRGB[1]*meanRGB[1] + meanRGB[2]*meanRGB[2]), 2);
                }
            }
            colorVariance = colorSum / pixelCount;
        }
        
        return new ColorFeatures(meanRGB, stdRGB, skewnessRGB, kurtosisRGB, 
                               meanHue, meanSaturation, meanValue, colorVariance);
    }
    
    /**
     * 提取直方图特征 / Extract Histogram Features
     * <p>
     * 从图像直方图中提取特征。
     * Extracts features from image histogram.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param bins 直方图bin数 / Number of histogram bins
     * @return 直方图特征向量 / Histogram feature vector
     */
    public static IVector<Double> extractHistogramFeatures(ImageData image, int bins) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (bins <= 0) {
            throw new IllegalArgumentException("bin数必须大于0 / Number of bins must be greater than 0");
        }
        
        java.util.List<Double> features = new java.util.ArrayList<>();
        
        // 对每个通道计算直方图特征 / Calculate histogram features for each channel
        for (int c = 0; c < image.getChannels(); c++) {
            IMatrix<Double> channel = image.getChannel(c);
            IVector<Double> histogram = calculateHistogram(channel, bins);
            
            // 直方图统计特征 / Histogram statistical features
            features.add(histogram.mean());
            features.add(histogram.std());
            features.add(histogram.max());
            features.add(histogram.min());
            features.add(histogram.median());
            features.add(histogram.var());
            features.add(histogram.skewness());
            features.add(histogram.kurtosis());
            
            // 直方图形状特征 / Histogram shape features
            features.add(calculateHistogramEntropy(histogram));
            features.add(calculateHistogramEnergy(histogram));
            features.add(calculateHistogramUniformity(histogram));
        }
        
        // 转换为向量 / Convert to vector
        double[] featureArray = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            featureArray[i] = features.get(i);
        }
        
        return Linalg.vector(featureArray);
    }
    
    /**
     * 提取梯度特征 / Extract Gradient Features
     * <p>
     * 从图像梯度中提取特征。
     * Extracts features from image gradients.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @return 梯度特征向量 / Gradient feature vector
     */
    public static IVector<Double> extractGradientFeatures(ImageData image) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        
        // 计算梯度 / Calculate gradients
        IMatrix<Double> gradientX = calculateGradientX(channel);
        IMatrix<Double> gradientY = calculateGradientY(channel);
        IMatrix<Double> gradientMagnitude = calculateGradientMagnitude(gradientX, gradientY);
        IMatrix<Double> gradientDirection = calculateGradientDirection(gradientX, gradientY);
        
        java.util.List<Double> features = new java.util.ArrayList<>();
        
        // 梯度幅值特征 / Gradient magnitude features
        IVector<Double> magnitudeVector = gradientMagnitude.flatten();
        features.add(magnitudeVector.mean());
        features.add(magnitudeVector.std());
        features.add(magnitudeVector.max());
        features.add(magnitudeVector.min());
        features.add(magnitudeVector.median());
        features.add(magnitudeVector.var());
        features.add(magnitudeVector.skewness());
        features.add(magnitudeVector.kurtosis());
        
        // 梯度方向特征 / Gradient direction features
        IVector<Double> directionVector = gradientDirection.flatten();
        features.add(directionVector.mean());
        features.add(directionVector.std());
        features.add(directionVector.var());
        
        // 转换为向量 / Convert to vector
        double[] featureArray = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            featureArray[i] = features.get(i);
        }
        
        return Linalg.vector(featureArray);
    }
    
    // ========== 辅助方法 / Helper Methods ==========
    
    /**
     * 量化图像 / Quantize image
     */
    private static IMatrix<Double> quantizeImage(IMatrix<Double> image, int levels) {
        IMatrix<Double> quantized = image.copy();
        double min = image.min();
        double max = image.max();
        double range = max - min;
        
        if (range > 0) {
            for (int y = 0; y < image.getRowNum(); y++) {
                for (int x = 0; x < image.getColNum(); x++) {
                    double normalized = (image.get(y, x) - min) / range;
                    int quantizedValue = (int) (normalized * (levels - 1));
                    quantized.set(y, x, (double) quantizedValue);
                }
            }
        }
        
        return quantized;
    }
    
    /**
     * 计算灰度共生矩阵 / Calculate GLCM
     */
    private static IMatrix<Double> calculateGLCM(IMatrix<Double> image, int distance, double angle) {
        int levels = (int) (image.max() + 1);
        IMatrix<Double> glcm = Linalg.zeros(levels, levels);
        
        int height = image.getRowNum();
        int width = image.getColNum();
        
        // 计算偏移量 / Calculate offset
        int dx = (int) (distance * Math.cos(angle));
        int dy = (int) (distance * Math.sin(angle));
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int ny = y + dy;
                int nx = x + dx;
                
                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    int i = image.get(y, x).intValue();
                    int j = image.get(ny, nx).intValue();
                    glcm.set(i, j, glcm.get(i, j) + 1);
                }
            }
        }
        
        return glcm;
    }
    
    /**
     * 计算能量 / Calculate energy
     */
    private static double calculateEnergy(IMatrix<Double> glcm) {
        double energy = 0.0;
        for (int i = 0; i < glcm.getRowNum(); i++) {
            for (int j = 0; j < glcm.getColNum(); j++) {
                double value = glcm.get(i, j);
                energy += value * value;
            }
        }
        return energy;
    }
    
    /**
     * 计算对比度 / Calculate contrast
     */
    private static double calculateContrast(IMatrix<Double> glcm) {
        double contrast = 0.0;
        for (int i = 0; i < glcm.getRowNum(); i++) {
            for (int j = 0; j < glcm.getColNum(); j++) {
                double value = glcm.get(i, j);
                contrast += value * (i - j) * (i - j);
            }
        }
        return contrast;
    }
    
    /**
     * 计算相关性 / Calculate correlation
     */
    private static double calculateCorrelation(IMatrix<Double> glcm) {
        // 计算均值和标准差 / Calculate mean and standard deviation
        double meanI = 0.0, meanJ = 0.0;
        double stdI = 0.0, stdJ = 0.0;
        
        for (int i = 0; i < glcm.getRowNum(); i++) {
            for (int j = 0; j < glcm.getColNum(); j++) {
                double value = glcm.get(i, j);
                meanI += i * value;
                meanJ += j * value;
            }
        }
        
        for (int i = 0; i < glcm.getRowNum(); i++) {
            for (int j = 0; j < glcm.getColNum(); j++) {
                double value = glcm.get(i, j);
                stdI += (i - meanI) * (i - meanI) * value;
                stdJ += (j - meanJ) * (j - meanJ) * value;
            }
        }
        
        stdI = Math.sqrt(stdI);
        stdJ = Math.sqrt(stdJ);
        
        if (stdI == 0 || stdJ == 0) return 0.0;
        
        double correlation = 0.0;
        for (int i = 0; i < glcm.getRowNum(); i++) {
            for (int j = 0; j < glcm.getColNum(); j++) {
                double value = glcm.get(i, j);
                correlation += value * (i - meanI) * (j - meanJ) / (stdI * stdJ);
            }
        }
        
        return correlation;
    }
    
    /**
     * 计算同质性 / Calculate homogeneity
     */
    private static double calculateHomogeneity(IMatrix<Double> glcm) {
        double homogeneity = 0.0;
        for (int i = 0; i < glcm.getRowNum(); i++) {
            for (int j = 0; j < glcm.getColNum(); j++) {
                double value = glcm.get(i, j);
                homogeneity += value / (1 + Math.abs(i - j));
            }
        }
        return homogeneity;
    }
    
    /**
     * 计算熵 / Calculate entropy
     */
    private static double calculateEntropy(IMatrix<Double> glcm) {
        double entropy = 0.0;
        for (int i = 0; i < glcm.getRowNum(); i++) {
            for (int j = 0; j < glcm.getColNum(); j++) {
                double value = glcm.get(i, j);
                if (value > 0) {
                    entropy -= value * Math.log(value) / Math.log(2);
                }
            }
        }
        return entropy;
    }
    
    /**
     * 二值化图像 / Binarize image
     */
    private static IMatrix<Double> binarizeImage(IMatrix<Double> image, double threshold) {
        IMatrix<Double> binary = image.copy();
        for (int y = 0; y < image.getRowNum(); y++) {
            for (int x = 0; x < image.getColNum(); x++) {
                binary.set(y, x, image.get(y, x) > threshold ? 1.0 : 0.0);
            }
        }
        return binary;
    }
    
    /**
     * 计算面积 / Calculate area
     */
    private static double calculateArea(IMatrix<Double> binary) {
        double area = 0.0;
        for (int y = 0; y < binary.getRowNum(); y++) {
            for (int x = 0; x < binary.getColNum(); x++) {
                if (binary.get(y, x) > 0.5) {
                    area += 1.0;
                }
            }
        }
        return area;
    }
    
    /**
     * 计算周长 / Calculate perimeter
     */
    private static double calculatePerimeter(IMatrix<Double> binary) {
        double perimeter = 0.0;
        int height = binary.getRowNum();
        int width = binary.getColNum();
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (binary.get(y, x) > 0.5) {
                    // 检查邻域 / Check neighborhood
                    int neighbors = 0;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0) continue;
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                if (binary.get(ny, nx) > 0.5) {
                                    neighbors++;
                                }
                            }
                        }
                    }
                    if (neighbors < 8) {
                        perimeter += 1.0;
                    }
                }
            }
        }
        return perimeter;
    }
    
    /**
     * 计算紧致度 / Calculate compactness
     */
    private static double calculateCompactness(double area, double perimeter) {
        if (perimeter == 0) return 0.0;
        return (4 * Math.PI * area) / (perimeter * perimeter);
    }
    
    /**
     * 计算长宽比 / Calculate aspect ratio
     */
    private static double calculateAspectRatio(IMatrix<Double> binary) {
        // 简化的长宽比计算 / Simplified aspect ratio calculation
        int height = binary.getRowNum();
        int width = binary.getColNum();
        return (double) width / height;
    }
    
    /**
     * 计算圆形度 / Calculate circularity
     */
    private static double calculateCircularity(double area, double perimeter) {
        if (perimeter == 0) return 0.0;
        return (4 * Math.PI * area) / (perimeter * perimeter);
    }
    
    /**
     * 计算偏心率 / Calculate eccentricity
     */
    private static double calculateEccentricity(IMatrix<Double> binary) {
        // 简化的偏心率计算 / Simplified eccentricity calculation
        return 0.5; // 占位符 / Placeholder
    }
    
    /**
     * 计算实心度 / Calculate solidity
     */
    private static double calculateSolidity(IMatrix<Double> binary, double area) {
        if (area == 0) return 0.0;
        // 简化的实心度计算 / Simplified solidity calculation
        return 1.0; // 占位符 / Placeholder
    }
    
    /**
     * 计算范围 / Calculate extent
     */
    private static double calculateExtent(IMatrix<Double> binary, double area) {
        if (area == 0) return 0.0;
        int height = binary.getRowNum();
        int width = binary.getColNum();
        return area / (height * width);
    }
    
    /**
     * RGB转HSV / RGB to HSV
     */
    private static double[] rgbToHsv(double r, double g, double b) {
        double max = Math.max(r, Math.max(g, b));
        double min = Math.min(r, Math.min(g, b));
        double delta = max - min;
        
        double h = 0.0;
        if (delta != 0) {
            if (max == r) {
                h = 60 * (((g - b) / delta) % 6);
            } else if (max == g) {
                h = 60 * ((b - r) / delta + 2);
            } else {
                h = 60 * ((r - g) / delta + 4);
            }
        }
        if (h < 0) h += 360;
        
        double s = (max == 0) ? 0 : delta / max;
        double v = max;
        
        return new double[]{h, s, v};
    }
    
    /**
     * 计算直方图 / Calculate histogram
     */
    private static IVector<Double> calculateHistogram(IMatrix<Double> image, int bins) {
        IVector<Double> histogram = Linalg.zeros(bins);
        double min = image.min();
        double max = image.max();
        double range = max - min;
        
        if (range > 0) {
            for (int y = 0; y < image.getRowNum(); y++) {
                for (int x = 0; x < image.getColNum(); x++) {
                    double normalized = (image.get(y, x) - min) / range;
                    int bin = Math.min((int) (normalized * bins), bins - 1);
                    histogram.set(bin, histogram.get(bin) + 1);
                }
            }
        }
        
        return histogram;
    }
    
    /**
     * 计算直方图熵 / Calculate histogram entropy
     */
    private static double calculateHistogramEntropy(IVector<Double> histogram) {
        double sum = histogram.sum();
        if (sum == 0) return 0.0;
        
        double entropy = 0.0;
        for (int i = 0; i < histogram.length(); i++) {
            double p = histogram.get(i) / sum;
            if (p > 0) {
                entropy -= p * Math.log(p) / Math.log(2);
            }
        }
        return entropy;
    }
    
    /**
     * 计算直方图能量 / Calculate histogram energy
     */
    private static double calculateHistogramEnergy(IVector<Double> histogram) {
        double sum = histogram.sum();
        if (sum == 0) return 0.0;
        
        double energy = 0.0;
        for (int i = 0; i < histogram.length(); i++) {
            double p = histogram.get(i) / sum;
            energy += p * p;
        }
        return energy;
    }
    
    /**
     * 计算直方图均匀性 / Calculate histogram uniformity
     */
    private static double calculateHistogramUniformity(IVector<Double> histogram) {
        double sum = histogram.sum();
        if (sum == 0) return 0.0;
        
        double uniformity = 0.0;
        for (int i = 0; i < histogram.length(); i++) {
            double p = histogram.get(i) / sum;
            uniformity += p * p;
        }
        return uniformity;
    }
    
    /**
     * 计算X方向梯度 / Calculate X-direction gradient
     */
    private static IMatrix<Double> calculateGradientX(IMatrix<Double> image) {
        int height = image.getRowNum();
        int width = image.getColNum();
        IMatrix<Double> gradientX = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 1; x < width - 1; x++) {
                double gx = (image.get(y, x + 1) - image.get(y, x - 1)) / 2.0;
                gradientX.set(y, x, gx);
            }
        }
        
        return gradientX;
    }
    
    /**
     * 计算Y方向梯度 / Calculate Y-direction gradient
     */
    private static IMatrix<Double> calculateGradientY(IMatrix<Double> image) {
        int height = image.getRowNum();
        int width = image.getColNum();
        IMatrix<Double> gradientY = Linalg.zeros(height, width);
        
        for (int y = 1; y < height - 1; y++) {
            for (int x = 0; x < width; x++) {
                double gy = (image.get(y + 1, x) - image.get(y - 1, x)) / 2.0;
                gradientY.set(y, x, gy);
            }
        }
        
        return gradientY;
    }
    
    /**
     * 计算梯度幅值 / Calculate gradient magnitude
     */
    private static IMatrix<Double> calculateGradientMagnitude(IMatrix<Double> gradientX, IMatrix<Double> gradientY) {
        int height = gradientX.getRowNum();
        int width = gradientX.getColNum();
        IMatrix<Double> magnitude = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double gx = gradientX.get(y, x);
                double gy = gradientY.get(y, x);
                double mag = Math.sqrt(gx * gx + gy * gy);
                magnitude.set(y, x, mag);
            }
        }
        
        return magnitude;
    }
    
    /**
     * 计算梯度方向 / Calculate gradient direction
     */
    private static IMatrix<Double> calculateGradientDirection(IMatrix<Double> gradientX, IMatrix<Double> gradientY) {
        int height = gradientX.getRowNum();
        int width = gradientX.getColNum();
        IMatrix<Double> direction = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double gx = gradientX.get(y, x);
                double gy = gradientY.get(y, x);
                double dir = Math.atan2(gy, gx);
                direction.set(y, x, dir);
            }
        }
        
        return direction;
    }
}
