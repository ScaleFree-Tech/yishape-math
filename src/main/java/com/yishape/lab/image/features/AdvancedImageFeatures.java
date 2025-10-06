package com.yishape.lab.image.features;

import com.yishape.lab.image.ImageData;
import com.yishape.lab.image.core.IImageAnalyzer;
import com.yishape.lab.image.core.ImageProcessingException;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;

import java.util.Map;

/**
 * 高级图像特征提取类 / Advanced Image Feature Extraction Class
 * <p>
 * 提供高级图像特征提取功能，包括LBP、HOG、GLCM、Gabor滤波器组等。
 * 支持多尺度特征提取和特征融合。
 * </p>
 * <p>
 * Provides advanced image feature extraction functionality including LBP, HOG, GLCM, Gabor filter banks, etc.
 * Supports multi-scale feature extraction and feature fusion.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public class AdvancedImageFeatures {
    
    /**
     * LBP特征提取器 / LBP Feature Extractor
     */
    public static class LBPAnalyzer implements IImageAnalyzer {
        
        private int radius = 1;
        private int neighbors = 8;
        private boolean uniform = true;
        
        /**
         * LBP特征结果 / LBP Feature Result
         */
        public static class LBPResult implements AnalysisResult {
            private IVector<Double> histogram;
            private IMatrix<Double> lbpImage;
            private double uniformity;
            private Map<String, Double> statistics;
            
            public LBPResult(IVector<Double> histogram, IMatrix<Double> lbpImage, 
                           double uniformity, Map<String, Double> statistics) {
                this.histogram = histogram;
                this.lbpImage = lbpImage;
                this.uniformity = uniformity;
                this.statistics = statistics;
            }
            
            @Override
            public String getAnalysisType() { return "LBP"; }
            
            @Override
            public IVector<Double> getFeatureVector() { return histogram; }
            
            @Override
            public Map<String, Double> getNumericResults() { return statistics; }
            
            @Override
            public String getDescription() { 
                return String.format("LBP features with radius=%d, neighbors=%d, uniformity=%.3f", 
                                   1, 8, uniformity); 
            }
            
            @Override
            public double getConfidence() { return uniformity; }
            
            @Override
            public String toJSON() {
                return String.format("{\"type\":\"LBP\",\"uniformity\":%.3f,\"histogram_size\":%d}", 
                                   uniformity, histogram.length());
            }
            
            public IMatrix<Double> getLBPImage() { return lbpImage; }
            public double getUniformity() { return uniformity; }
        }
        
        @Override
        public AnalysisResult analyze(ImageData input) throws ImageProcessingException {
            return analyze(input, getDefaultParameters());
        }
        
        @Override
        public AnalysisResult analyze(ImageData input, Map<String, Object> parameters) throws ImageProcessingException {
            if (!validateInput(input)) {
                throw ImageProcessingException.invalidInput("Input must be grayscale image");
            }
            
            // 应用参数 / Apply parameters
            if (parameters != null) {
                radius = (Integer) parameters.getOrDefault("radius", radius);
                neighbors = (Integer) parameters.getOrDefault("neighbors", neighbors);
                uniform = (Boolean) parameters.getOrDefault("uniform", uniform);
            }
            
            // 转换为灰度图像 / Convert to grayscale
            ImageData grayscale = input.toGrayscale();
            IMatrix<Double> image = grayscale.getChannel(0);
            
            // 计算LBP / Calculate LBP
            IMatrix<Double> lbpImage = calculateLBP(image, radius, neighbors);
            
            // 计算LBP直方图 / Calculate LBP histogram
            int numPatterns = uniform ? neighbors + 2 : (int) Math.pow(2, neighbors);
            IVector<Double> histogram = calculateLBPHistogram(lbpImage, numPatterns, uniform);
            
            // 计算统计信息 / Calculate statistics
            Map<String, Double> statistics = calculateLBPStatistics(lbpImage, histogram);
            
            // 计算均匀性 / Calculate uniformity
            double uniformity = calculateUniformity(histogram);
            
            return new LBPResult(histogram, lbpImage, uniformity, statistics);
        }
        
        @Override
        public java.util.List<AnalysisResult> analyzeBatch(java.util.List<ImageData> inputs) throws ImageProcessingException {
            java.util.List<AnalysisResult> results = new java.util.ArrayList<>();
            for (ImageData input : inputs) {
                results.add(analyze(input));
            }
            return results;
        }
        
        @Override
        public String getName() { return "LBPAnalyzer"; }
        
        @Override
        public String getDescription() { return "Local Binary Pattern feature extractor"; }
        
        @Override
        public String getAnalysisType() { return "LBP"; }
        
        @Override
        public java.util.Set<String> getSupportedParameters() {
            return java.util.Set.of("radius", "neighbors", "uniform");
        }
        
        @Override
        public Map<String, Object> getDefaultParameters() {
            Map<String, Object> params = new java.util.HashMap<>();
            params.put("radius", 1);
            params.put("neighbors", 8);
            params.put("uniform", true);
            return params;
        }
        
        @Override
        public boolean validateInput(ImageData input) {
            return input != null && input.getChannels() >= 1;
        }
        
        @Override
        public boolean validateParameters(Map<String, Object> parameters) {
            if (parameters == null) return true;
            
            Integer r = (Integer) parameters.get("radius");
            Integer n = (Integer) parameters.get("neighbors");
            
            return (r == null || r > 0) && (n == null || (n > 0 && n % 2 == 0));
        }
        
        @Override
        public int getFeatureDimension() {
            return uniform ? neighbors + 2 : (int) Math.pow(2, neighbors);
        }
        
        @Override
        public IImageAnalyzer clone() {
            LBPAnalyzer cloned = new LBPAnalyzer();
            cloned.radius = this.radius;
            cloned.neighbors = this.neighbors;
            cloned.uniform = this.uniform;
            return cloned;
        }
        
        // ========== 私有方法 / Private Methods ==========
        
        private IMatrix<Double> calculateLBP(IMatrix<Double> image, int radius, int neighbors) {
            int height = image.getRowNum();
            int width = image.getColNum();
            IMatrix<Double> lbp = Linalg.zeros(height, width);
            
            // 计算邻域采样点 / Calculate neighborhood sampling points
            double angleStep = 2 * Math.PI / neighbors;
            int[] dx = new int[neighbors];
            int[] dy = new int[neighbors];
            
            for (int i = 0; i < neighbors; i++) {
                double angle = i * angleStep;
                dx[i] = (int) Math.round(radius * Math.cos(angle));
                dy[i] = (int) Math.round(radius * Math.sin(angle));
            }
            
            // 计算LBP值 / Calculate LBP values
            for (int y = radius; y < height - radius; y++) {
                for (int x = radius; x < width - radius; x++) {
                    double centerValue = image.get(y, x);
                    int pattern = 0;
                    
                    for (int i = 0; i < neighbors; i++) {
                        int nx = x + dx[i];
                        int ny = y + dy[i];
                        
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            double neighborValue = image.get(ny, nx);
                            if (neighborValue >= centerValue) {
                                pattern |= (1 << i);
                            }
                        }
                    }
                    
                    lbp.set(y, x, (double) pattern);
                }
            }
            
            return lbp;
        }
        
        private IVector<Double> calculateLBPHistogram(IMatrix<Double> lbpImage, int numPatterns, boolean uniform) {
            IVector<Double> histogram = Linalg.zeros(numPatterns);
            int height = lbpImage.getRowNum();
            int width = lbpImage.getColNum();
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int pattern = lbpImage.get(y, x).intValue();
                    
                    if (uniform) {
                        int bin = getUniformPattern(pattern, neighbors);
                        histogram.set(bin, histogram.get(bin) + 1);
                    } else {
                        if (pattern < numPatterns) {
                            histogram.set(pattern, histogram.get(pattern) + 1);
                        }
                    }
                }
            }
            
            // 归一化 / Normalize
            double sum = histogram.sum();
            if (sum > 0) {
                histogram = histogram.multiplyScalar(1.0 / sum);
            }
            
            return histogram;
        }
        
        private int getUniformPattern(int pattern, int neighbors) {
            // 计算模式的跳变次数 / Calculate number of transitions in pattern
            int transitions = 0;
            int prev = (pattern >> (neighbors - 1)) & 1;
            
            for (int i = 0; i < neighbors; i++) {
                int curr = (pattern >> i) & 1;
                if (curr != prev) {
                    transitions++;
                }
                prev = curr;
            }
            
            // 如果跳变次数 <= 2，则为均匀模式 / If transitions <= 2, it's uniform pattern
            if (transitions <= 2) {
                return Integer.bitCount(pattern);
            } else {
                return neighbors + 1; // 非均匀模式 / Non-uniform pattern
            }
        }
        
        private Map<String, Double> calculateLBPStatistics(IMatrix<Double> lbpImage, IVector<Double> histogram) {
            Map<String, Double> stats = new java.util.HashMap<>();
            
            // 基本统计 / Basic statistics
            IVector<Double> flatImage = lbpImage.flatten();
            stats.put("mean", flatImage.mean());
            stats.put("std", flatImage.std());
            stats.put("entropy", calculateEntropy(histogram));
            stats.put("energy", histogram.dot(histogram));
            
            return stats;
        }
        
        private double calculateUniformity(IVector<Double> histogram) {
            double uniformity = 0.0;
            for (int i = 0; i < histogram.length(); i++) {
                double p = histogram.get(i);
                uniformity += p * p;
            }
            return uniformity;
        }
        
        private double calculateEntropy(IVector<Double> histogram) {
            double entropy = 0.0;
            for (int i = 0; i < histogram.length(); i++) {
                double p = histogram.get(i);
                if (p > 0) {
                    entropy -= p * Math.log(p) / Math.log(2);
                }
            }
            return entropy;
        }
    }
    
    /**
     * HOG特征提取器 / HOG Feature Extractor
     */
    public static class HOGAnalyzer implements IImageAnalyzer {
        
        private int cellSize = 8;
        private int blockSize = 2;
        private int numBins = 9;
        private boolean signedGradients = false;
        
        /**
         * HOG特征结果 / HOG Feature Result
         */
        public static class HOGResult implements AnalysisResult {
            private IVector<Double> features;
            private IMatrix<Double> gradientMagnitude;
            private IMatrix<Double> gradientDirection;
            private Map<String, Double> statistics;
            
            public HOGResult(IVector<Double> features, IMatrix<Double> gradientMagnitude,
                           IMatrix<Double> gradientDirection, Map<String, Double> statistics) {
                this.features = features;
                this.gradientMagnitude = gradientMagnitude;
                this.gradientDirection = gradientDirection;
                this.statistics = statistics;
            }
            
            @Override
            public String getAnalysisType() { return "HOG"; }
            
            @Override
            public IVector<Double> getFeatureVector() { return features; }
            
            @Override
            public Map<String, Double> getNumericResults() { return statistics; }
            
            @Override
            public String getDescription() { 
                return String.format("HOG features with %d dimensions", features.length()); 
            }
            
            @Override
            public double getConfidence() { return 1.0; }
            
            @Override
            public String toJSON() {
                return String.format("{\"type\":\"HOG\",\"feature_size\":%d}", features.length());
            }
            
            public IMatrix<Double> getGradientMagnitude() { return gradientMagnitude; }
            public IMatrix<Double> getGradientDirection() { return gradientDirection; }
        }
        
        @Override
        public AnalysisResult analyze(ImageData input) throws ImageProcessingException {
            return analyze(input, getDefaultParameters());
        }
        
        @Override
        public AnalysisResult analyze(ImageData input, Map<String, Object> parameters) throws ImageProcessingException {
            if (!validateInput(input)) {
                throw ImageProcessingException.invalidInput("Input must be grayscale image");
            }
            
            // 应用参数 / Apply parameters
            if (parameters != null) {
                cellSize = (Integer) parameters.getOrDefault("cellSize", cellSize);
                blockSize = (Integer) parameters.getOrDefault("blockSize", blockSize);
                numBins = (Integer) parameters.getOrDefault("numBins", numBins);
                signedGradients = (Boolean) parameters.getOrDefault("signedGradients", signedGradients);
            }
            
            // 转换为灰度图像 / Convert to grayscale
            ImageData grayscale = input.toGrayscale();
            IMatrix<Double> image = grayscale.getChannel(0);
            
            // 计算梯度 / Calculate gradients
            IMatrix<Double> gradX = calculateGradientX(image);
            IMatrix<Double> gradY = calculateGradientY(image);
            IMatrix<Double> magnitude = calculateMagnitude(gradX, gradY);
            IMatrix<Double> direction = calculateDirection(gradX, gradY);
            
            // 计算HOG特征 / Calculate HOG features
            IVector<Double> features = calculateHOGFeatures(magnitude, direction);
            
            // 计算统计信息 / Calculate statistics
            Map<String, Double> statistics = calculateHOGStatistics(features, magnitude);
            
            return new HOGResult(features, magnitude, direction, statistics);
        }
        
        @Override
        public java.util.List<AnalysisResult> analyzeBatch(java.util.List<ImageData> inputs) throws ImageProcessingException {
            java.util.List<AnalysisResult> results = new java.util.ArrayList<>();
            for (ImageData input : inputs) {
                results.add(analyze(input));
            }
            return results;
        }
        
        @Override
        public String getName() { return "HOGAnalyzer"; }
        
        @Override
        public String getDescription() { return "Histogram of Oriented Gradients feature extractor"; }
        
        @Override
        public String getAnalysisType() { return "HOG"; }
        
        @Override
        public java.util.Set<String> getSupportedParameters() {
            return java.util.Set.of("cellSize", "blockSize", "numBins", "signedGradients");
        }
        
        @Override
        public Map<String, Object> getDefaultParameters() {
            Map<String, Object> params = new java.util.HashMap<>();
            params.put("cellSize", 8);
            params.put("blockSize", 2);
            params.put("numBins", 9);
            params.put("signedGradients", false);
            return params;
        }
        
        @Override
        public boolean validateInput(ImageData input) {
            return input != null && input.getChannels() >= 1;
        }
        
        @Override
        public boolean validateParameters(Map<String, Object> parameters) {
            if (parameters == null) return true;
            
            Integer cs = (Integer) parameters.get("cellSize");
            Integer bs = (Integer) parameters.get("blockSize");
            Integer nb = (Integer) parameters.get("numBins");
            
            return (cs == null || cs > 0) && (bs == null || bs > 0) && (nb == null || nb > 0);
        }
        
        @Override
        public int getFeatureDimension() {
            // 计算HOG特征维度 / Calculate HOG feature dimension
            // 这是一个估算，实际维度取决于图像尺寸 / This is an estimate, actual dimension depends on image size
            return blockSize * blockSize * numBins * 36; // 假设64x128图像 / Assuming 64x128 image
        }
        
        @Override
        public IImageAnalyzer clone() {
            HOGAnalyzer cloned = new HOGAnalyzer();
            cloned.cellSize = this.cellSize;
            cloned.blockSize = this.blockSize;
            cloned.numBins = this.numBins;
            cloned.signedGradients = this.signedGradients;
            return cloned;
        }
        
        // ========== 私有方法 / Private Methods ==========
        
        private IMatrix<Double> calculateGradientX(IMatrix<Double> image) {
            int height = image.getRowNum();
            int width = image.getColNum();
            IMatrix<Double> gradX = Linalg.zeros(height, width);
            
            for (int y = 0; y < height; y++) {
                for (int x = 1; x < width - 1; x++) {
                    double gx = image.get(y, x + 1) - image.get(y, x - 1);
                    gradX.set(y, x, gx);
                }
            }
            
            return gradX;
        }
        
        private IMatrix<Double> calculateGradientY(IMatrix<Double> image) {
            int height = image.getRowNum();
            int width = image.getColNum();
            IMatrix<Double> gradY = Linalg.zeros(height, width);
            
            for (int y = 1; y < height - 1; y++) {
                for (int x = 0; x < width; x++) {
                    double gy = image.get(y + 1, x) - image.get(y - 1, x);
                    gradY.set(y, x, gy);
                }
            }
            
            return gradY;
        }
        
        private IMatrix<Double> calculateMagnitude(IMatrix<Double> gradX, IMatrix<Double> gradY) {
            int height = gradX.getRowNum();
            int width = gradX.getColNum();
            IMatrix<Double> magnitude = Linalg.zeros(height, width);
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double gx = gradX.get(y, x);
                    double gy = gradY.get(y, x);
                    magnitude.set(y, x, Math.sqrt(gx * gx + gy * gy));
                }
            }
            
            return magnitude;
        }
        
        private IMatrix<Double> calculateDirection(IMatrix<Double> gradX, IMatrix<Double> gradY) {
            int height = gradX.getRowNum();
            int width = gradX.getColNum();
            IMatrix<Double> direction = Linalg.zeros(height, width);
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double gx = gradX.get(y, x);
                    double gy = gradY.get(y, x);
                    double angle = Math.atan2(gy, gx);
                    
                    if (!signedGradients) {
                        angle = Math.abs(angle);
                    }
                    
                    direction.set(y, x, angle);
                }
            }
            
            return direction;
        }
        
        private IVector<Double> calculateHOGFeatures(IMatrix<Double> magnitude, IMatrix<Double> direction) {
            int height = magnitude.getRowNum();
            int width = magnitude.getColNum();
            
            // 计算cell的数量 / Calculate number of cells
            int cellsY = height / cellSize;
            int cellsX = width / cellSize;
            
            // 计算block的数量 / Calculate number of blocks
            int blocksY = cellsY - blockSize + 1;
            int blocksX = cellsX - blockSize + 1;
            
            // 计算每个cell的直方图 / Calculate histogram for each cell
            IMatrix<Double> cellHistograms = Linalg.zeros(cellsY * cellsX, numBins);
            
            for (int cy = 0; cy < cellsY; cy++) {
                for (int cx = 0; cx < cellsX; cx++) {
                    IVector<Double> histogram = calculateCellHistogram(magnitude, direction, 
                                                                     cx * cellSize, cy * cellSize);
                    int cellIndex = cy * cellsX + cx;
                    
                    for (int bin = 0; bin < numBins; bin++) {
                        cellHistograms.set(cellIndex, bin, histogram.get(bin));
                    }
                }
            }
            
            // 计算block的特征 / Calculate block features
            java.util.List<Double> features = new java.util.ArrayList<>();
            
            for (int by = 0; by < blocksY; by++) {
                for (int bx = 0; bx < blocksX; bx++) {
                    // 收集block内的cell直方图 / Collect cell histograms within block
                    java.util.List<Double> blockFeatures = new java.util.ArrayList<>();
                    
                    for (int dy = 0; dy < blockSize; dy++) {
                        for (int dx = 0; dx < blockSize; dx++) {
                            int cellIndex = (by + dy) * cellsX + (bx + dx);
                            for (int bin = 0; bin < numBins; bin++) {
                                blockFeatures.add(cellHistograms.get(cellIndex, bin));
                            }
                        }
                    }
                    
                    // L2归一化 / L2 normalization
                    double norm = 0.0;
                    for (double f : blockFeatures) {
                        norm += f * f;
                    }
                    norm = Math.sqrt(norm);
                    
                    if (norm > 1e-10) {
                        for (int i = 0; i < blockFeatures.size(); i++) {
                            blockFeatures.set(i, blockFeatures.get(i) / norm);
                        }
                    }
                    
                    features.addAll(blockFeatures);
                }
            }
            
            // 转换为向量 / Convert to vector
            double[] featureArray = new double[features.size()];
            for (int i = 0; i < features.size(); i++) {
                featureArray[i] = features.get(i);
            }
            
            return Linalg.vector(featureArray);
        }
        
        private IVector<Double> calculateCellHistogram(IMatrix<Double> magnitude, IMatrix<Double> direction, 
                                                      int startX, int startY) {
            IVector<Double> histogram = Linalg.zeros(numBins);
            double angleRange = signedGradients ? (2 * Math.PI) : Math.PI;
            double binSize = angleRange / numBins;
            
            for (int dy = 0; dy < cellSize; dy++) {
                for (int dx = 0; dx < cellSize; dx++) {
                    int y = startY + dy;
                    int x = startX + dx;
                    
                    if (y < magnitude.getRowNum() && x < magnitude.getColNum()) {
                        double mag = magnitude.get(y, x);
                        double angle = direction.get(y, x);
                        
                        // 将角度映射到[0, angleRange]范围 / Map angle to [0, angleRange] range
                        if (angle < 0) angle += angleRange;
                        
                        // 计算bin索引 / Calculate bin index
                        int bin = (int) (angle / binSize);
                        bin = Math.min(bin, numBins - 1);
                        
                        // 双线性插值 / Bilinear interpolation
                        double binCenter = (bin + 0.5) * binSize;
                        double weight = 1.0 - Math.abs(angle - binCenter) / binSize;
                        
                        histogram.set(bin, histogram.get(bin) + mag * weight);
                        
                        // 相邻bin也贡献一部分 / Adjacent bin also contributes
                        int adjacentBin = (angle > binCenter) ? (bin + 1) % numBins : (bin - 1 + numBins) % numBins;
                        histogram.set(adjacentBin, histogram.get(adjacentBin) + mag * (1.0 - weight));
                    }
                }
            }
            
            return histogram;
        }
        
        private Map<String, Double> calculateHOGStatistics(IVector<Double> features, IMatrix<Double> magnitude) {
            Map<String, Double> stats = new java.util.HashMap<>();
            
            stats.put("feature_mean", features.mean());
            stats.put("feature_std", features.std());
            stats.put("feature_max", features.max());
            stats.put("feature_min", features.min());
            stats.put("magnitude_mean", magnitude.mean());
            stats.put("magnitude_std", magnitude.std());
            
            return stats;
        }
    }
}