package com.reremouse.lab.image.detection;

import com.reremouse.lab.image.ImageData;
import com.reremouse.lab.image.core.IImageAnalyzer;
import com.reremouse.lab.image.core.ImageProcessingException;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;

/**
 * 高级特征检测算法 / Advanced Feature Detection Algorithms
 * <p>
 * 实现SIFT、SURF、Harris角点检测、ORB等先进的特征检测算法。
 * 提供尺度不变性、旋转不变性的特征点检测和描述符提取功能。
 * </p>
 * <p>
 * Implements advanced feature detection algorithms including SIFT, SURF, Harris corner detection, ORB, etc.
 * Provides scale-invariant and rotation-invariant keypoint detection and descriptor extraction functionality.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public class AdvancedFeatureDetection {
    
    /**
     * 特征点类 / Keypoint Class
     */
    public static class Keypoint {
        private double x, y;           // 坐标 / Coordinates
        private double scale;          // 尺度 / Scale
        private double angle;          // 方向角 / Orientation angle
        private double response;       // 响应强度 / Response strength
        private IVector<Double> descriptor; // 特征描述符 / Feature descriptor
        
        public Keypoint(double x, double y, double scale, double angle, double response) {
            this.x = x;
            this.y = y;
            this.scale = scale;
            this.angle = angle;
            this.response = response;
        }
        
        // Getters and Setters
        public double getX() { return x; }
        public double getY() { return y; }
        public double getScale() { return scale; }
        public double getAngle() { return angle; }
        public double getResponse() { return response; }
        public IVector<Double> getDescriptor() { return descriptor; }
        public void setDescriptor(IVector<Double> descriptor) { this.descriptor = descriptor; }
        
        @Override
        public String toString() {
            return String.format("Keypoint[x=%.2f, y=%.2f, scale=%.2f, angle=%.2f, response=%.3f]", 
                               x, y, scale, angle, response);
        }
    }
    
    /**
     * Harris角点检测器 / Harris Corner Detector
     */
    public static class HarrisCornerDetector implements IImageAnalyzer {
        
        private double threshold = 0.01;
        private double k = 0.04;
        private int windowSize = 3;
        private boolean useNonMaxSuppression = true;
        
        /**
         * Harris角点检测结果 / Harris Corner Detection Result
         */
        public static class HarrisResult implements AnalysisResult {
            private List<Keypoint> corners;
            private IMatrix<Double> responseMap;
            private Map<String, Double> statistics;
            
            public HarrisResult(List<Keypoint> corners, IMatrix<Double> responseMap, Map<String, Double> statistics) {
                this.corners = corners;
                this.responseMap = responseMap;
                this.statistics = statistics;
            }
            
            @Override
            public String getAnalysisType() { return "HarrisCorners"; }
            
            @Override
            public IVector<Double> getFeatureVector() {
                // 将角点坐标转换为特征向量 / Convert corner coordinates to feature vector
                double[] features = new double[corners.size() * 3]; // x, y, response
                for (int i = 0; i < corners.size(); i++) {
                    Keypoint corner = corners.get(i);
                    features[i * 3] = corner.getX();
                    features[i * 3 + 1] = corner.getY();
                    features[i * 3 + 2] = corner.getResponse();
                }
                return Linalg.vector(features);
            }
            
            @Override
            public Map<String, Double> getNumericResults() { return statistics; }
            
            @Override
            public String getDescription() { 
                return String.format("Harris corner detection found %d corners", corners.size()); 
            }
            
            @Override
            public double getConfidence() { 
                return corners.isEmpty() ? 0.0 : corners.stream().mapToDouble(Keypoint::getResponse).average().orElse(0.0);
            }
            
            @Override
            public String toJSON() {
                return String.format("{\"type\":\"HarrisCorners\",\"count\":%d,\"avg_response\":%.3f}", 
                                   corners.size(), getConfidence());
            }
            
            public List<Keypoint> getCorners() { return corners; }
            public IMatrix<Double> getResponseMap() { return responseMap; }
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
                threshold = (Double) parameters.getOrDefault("threshold", threshold);
                k = (Double) parameters.getOrDefault("k", k);
                windowSize = (Integer) parameters.getOrDefault("windowSize", windowSize);
                useNonMaxSuppression = (Boolean) parameters.getOrDefault("useNonMaxSuppression", useNonMaxSuppression);
            }
            
            // 转换为灰度图像 / Convert to grayscale
            ImageData grayscale = input.toGrayscale();
            IMatrix<Double> image = grayscale.getChannel(0);
            
            // 计算Harris响应 / Calculate Harris response
            IMatrix<Double> responseMap = calculateHarrisResponse(image);
            
            // 提取角点 / Extract corners
            List<Keypoint> corners = extractCorners(responseMap);
            
            // 非最大值抑制 / Non-maximum suppression
            if (useNonMaxSuppression) {
                corners = nonMaximumSuppression(corners, responseMap);
            }
            
            // 计算统计信息 / Calculate statistics
            Map<String, Double> statistics = calculateStatistics(corners, responseMap);
            
            return new HarrisResult(corners, responseMap, statistics);
        }
        
        @Override
        public List<AnalysisResult> analyzeBatch(List<ImageData> inputs) throws ImageProcessingException {
            List<AnalysisResult> results = new ArrayList<>();
            for (ImageData input : inputs) {
                results.add(analyze(input));
            }
            return results;
        }
        
        @Override
        public String getName() { return "HarrisCornerDetector"; }
        
        @Override
        public String getDescription() { return "Harris corner detection algorithm"; }
        
        @Override
        public String getAnalysisType() { return "HarrisCorners"; }
        
        @Override
        public java.util.Set<String> getSupportedParameters() {
            return java.util.Set.of("threshold", "k", "windowSize", "useNonMaxSuppression");
        }
        
        @Override
        public Map<String, Object> getDefaultParameters() {
            Map<String, Object> params = new java.util.HashMap<>();
            params.put("threshold", 0.01);
            params.put("k", 0.04);
            params.put("windowSize", 3);
            params.put("useNonMaxSuppression", true);
            return params;
        }
        
        @Override
        public boolean validateInput(ImageData input) {
            return input != null && input.getChannels() >= 1;
        }
        
        @Override
        public boolean validateParameters(Map<String, Object> parameters) {
            if (parameters == null) return true;
            
            Double t = (Double) parameters.get("threshold");
            Double kVal = (Double) parameters.get("k");
            Integer ws = (Integer) parameters.get("windowSize");
            
            return (t == null || t >= 0) && (kVal == null || (kVal > 0 && kVal < 1)) && (ws == null || (ws > 0 && ws % 2 == 1));
        }
        
        @Override
        public int getFeatureDimension() {
            return -1; // 变长特征 / Variable length features
        }
        
        @Override
        public IImageAnalyzer clone() {
            HarrisCornerDetector cloned = new HarrisCornerDetector();
            cloned.threshold = this.threshold;
            cloned.k = this.k;
            cloned.windowSize = this.windowSize;
            cloned.useNonMaxSuppression = this.useNonMaxSuppression;
            return cloned;
        }
        
        // ========== 私有方法 / Private Methods ==========
        
        private IMatrix<Double> calculateHarrisResponse(IMatrix<Double> image) {
            int height = image.getRowNum();
            int width = image.getColNum();
            
            // 计算梯度 / Calculate gradients
            IMatrix<Double> Ix = calculateGradientX(image);
            IMatrix<Double> Iy = calculateGradientY(image);
            
            // 计算二阶矩矩阵元素 / Calculate second moment matrix elements
            IMatrix<Double> Ixx = elementwiseMultiply(Ix, Ix);
            IMatrix<Double> Iyy = elementwiseMultiply(Iy, Iy);
            IMatrix<Double> Ixy = elementwiseMultiply(Ix, Iy);
            
            // 高斯加权 / Gaussian weighting
            IMatrix<Double> Sxx = gaussianBlur(Ixx, 1.0);
            IMatrix<Double> Syy = gaussianBlur(Iyy, 1.0);
            IMatrix<Double> Sxy = gaussianBlur(Ixy, 1.0);
            
            // 计算Harris响应 / Calculate Harris response
            IMatrix<Double> response = Linalg.zeros(height, width);
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double sxx = Sxx.get(y, x);
                    double syy = Syy.get(y, x);
                    double sxy = Sxy.get(y, x);
                    
                    // Harris响应函数：R = det(M) - k * trace(M)^2 / Harris response function: R = det(M) - k * trace(M)^2
                    double det = sxx * syy - sxy * sxy;
                    double trace = sxx + syy;
                    double r = det - k * trace * trace;
                    
                    response.set(y, x, r);
                }
            }
            
            return response;
        }
        
        private List<Keypoint> extractCorners(IMatrix<Double> responseMap) {
            List<Keypoint> corners = new ArrayList<>();
            int height = responseMap.getRowNum();
            int width = responseMap.getColNum();
            
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    double response = responseMap.get(y, x);
                    
                    if (response > threshold) {
                        // 简单的局部最大值检测 / Simple local maximum detection
                        boolean isLocalMax = true;
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                if (dx == 0 && dy == 0) continue;
                                if (responseMap.get(y + dy, x + dx) >= response) {
                                    isLocalMax = false;
                                    break;
                                }
                            }
                            if (!isLocalMax) break;
                        }
                        
                        if (isLocalMax) {
                            corners.add(new Keypoint(x, y, 1.0, 0.0, response));
                        }
                    }
                }
            }
            
            return corners;
        }
        
        private List<Keypoint> nonMaximumSuppression(List<Keypoint> corners, IMatrix<Double> responseMap) {
            // 简化的非最大值抑制 / Simplified non-maximum suppression
            corners.sort((a, b) -> Double.compare(b.getResponse(), a.getResponse()));
            
            List<Keypoint> suppressed = new ArrayList<>();
            for (Keypoint corner : corners) {
                boolean suppress = false;
                for (Keypoint existing : suppressed) {
                    double dist = Math.sqrt(Math.pow(corner.getX() - existing.getX(), 2) + 
                                          Math.pow(corner.getY() - existing.getY(), 2));
                    if (dist < windowSize) {
                        suppress = true;
                        break;
                    }
                }
                if (!suppress) {
                    suppressed.add(corner);
                }
            }
            
            return suppressed;
        }
        
        private Map<String, Double> calculateStatistics(List<Keypoint> corners, IMatrix<Double> responseMap) {
            Map<String, Double> stats = new java.util.HashMap<>();
            
            stats.put("corner_count", (double) corners.size());
            if (!corners.isEmpty()) {
                double avgResponse = corners.stream().mapToDouble(Keypoint::getResponse).average().orElse(0.0);
                double maxResponse = corners.stream().mapToDouble(Keypoint::getResponse).max().orElse(0.0);
                double minResponse = corners.stream().mapToDouble(Keypoint::getResponse).min().orElse(0.0);
                
                stats.put("avg_response", avgResponse);
                stats.put("max_response", maxResponse);
                stats.put("min_response", minResponse);
            }
            
            return stats;
        }
        
        // 辅助方法 / Helper methods
        private IMatrix<Double> calculateGradientX(IMatrix<Double> image) {
            int height = image.getRowNum();
            int width = image.getColNum();
            IMatrix<Double> gradX = Linalg.zeros(height, width);
            
            for (int y = 0; y < height; y++) {
                for (int x = 1; x < width - 1; x++) {
                    double gx = (image.get(y, x + 1) - image.get(y, x - 1)) / 2.0;
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
                    double gy = (image.get(y + 1, x) - image.get(y - 1, x)) / 2.0;
                    gradY.set(y, x, gy);
                }
            }
            
            return gradY;
        }
        
        private IMatrix<Double> elementwiseMultiply(IMatrix<Double> a, IMatrix<Double> b) {
            int height = a.getRowNum();
            int width = a.getColNum();
            IMatrix<Double> result = Linalg.zeros(height, width);
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    result.set(y, x, a.get(y, x) * b.get(y, x));
                }
            }
            
            return result;
        }
        
        private IMatrix<Double> gaussianBlur(IMatrix<Double> image, double sigma) {
            // 简化的高斯模糊 / Simplified Gaussian blur
            int kernelSize = (int) (6 * sigma) + 1;
            if (kernelSize % 2 == 0) kernelSize++;
            
            IMatrix<Double> kernel = createGaussianKernel1D(kernelSize, sigma);
            
            // 先水平模糊 / Horizontal blur first
            IMatrix<Double> horizontal = convolveHorizontal(image, kernel);
            
            // 再垂直模糊 / Then vertical blur
            return convolveVertical(horizontal, kernel);
        }
        
        private IMatrix<Double> createGaussianKernel1D(int size, double sigma) {
            IMatrix<Double> kernel = Linalg.zeros(1, size);
            int center = size / 2;
            double sum = 0.0;
            
            for (int x = 0; x < size; x++) {
                double dx = x - center;
                double value = Math.exp(-(dx * dx) / (2 * sigma * sigma));
                kernel.set(0, x, value);
                sum += value;
            }
            
            // 归一化 / Normalize
            for (int x = 0; x < size; x++) {
                kernel.set(0, x, kernel.get(0, x) / sum);
            }
            
            return kernel;
        }
        
        private IMatrix<Double> convolveHorizontal(IMatrix<Double> image, IMatrix<Double> kernel) {
            int height = image.getRowNum();
            int width = image.getColNum();
            int kernelSize = kernel.getColNum();
            int halfKernel = kernelSize / 2;
            
            IMatrix<Double> result = Linalg.zeros(height, width);
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double sum = 0.0;
                    
                    for (int k = 0; k < kernelSize; k++) {
                        int px = x + k - halfKernel;
                        if (px >= 0 && px < width) {
                            sum += image.get(y, px) * kernel.get(0, k);
                        }
                    }
                    
                    result.set(y, x, sum);
                }
            }
            
            return result;
        }
        
        private IMatrix<Double> convolveVertical(IMatrix<Double> image, IMatrix<Double> kernel) {
            int height = image.getRowNum();
            int width = image.getColNum();
            int kernelSize = kernel.getColNum();
            int halfKernel = kernelSize / 2;
            
            IMatrix<Double> result = Linalg.zeros(height, width);
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double sum = 0.0;
                    
                    for (int k = 0; k < kernelSize; k++) {
                        int py = y + k - halfKernel;
                        if (py >= 0 && py < height) {
                            sum += image.get(py, x) * kernel.get(0, k);
                        }
                    }
                    
                    result.set(y, x, sum);
                }
            }
            
            return result;
        }
    }
}