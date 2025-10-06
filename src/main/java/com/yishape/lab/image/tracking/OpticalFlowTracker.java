package com.yishape.lab.image.tracking;

import com.yishape.lab.image.ImageData;
import com.yishape.lab.image.core.IImageProcessor;
import com.yishape.lab.image.core.ImageProcessingException;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;

import java.util.Map;

/**
 * 光流和运动追踪算法 / Optical Flow and Motion Tracking Algorithms
 * <p>
 * 实现Lucas-Kanade光流、Horn-Schunck光流、Farneback稠密光流等运动分析算法。
 * 提供帧间运动估计、目标追踪、运动场分析等功能。
 * </p>
 * <p>
 * Implements optical flow and motion tracking algorithms including Lucas-Kanade, Horn-Schunck, Farneback dense optical flow.
 * Provides inter-frame motion estimation, object tracking, and motion field analysis functionality.
 * </p>
 *
 * @author RereMouse
 * @version 2.0
 * @since 2.0
 */
public class OpticalFlowTracker {
    
    /**
     * 光流向量类 / Optical Flow Vector Class
     */
    public static class FlowVector {
        private double x, y;     // 位置 / Position
        private double dx, dy;   // 光流向量 / Flow vector
        private double magnitude; // 幅值 / Magnitude
        private double angle;    // 角度 / Angle
        private double confidence; // 置信度 / Confidence
        
        public FlowVector(double x, double y, double dx, double dy) {
            this.x = x;
            this.y = y;
            this.dx = dx;
            this.dy = dy;
            this.magnitude = Math.sqrt(dx * dx + dy * dy);
            this.angle = Math.atan2(dy, dx);
            this.confidence = 1.0;
        }
        
        // Getters
        public double getX() { return x; }
        public double getY() { return y; }
        public double getDx() { return dx; }
        public double getDy() { return dy; }
        public double getMagnitude() { return magnitude; }
        public double getAngle() { return angle; }
        public double getConfidence() { return confidence; }
        public void setConfidence(double confidence) { this.confidence = confidence; }
        
        @Override
        public String toString() {
            return String.format("FlowVector[pos=(%.2f,%.2f), flow=(%.3f,%.3f), mag=%.3f, angle=%.3f]", 
                               x, y, dx, dy, magnitude, angle);
        }
    }
    
    /**
     * 光流结果类 / Optical Flow Result Class
     */
    public static class OpticalFlowResult {
        private IMatrix<Double> flowX;        // X方向光流 / X-direction flow
        private IMatrix<Double> flowY;        // Y方向光流 / Y-direction flow
        private IMatrix<Double> magnitude;    // 光流幅值 / Flow magnitude
        private IMatrix<Double> angle;        // 光流角度 / Flow angle
        private double avgMagnitude;          // 平均幅值 / Average magnitude
        private double maxMagnitude;          // 最大幅值 / Maximum magnitude
        private Map<String, Double> statistics; // 统计信息 / Statistics
        
        public OpticalFlowResult(IMatrix<Double> flowX, IMatrix<Double> flowY) {
            this.flowX = flowX;
            this.flowY = flowY;
            calculateDerivedFields();
        }
        
        private void calculateDerivedFields() {
            int height = flowX.getRowNum();
            int width = flowX.getColNum();
            
            magnitude = Linalg.zeros(height, width);
            angle = Linalg.zeros(height, width);
            
            double totalMagnitude = 0.0;
            double max = 0.0;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double fx = flowX.get(y, x);
                    double fy = flowY.get(y, x);
                    double mag = Math.sqrt(fx * fx + fy * fy);
                    double ang = Math.atan2(fy, fx);
                    
                    magnitude.set(y, x, mag);
                    angle.set(y, x, ang);
                    
                    totalMagnitude += mag;
                    if (mag > max) max = mag;
                }
            }
            
            avgMagnitude = totalMagnitude / (height * width);
            maxMagnitude = max;
            
            // 计算统计信息 / Calculate statistics
            statistics = new java.util.HashMap<>();
            statistics.put("avg_magnitude", avgMagnitude);
            statistics.put("max_magnitude", maxMagnitude);
            statistics.put("flow_density", calculateFlowDensity());
        }
        
        private double calculateFlowDensity() {
            int count = 0;
            int total = magnitude.getRowNum() * magnitude.getColNum();
            
            for (int y = 0; y < magnitude.getRowNum(); y++) {
                for (int x = 0; x < magnitude.getColNum(); x++) {
                    if (magnitude.get(y, x) > 0.1) { // 阈值可调 / Adjustable threshold
                        count++;
                    }
                }
            }
            
            return (double) count / total;
        }
        
        // Getters
        public IMatrix<Double> getFlowX() { return flowX; }
        public IMatrix<Double> getFlowY() { return flowY; }
        public IMatrix<Double> getMagnitude() { return magnitude; }
        public IMatrix<Double> getAngle() { return angle; }
        public double getAvgMagnitude() { return avgMagnitude; }
        public double getMaxMagnitude() { return maxMagnitude; }
        public Map<String, Double> getStatistics() { return statistics; }
    }
    
    /**
     * Lucas-Kanade光流算法 / Lucas-Kanade Optical Flow Algorithm
     */
    public static class LucasKanadeOpticalFlow implements IImageProcessor {
        
        private int windowSize = 15;
        private double threshold = 0.01;
        private int maxIterations = 30;
        private double epsilon = 0.01;
        
        @Override
        public ImageData process(ImageData input) throws ImageProcessingException {
            throw new UnsupportedOperationException("Lucas-Kanade requires two frames for optical flow calculation");
        }
        
        @Override
        public ImageData process(ImageData input, Map<String, Object> parameters) throws ImageProcessingException {
            throw new UnsupportedOperationException("Use calculateOpticalFlow method for Lucas-Kanade optical flow");
        }
        
        /**
         * 计算两帧之间的光流 / Calculate optical flow between two frames
         * 
         * @param frame1 第一帧 / First frame
         * @param frame2 第二帧 / Second frame
         * @return 光流结果 / Optical flow result
         * @throws ImageProcessingException 处理异常 / Processing exception
         */
        public OpticalFlowResult calculateOpticalFlow(ImageData frame1, ImageData frame2) throws ImageProcessingException {
            return calculateOpticalFlow(frame1, frame2, getDefaultParameters());
        }
        
        /**
         * 计算两帧之间的光流 / Calculate optical flow between two frames
         * 
         * @param frame1 第一帧 / First frame
         * @param frame2 第二帧 / Second frame
         * @param parameters 参数 / Parameters
         * @return 光流结果 / Optical flow result
         * @throws ImageProcessingException 处理异常 / Processing exception
         */
        public OpticalFlowResult calculateOpticalFlow(ImageData frame1, ImageData frame2, Map<String, Object> parameters) 
                throws ImageProcessingException {
            
            if (frame1 == null || frame2 == null) {
                throw ImageProcessingException.invalidInput("Both frames must be provided");
            }
            
            if (frame1.getWidth() != frame2.getWidth() || frame1.getHeight() != frame2.getHeight()) {
                throw ImageProcessingException.invalidInput("Frames must have the same dimensions");
            }
            
            // 应用参数 / Apply parameters
            if (parameters != null) {
                windowSize = (Integer) parameters.getOrDefault("windowSize", windowSize);
                threshold = (Double) parameters.getOrDefault("threshold", threshold);
                maxIterations = (Integer) parameters.getOrDefault("maxIterations", maxIterations);
                epsilon = (Double) parameters.getOrDefault("epsilon", epsilon);
            }
            
            // 转换为灰度图像 / Convert to grayscale
            ImageData gray1 = frame1.toGrayscale();
            ImageData gray2 = frame2.toGrayscale();
            
            IMatrix<Double> I1 = gray1.getChannel(0);
            IMatrix<Double> I2 = gray2.getChannel(0);
            
            // 计算图像梯度和时间导数 / Calculate image gradients and temporal derivative
            IMatrix<Double> Ix = calculateGradientX(I1, I2);
            IMatrix<Double> Iy = calculateGradientY(I1, I2);
            IMatrix<Double> It = calculateTemporalDerivative(I1, I2);
            
            // 计算光流 / Calculate optical flow
            return calculateLucasKanadeFlow(Ix, Iy, It);
        }
        
        @Override
        public String getName() { return "LucasKanadeOpticalFlow"; }
        
        @Override
        public String getDescription() { return "Lucas-Kanade optical flow algorithm"; }
        
        @Override
        public java.util.Set<String> getSupportedParameters() {
            return java.util.Set.of("windowSize", "threshold", "maxIterations", "epsilon");
        }
        
        @Override
        public Map<String, Object> getDefaultParameters() {
            Map<String, Object> params = new java.util.HashMap<>();
            params.put("windowSize", 15);
            params.put("threshold", 0.01);
            params.put("maxIterations", 30);
            params.put("epsilon", 0.01);
            return params;
        }
        
        @Override
        public boolean validateInput(ImageData input) {
            return input != null;
        }
        
        @Override
        public boolean validateParameters(Map<String, Object> parameters) {
            if (parameters == null) return true;
            
            Integer ws = (Integer) parameters.get("windowSize");
            Double t = (Double) parameters.get("threshold");
            Integer mi = (Integer) parameters.get("maxIterations");
            Double e = (Double) parameters.get("epsilon");
            
            return (ws == null || (ws > 0 && ws % 2 == 1)) && 
                   (t == null || t >= 0) && 
                   (mi == null || mi > 0) && 
                   (e == null || e > 0);
        }
        
        @Override
        public IImageProcessor clone() {
            LucasKanadeOpticalFlow cloned = new LucasKanadeOpticalFlow();
            cloned.windowSize = this.windowSize;
            cloned.threshold = this.threshold;
            cloned.maxIterations = this.maxIterations;
            cloned.epsilon = this.epsilon;
            return cloned;
        }
        
        // ========== 私有方法 / Private Methods ==========
        
        private IMatrix<Double> calculateGradientX(IMatrix<Double> I1, IMatrix<Double> I2) {
            int height = I1.getRowNum();
            int width = I1.getColNum();
            IMatrix<Double> Ix = Linalg.zeros(height, width);
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width - 1; x++) {
                    // 使用中心差分法 / Use central difference
                    double ix1 = (x == width - 1) ? 0 : (I1.get(y, x + 1) - I1.get(y, x));
                    double ix2 = (x == width - 1) ? 0 : (I2.get(y, x + 1) - I2.get(y, x));
                    Ix.set(y, x, (ix1 + ix2) / 2.0);
                }
            }
            
            return Ix;
        }
        
        private IMatrix<Double> calculateGradientY(IMatrix<Double> I1, IMatrix<Double> I2) {
            int height = I1.getRowNum();
            int width = I1.getColNum();
            IMatrix<Double> Iy = Linalg.zeros(height, width);
            
            for (int y = 0; y < height - 1; y++) {
                for (int x = 0; x < width; x++) {
                    // 使用中心差分法 / Use central difference
                    double iy1 = (y == height - 1) ? 0 : (I1.get(y + 1, x) - I1.get(y, x));
                    double iy2 = (y == height - 1) ? 0 : (I2.get(y + 1, x) - I2.get(y, x));
                    Iy.set(y, x, (iy1 + iy2) / 2.0);
                }
            }
            
            return Iy;
        }
        
        private IMatrix<Double> calculateTemporalDerivative(IMatrix<Double> I1, IMatrix<Double> I2) {
            int height = I1.getRowNum();
            int width = I1.getColNum();
            IMatrix<Double> It = Linalg.zeros(height, width);
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    It.set(y, x, I2.get(y, x) - I1.get(y, x));
                }
            }
            
            return It;
        }
        
        private OpticalFlowResult calculateLucasKanadeFlow(IMatrix<Double> Ix, IMatrix<Double> Iy, IMatrix<Double> It) {
            int height = Ix.getRowNum();
            int width = Ix.getColNum();
            int halfWindow = windowSize / 2;
            
            IMatrix<Double> flowX = Linalg.zeros(height, width);
            IMatrix<Double> flowY = Linalg.zeros(height, width);
            
            // 对每个像素计算光流 / Calculate optical flow for each pixel
            for (int y = halfWindow; y < height - halfWindow; y++) {
                for (int x = halfWindow; x < width - halfWindow; x++) {
                    
                    // 构建窗口内的线性方程组 / Build linear system within window
                    double sumIxIx = 0, sumIxIy = 0, sumIyIy = 0;
                    double sumIxIt = 0, sumIyIt = 0;
                    
                    for (int dy = -halfWindow; dy <= halfWindow; dy++) {
                        for (int dx = -halfWindow; dx <= halfWindow; dx++) {
                            int px = x + dx;
                            int py = y + dy;
                            
                            double ix = Ix.get(py, px);
                            double iy = Iy.get(py, px);
                            double it = It.get(py, px);
                            
                            sumIxIx += ix * ix;
                            sumIxIy += ix * iy;
                            sumIyIy += iy * iy;
                            sumIxIt += ix * it;
                            sumIyIt += iy * it;
                        }
                    }
                    
                    // 求解2x2线性方程组 / Solve 2x2 linear system
                    // [sumIxIx  sumIxIy] [u] = [-sumIxIt]
                    // [sumIxIy  sumIyIy] [v]   [-sumIyIt]
                    
                    double det = sumIxIx * sumIyIy - sumIxIy * sumIxIy;
                    
                    if (Math.abs(det) > threshold) {
                        double u = (sumIyIy * (-sumIxIt) - sumIxIy * (-sumIyIt)) / det;
                        double v = (sumIxIx * (-sumIyIt) - sumIxIy * (-sumIxIt)) / det;
                        
                        flowX.set(y, x, u);
                        flowY.set(y, x, v);
                    }
                }
            }
            
            return new OpticalFlowResult(flowX, flowY);
        }
    }
    
    /**
     * Horn-Schunck光流算法 / Horn-Schunck Optical Flow Algorithm
     */
    public static class HornSchunckOpticalFlow implements IImageProcessor {
        
        private double alpha = 1.0;          // 平滑约束权重 / Smoothness constraint weight
        private int maxIterations = 100;     // 最大迭代次数 / Maximum iterations
        private double epsilon = 0.01;       // 收敛阈值 / Convergence threshold
        
        @Override
        public ImageData process(ImageData input) throws ImageProcessingException {
            throw new UnsupportedOperationException("Horn-Schunck requires two frames for optical flow calculation");
        }
        
        @Override
        public ImageData process(ImageData input, Map<String, Object> parameters) throws ImageProcessingException {
            throw new UnsupportedOperationException("Use calculateOpticalFlow method for Horn-Schunck optical flow");
        }
        
        /**
         * 计算两帧之间的稠密光流 / Calculate dense optical flow between two frames
         * 
         * @param frame1 第一帧 / First frame
         * @param frame2 第二帧 / Second frame
         * @return 光流结果 / Optical flow result
         * @throws ImageProcessingException 处理异常 / Processing exception
         */
        public OpticalFlowResult calculateOpticalFlow(ImageData frame1, ImageData frame2) throws ImageProcessingException {
            return calculateOpticalFlow(frame1, frame2, getDefaultParameters());
        }
        
        /**
         * 计算两帧之间的稠密光流 / Calculate dense optical flow between two frames
         * 
         * @param frame1 第一帧 / First frame
         * @param frame2 第二帧 / Second frame
         * @param parameters 参数 / Parameters
         * @return 光流结果 / Optical flow result
         * @throws ImageProcessingException 处理异常 / Processing exception
         */
        public OpticalFlowResult calculateOpticalFlow(ImageData frame1, ImageData frame2, Map<String, Object> parameters) 
                throws ImageProcessingException {
            
            if (frame1 == null || frame2 == null) {
                throw ImageProcessingException.invalidInput("Both frames must be provided");
            }
            
            if (frame1.getWidth() != frame2.getWidth() || frame1.getHeight() != frame2.getHeight()) {
                throw ImageProcessingException.invalidInput("Frames must have the same dimensions");
            }
            
            // 应用参数 / Apply parameters
            if (parameters != null) {
                alpha = (Double) parameters.getOrDefault("alpha", alpha);
                maxIterations = (Integer) parameters.getOrDefault("maxIterations", maxIterations);
                epsilon = (Double) parameters.getOrDefault("epsilon", epsilon);
            }
            
            // 转换为灰度图像 / Convert to grayscale
            ImageData gray1 = frame1.toGrayscale();
            ImageData gray2 = frame2.toGrayscale();
            
            IMatrix<Double> I1 = gray1.getChannel(0);
            IMatrix<Double> I2 = gray2.getChannel(0);
            
            // 计算图像梯度和时间导数 / Calculate image gradients and temporal derivative
            IMatrix<Double> Ix = calculateSobelX(I1);
            IMatrix<Double> Iy = calculateSobelY(I1);
            IMatrix<Double> It = calculateTemporalDerivative(I1, I2);
            
            // 计算Horn-Schunck光流 / Calculate Horn-Schunck optical flow
            return calculateHornSchunckFlow(Ix, Iy, It);
        }
        
        @Override
        public String getName() { return "HornSchunckOpticalFlow"; }
        
        @Override
        public String getDescription() { return "Horn-Schunck dense optical flow algorithm"; }
        
        @Override
        public java.util.Set<String> getSupportedParameters() {
            return java.util.Set.of("alpha", "maxIterations", "epsilon");
        }
        
        @Override
        public Map<String, Object> getDefaultParameters() {
            Map<String, Object> params = new java.util.HashMap<>();
            params.put("alpha", 1.0);
            params.put("maxIterations", 100);
            params.put("epsilon", 0.01);
            return params;
        }
        
        @Override
        public boolean validateInput(ImageData input) {
            return input != null;
        }
        
        @Override
        public boolean validateParameters(Map<String, Object> parameters) {
            if (parameters == null) return true;
            
            Double a = (Double) parameters.get("alpha");
            Integer mi = (Integer) parameters.get("maxIterations");
            Double e = (Double) parameters.get("epsilon");
            
            return (a == null || a > 0) && (mi == null || mi > 0) && (e == null || e > 0);
        }
        
        @Override
        public IImageProcessor clone() {
            HornSchunckOpticalFlow cloned = new HornSchunckOpticalFlow();
            cloned.alpha = this.alpha;
            cloned.maxIterations = this.maxIterations;
            cloned.epsilon = this.epsilon;
            return cloned;
        }
        
        // ========== 私有方法 / Private Methods ==========
        
        private IMatrix<Double> calculateSobelX(IMatrix<Double> image) {
            int height = image.getRowNum();
            int width = image.getColNum();
            IMatrix<Double> result = Linalg.zeros(height, width);
            
            // Sobel X核 / Sobel X kernel
            double[][] sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
            
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    double sum = 0.0;
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            sum += image.get(y + ky - 1, x + kx - 1) * sobelX[ky][kx];
                        }
                    }
                    result.set(y, x, sum / 8.0);
                }
            }
            
            return result;
        }
        
        private IMatrix<Double> calculateSobelY(IMatrix<Double> image) {
            int height = image.getRowNum();
            int width = image.getColNum();
            IMatrix<Double> result = Linalg.zeros(height, width);
            
            // Sobel Y核 / Sobel Y kernel
            double[][] sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
            
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    double sum = 0.0;
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            sum += image.get(y + ky - 1, x + kx - 1) * sobelY[ky][kx];
                        }
                    }
                    result.set(y, x, sum / 8.0);
                }
            }
            
            return result;
        }
        
        private IMatrix<Double> calculateTemporalDerivative(IMatrix<Double> I1, IMatrix<Double> I2) {
            return I2.sub(I1);
        }
        
        private OpticalFlowResult calculateHornSchunckFlow(IMatrix<Double> Ix, IMatrix<Double> Iy, IMatrix<Double> It) {
            int height = Ix.getRowNum();
            int width = Ix.getColNum();
            
            // 初始化光流场 / Initialize optical flow field
            IMatrix<Double> u = Linalg.zeros(height, width);
            IMatrix<Double> v = Linalg.zeros(height, width);
            
            // 迭代求解 / Iterative solution
            for (int iter = 0; iter < maxIterations; iter++) {
                IMatrix<Double> uNew = Linalg.zeros(height, width);
                IMatrix<Double> vNew = Linalg.zeros(height, width);
                
                double maxChange = 0.0;
                
                for (int y = 1; y < height - 1; y++) {
                    for (int x = 1; x < width - 1; x++) {
                        
                        // 计算邻域平均值 / Calculate neighborhood average
                        double uAvg = (u.get(y-1, x) + u.get(y+1, x) + u.get(y, x-1) + u.get(y, x+1)) / 4.0;
                        double vAvg = (v.get(y-1, x) + v.get(y+1, x) + v.get(y, x-1) + v.get(y, x+1)) / 4.0;
                        
                        double ix = Ix.get(y, x);
                        double iy = Iy.get(y, x);
                        double it = It.get(y, x);
                        
                        // Horn-Schunck更新公式 / Horn-Schunck update formula
                        double denominator = alpha * alpha + ix * ix + iy * iy;
                        if (denominator > 1e-10) {
                            double factor = (ix * uAvg + iy * vAvg + it) / denominator;
                            
                            double uUpdate = uAvg - ix * factor;
                            double vUpdate = vAvg - iy * factor;
                            
                            uNew.set(y, x, uUpdate);
                            vNew.set(y, x, vUpdate);
                            
                            // 计算变化量 / Calculate change
                            double change = Math.abs(uUpdate - u.get(y, x)) + Math.abs(vUpdate - v.get(y, x));
                            if (change > maxChange) {
                                maxChange = change;
                            }
                        } else {
                            uNew.set(y, x, uAvg);
                            vNew.set(y, x, vAvg);
                        }
                    }
                }
                
                u = uNew;
                v = vNew;
                
                // 检查收敛 / Check convergence
                if (maxChange < epsilon) {
                    break;
                }
            }
            
            return new OpticalFlowResult(u, v);
        }
    }
}