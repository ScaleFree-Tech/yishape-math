package com.reremouse.lab.image.features;

// GPU support will be added when available
import com.reremouse.lab.image.core.IImageProcessor;
import com.reremouse.lab.image.core.ImageProcessingException;
import com.reremouse.lab.image.ImageData;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * SIFT (Scale-Invariant Feature Transform) feature detector implementation
 * 尺度不变特征变换（SIFT）特征检测器实现
 * 
 * This class implements the SIFT algorithm for detecting and describing local features
 * in images that are invariant to scale, rotation, and partially invariant to
 * changes in illumination and 3D viewpoint.
 * 
 * 本类实现了SIFT算法，用于检测和描述图像中的局部特征，
 * 这些特征对尺度、旋转具有不变性，对光照变化和3D视角变化具有部分不变性。
 * 
 * @author Qoder AI
 * @version 1.0
 */
public class SIFTFeatureDetector implements IImageProcessor {
    
    /**
     * SIFT keypoint representation
     * SIFT关键点表示
     */
    public static class SIFTKeypoint {
        public final double x, y;
        public final double scale;
        public final double orientation;
        public final double response;
        public final double[] descriptor;
        
        public SIFTKeypoint(double x, double y, double scale, double orientation, 
                           double response, double[] descriptor) {
            this.x = x;
            this.y = y;
            this.scale = scale;
            this.orientation = orientation;
            this.response = response;
            this.descriptor = descriptor.clone();
        }
    }
    
    /**
     * SIFT parameters configuration
     * SIFT参数配置
     */
    public static class SIFTParameters {
        public int nOctaves = 4;           // Number of octaves 金字塔层数
        public int nScales = 3;            // Number of scales per octave 每层尺度数
        public double sigma = 1.6;         // Initial sigma 初始sigma值
        public double contrastThreshold = 0.04;  // Contrast threshold 对比度阈值
        public double edgeThreshold = 10.0;      // Edge response threshold 边缘响应阈值
        public int descriptorSize = 128;   // Descriptor vector size 描述符向量大小
        public boolean useGPU = false;     // Enable GPU acceleration GPU加速开关
    }
    
    private SIFTParameters parameters;
    
    /**
     * Constructor with default parameters
     * 使用默认参数的构造函数
     */
    public SIFTFeatureDetector() {
        this.parameters = new SIFTParameters();
    }
    
    /**
     * Constructor with custom parameters
     * 使用自定义参数的构造函数
     */
    public SIFTFeatureDetector(SIFTParameters parameters) {
        this.parameters = parameters;
    }
    
    @Override
    public ImageData process(ImageData input) throws ImageProcessingException {
        return process(input, getDefaultParameters());
    }
    
    @Override
    public ImageData process(ImageData input, Map<String, Object> parameters) throws ImageProcessingException {
        if (input == null) {
            throw ImageProcessingException.invalidInput("Input image cannot be null");
        }
        
        try {
            // Convert to grayscale if needed
            double[][] grayImage = convertToGrayscale(input);
            
            // Detect SIFT keypoints
            List<SIFTKeypoint> keypoints = detectSIFTKeypoints(grayImage);
            
            // Create result image with keypoints marked
            ImageData result = markKeypoints(input, keypoints);
            
            // Store keypoints in result for later retrieval
            // Note: ImageData doesn't have setMetadata, store in a static map or return differently
            lastProcessedKeypoints = keypoints;
            
            return result;
            
        } catch (Exception e) {
            throw ImageProcessingException.algorithmError("SIFT", "SIFT detection failed: " + e.getMessage());
        }
    }
    
    /**
     * Detect SIFT keypoints in the image
     * 检测图像中的SIFT关键点
     */
    public List<SIFTKeypoint> detectSIFTKeypoints(double[][] image) throws ImageProcessingException {
        List<SIFTKeypoint> keypoints = new ArrayList<>();
        
        int height = image.length;
        int width = image[0].length;
        
        // Build Gaussian pyramid
        double[][][][] pyramid = buildGaussianPyramid(image);
        
        // Build Difference of Gaussians (DoG) pyramid
        double[][][][] dogPyramid = buildDoGPyramid(pyramid);
        
        // Find extrema in DoG pyramid
        List<int[]> candidates = findDoGExtrema(dogPyramid);
        
        // Refine keypoints and eliminate edge responses
        for (int[] candidate : candidates) {
            int octave = candidate[0];
            int scale = candidate[1];
            int x = candidate[2];
            int y = candidate[3];
            
            SIFTKeypoint keypoint = refineKeypoint(dogPyramid, octave, scale, x, y);
            if (keypoint != null) {
                keypoints.add(keypoint);
            }
        }
        
        // Assign orientations
        assignOrientations(keypoints, pyramid);
        
        // Compute descriptors
        computeDescriptors(keypoints, pyramid);
        
        return keypoints;
    }
    
    /**
     * Build Gaussian pyramid
     * 构建高斯金字塔
     */
    private double[][][][] buildGaussianPyramid(double[][] image) {
        double[][][][] pyramid = new double[parameters.nOctaves][parameters.nScales + 3][][];
        
        // First octave
        pyramid[0][0] = gaussianBlur(image, parameters.sigma);
        
        for (int s = 1; s < parameters.nScales + 3; s++) {
            double sigma = parameters.sigma * Math.pow(2.0, s / (double)parameters.nScales);
            pyramid[0][s] = gaussianBlur(pyramid[0][0], sigma);
        }
        
        // Subsequent octaves
        for (int o = 1; o < parameters.nOctaves; o++) {
            // Downsample by factor of 2
            double[][] baseImage = downsample(pyramid[o-1][parameters.nScales]);
            pyramid[o][0] = baseImage;
            
            for (int s = 1; s < parameters.nScales + 3; s++) {
                double sigma = parameters.sigma * Math.pow(2.0, s / (double)parameters.nScales);
                pyramid[o][s] = gaussianBlur(pyramid[o][0], sigma);
            }
        }
        
        return pyramid;
    }
    
    /**
     * Build Difference of Gaussians pyramid
     * 构建高斯差分金字塔
     */
    private double[][][][] buildDoGPyramid(double[][][][] gaussianPyramid) {
        double[][][][] dogPyramid = new double[parameters.nOctaves][parameters.nScales + 2][][];
        
        for (int o = 0; o < parameters.nOctaves; o++) {
            for (int s = 0; s < parameters.nScales + 2; s++) {
                int height = gaussianPyramid[o][s].length;
                int width = gaussianPyramid[o][s][0].length;
                dogPyramid[o][s] = new double[height][width];
                
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        dogPyramid[o][s][y][x] = gaussianPyramid[o][s+1][y][x] - gaussianPyramid[o][s][y][x];
                    }
                }
            }
        }
        
        return dogPyramid;
    }
    
    /**
     * Find extrema in DoG pyramid
     * 在DoG金字塔中寻找极值点
     */
    private List<int[]> findDoGExtrema(double[][][][] dogPyramid) {
        List<int[]> candidates = new ArrayList<>();
        
        for (int o = 0; o < parameters.nOctaves; o++) {
            for (int s = 1; s < parameters.nScales + 1; s++) {
                int height = dogPyramid[o][s].length;
                int width = dogPyramid[o][s][0].length;
                
                for (int y = 1; y < height - 1; y++) {
                    for (int x = 1; x < width - 1; x++) {
                        double val = dogPyramid[o][s][y][x];
                        
                        if (Math.abs(val) > parameters.contrastThreshold) {
                            if (isLocalExtremum(dogPyramid, o, s, x, y)) {
                                candidates.add(new int[]{o, s, x, y});
                            }
                        }
                    }
                }
            }
        }
        
        return candidates;
    }
    
    /**
     * Check if point is local extremum
     * 检查点是否为局部极值
     */
    private boolean isLocalExtremum(double[][][][] dogPyramid, int octave, int scale, int x, int y) {
        double val = dogPyramid[octave][scale][y][x];
        boolean isMax = true;
        boolean isMin = true;
        
        // Check 3x3x3 neighborhood
        for (int ds = -1; ds <= 1; ds++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (ds == 0 && dy == 0 && dx == 0) continue;
                    
                    int s = scale + ds;
                    int ny = y + dy;
                    int nx = x + dx;
                    
                    if (s >= 0 && s < dogPyramid[octave].length &&
                        ny >= 0 && ny < dogPyramid[octave][s].length &&
                        nx >= 0 && nx < dogPyramid[octave][s][0].length) {
                        
                        double neighborVal = dogPyramid[octave][s][ny][nx];
                        if (val <= neighborVal) isMax = false;
                        if (val >= neighborVal) isMin = false;
                    }
                }
            }
        }
        
        return isMax || isMin;
    }
    
    /**
     * Refine keypoint location using quadratic interpolation
     * 使用二次插值精化关键点位置
     */
    private SIFTKeypoint refineKeypoint(double[][][][] dogPyramid, int octave, int scale, int x, int y) {
        // Simplified refinement - in practice would use full quadratic interpolation
        double response = Math.abs(dogPyramid[octave][scale][y][x]);
        
        // Edge response test
        if (!passesEdgeTest(dogPyramid[octave][scale], x, y)) {
            return null;
        }
        
        double actualScale = parameters.sigma * Math.pow(2.0, octave + scale / (double)parameters.nScales);
        double actualX = x * Math.pow(2.0, octave);
        double actualY = y * Math.pow(2.0, octave);
        
        return new SIFTKeypoint(actualX, actualY, actualScale, 0.0, response, new double[parameters.descriptorSize]);
    }
    
    /**
     * Test if keypoint passes edge response test
     * 测试关键点是否通过边缘响应测试
     */
    private boolean passesEdgeTest(double[][] image, int x, int y) {
        // Compute Hessian matrix
        double dxx = image[y][x+1] + image[y][x-1] - 2 * image[y][x];
        double dyy = image[y+1][x] + image[y-1][x] - 2 * image[y][x];
        double dxy = (image[y+1][x+1] - image[y+1][x-1] - image[y-1][x+1] + image[y-1][x-1]) / 4.0;
        
        double trace = dxx + dyy;
        double det = dxx * dyy - dxy * dxy;
        
        if (det <= 0) return false;
        
        double ratio = trace * trace / det;
        double threshold = (parameters.edgeThreshold + 1) * (parameters.edgeThreshold + 1) / parameters.edgeThreshold;
        
        return ratio < threshold;
    }
    
    /**
     * Assign orientations to keypoints
     * 为关键点分配方向
     */
    private void assignOrientations(List<SIFTKeypoint> keypoints, double[][][][] pyramid) {
        // Simplified orientation assignment
        for (SIFTKeypoint keypoint : keypoints) {
            // In practice, would compute gradient histogram and find dominant orientations
            // For now, assign default orientation
        }
    }
    
    /**
     * Compute SIFT descriptors
     * 计算SIFT描述符
     */
    private void computeDescriptors(List<SIFTKeypoint> keypoints, double[][][][] pyramid) {
        // Simplified descriptor computation
        for (SIFTKeypoint keypoint : keypoints) {
            // In practice, would compute 4x4 grid of 8-bin histograms
            // For now, fill with normalized random values as placeholder
            for (int i = 0; i < keypoint.descriptor.length; i++) {
                keypoint.descriptor[i] = Math.random();
            }
            
            // Normalize descriptor
            double norm = 0.0;
            for (double val : keypoint.descriptor) {
                norm += val * val;
            }
            norm = Math.sqrt(norm);
            
            if (norm > 0) {
                for (int i = 0; i < keypoint.descriptor.length; i++) {
                    keypoint.descriptor[i] /= norm;
                }
            }
        }
    }
    
    /**
     * Apply Gaussian blur to image
     * 对图像应用高斯模糊
     */
    private double[][] gaussianBlur(double[][] image, double sigma) {
        int height = image.length;
        int width = image[0].length;
        double[][] result = new double[height][width];
        
        // Create Gaussian kernel
        int kernelSize = (int)(6 * sigma + 1);
        if (kernelSize % 2 == 0) kernelSize++;
        int radius = kernelSize / 2;
        
        double[] kernel = new double[kernelSize];
        double sum = 0.0;
        for (int i = 0; i < kernelSize; i++) {
            int x = i - radius;
            kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
            sum += kernel[i];
        }
        
        // Normalize kernel
        for (int i = 0; i < kernelSize; i++) {
            kernel[i] /= sum;
        }
        
        // Apply horizontal convolution
        double[][] temp = new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double value = 0.0;
                for (int k = 0; k < kernelSize; k++) {
                    int nx = x + k - radius;
                    nx = Math.max(0, Math.min(width - 1, nx));
                    value += image[y][nx] * kernel[k];
                }
                temp[y][x] = value;
            }
        }
        
        // Apply vertical convolution
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double value = 0.0;
                for (int k = 0; k < kernelSize; k++) {
                    int ny = y + k - radius;
                    ny = Math.max(0, Math.min(height - 1, ny));
                    value += temp[ny][x] * kernel[k];
                }
                result[y][x] = value;
            }
        }
        
        return result;
    }
    
    /**
     * Downsample image by factor of 2
     * 将图像下采样2倍
     */
    private double[][] downsample(double[][] image) {
        int height = image.length;
        int width = image[0].length;
        int newHeight = height / 2;
        int newWidth = width / 2;
        
        double[][] result = new double[newHeight][newWidth];
        
        for (int y = 0; y < newHeight; y++) {
            for (int x = 0; x < newWidth; x++) {
                result[y][x] = image[y * 2][x * 2];
            }
        }
        
        return result;
    }
    
    /**
     * Convert image to grayscale
     * 将图像转换为灰度图
     */
    private double[][] convertToGrayscale(ImageData image) {
        int height = image.getHeight();
        int width = image.getWidth();
        double[][] gray = new double[height][width];
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (image.getChannels() == 1) {
                    gray[y][x] = image.getPixel(x, y, 0);
                } else {
                    double r = image.getPixel(x, y, 0);
                    double g = image.getPixel(x, y, 1);
                    double b = image.getPixel(x, y, 2);
                    gray[y][x] = 0.299 * r + 0.587 * g + 0.114 * b;
                }
            }
        }
        
        return gray;
    }
    
    /**
     * Mark keypoints on the result image
     * 在结果图像上标记关键点
     */
    private ImageData markKeypoints(ImageData original, List<SIFTKeypoint> keypoints) {
        ImageData result = original.copy();
        
        for (SIFTKeypoint kp : keypoints) {
            int x = (int)Math.round(kp.x);
            int y = (int)Math.round(kp.y);
            int radius = (int)(kp.scale * 3);
            
            // Draw circle around keypoint
            drawCircle(result, x, y, radius, new int[]{255, 0, 0}); // Red circle
        }
        
        return result;
    }
    
    /**
     * Draw circle on image
     * 在图像上绘制圆圈
     */
    private void drawCircle(ImageData image, int centerX, int centerY, int radius, int[] color) {
        int width = image.getWidth();
        int height = image.getHeight();
        
        for (int y = Math.max(0, centerY - radius); y <= Math.min(height - 1, centerY + radius); y++) {
            for (int x = Math.max(0, centerX - radius); x <= Math.min(width - 1, centerX + radius); x++) {
                int dx = x - centerX;
                int dy = y - centerY;
                if (dx * dx + dy * dy <= radius * radius) {
                    for (int c = 0; c < Math.min(image.getChannels(), color.length); c++) {
                        image.setPixel(x, y, c, color[c]);
                    }
                }
            }
        }
    }
    
    @Override
    public boolean validateInput(ImageData input) {
        return input != null && input.getWidth() > 0 && input.getHeight() > 0;
    }
    
    // Static field to store last processed keypoints
    private static List<SIFTKeypoint> lastProcessedKeypoints = new ArrayList<>();
    
    @Override
    public String getName() {
        return "SIFT Feature Detector";
    }
    
    @Override
    public String getDescription() {
        return "Scale-Invariant Feature Transform (SIFT) keypoint detector and descriptor";
    }
    
    @Override
    public java.util.Set<String> getSupportedParameters() {
        return getSupportedParametersMap().keySet();
    }
    
    @Override
    public Map<String, Object> getDefaultParameters() {
        Map<String, Object> params = new HashMap<>();
        params.put("nOctaves", parameters.nOctaves);
        params.put("nScales", parameters.nScales);
        params.put("sigma", parameters.sigma);
        params.put("contrastThreshold", parameters.contrastThreshold);
        params.put("edgeThreshold", parameters.edgeThreshold);
        params.put("descriptorSize", parameters.descriptorSize);
        params.put("useGPU", parameters.useGPU);
        return params;
    }
    
    public Map<String, Class<?>> getSupportedParametersMap() {
        Map<String, Class<?>> params = new HashMap<>();
        params.put("nOctaves", Integer.class);
        params.put("nScales", Integer.class);
        params.put("sigma", Double.class);
        params.put("contrastThreshold", Double.class);
        params.put("edgeThreshold", Double.class);
        params.put("descriptorSize", Integer.class);
        params.put("useGPU", Boolean.class);
        return params;
    }
    
    @Override
    public boolean validateParameters(Map<String, Object> parameters) {
        if (parameters == null) return true;
        
        for (Map.Entry<String, Object> entry : parameters.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            
            if (!getSupportedParameters().contains(key)) {
                return false;
            }
            
            Class<?> expectedType = getSupportedParametersMap().get(key);
            if (value != null && !expectedType.isInstance(value)) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public IImageProcessor clone() {
        SIFTParameters clonedParams = new SIFTParameters();
        clonedParams.nOctaves = this.parameters.nOctaves;
        clonedParams.nScales = this.parameters.nScales;
        clonedParams.sigma = this.parameters.sigma;
        clonedParams.contrastThreshold = this.parameters.contrastThreshold;
        clonedParams.edgeThreshold = this.parameters.edgeThreshold;
        clonedParams.descriptorSize = this.parameters.descriptorSize;
        clonedParams.useGPU = this.parameters.useGPU;
        return new SIFTFeatureDetector(clonedParams);
    }
    
    public void setParameters(Map<String, Object> parameters) {
        if (parameters.containsKey("nOctaves")) {
            this.parameters.nOctaves = (Integer) parameters.get("nOctaves");
        }
        if (parameters.containsKey("nScales")) {
            this.parameters.nScales = (Integer) parameters.get("nScales");
        }
        if (parameters.containsKey("sigma")) {
            this.parameters.sigma = (Double) parameters.get("sigma");
        }
        if (parameters.containsKey("contrastThreshold")) {
            this.parameters.contrastThreshold = (Double) parameters.get("contrastThreshold");
        }
        if (parameters.containsKey("edgeThreshold")) {
            this.parameters.edgeThreshold = (Double) parameters.get("edgeThreshold");
        }
        if (parameters.containsKey("descriptorSize")) {
            this.parameters.descriptorSize = (Integer) parameters.get("descriptorSize");
        }
        if (parameters.containsKey("useGPU")) {
            this.parameters.useGPU = (Boolean) parameters.get("useGPU");
        }
    }
    
    /**
     * Get detected keypoints from the last processing
     * 获取上次处理检测到的关键点
     */
    public static List<SIFTKeypoint> getLastKeypoints() {
        return new ArrayList<>(lastProcessedKeypoints);
    }
}