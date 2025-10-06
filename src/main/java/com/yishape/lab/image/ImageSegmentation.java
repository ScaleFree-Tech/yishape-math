package com.yishape.lab.image;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 图像分割类 / Image Segmentation Class
 * <p>
 * 提供各种图像分割功能，包括阈值分割、区域生长、K-means聚类、分水岭算法等。
 * 充分利用现有的统计学功能和线性代数功能。
 * </p>
 * <p>
 * Provides various image segmentation functionality including thresholding, region growing,
 * K-means clustering, watershed algorithm, etc. Fully utilizes existing statistical and linear algebra functionality.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ImageSegmentation {
    
    /**
     * 分割方法枚举 / Segmentation Method Enum
     */
    public enum SegmentationMethod {
        THRESHOLD,          // 阈值分割 / Thresholding
        OTSU,               // Otsu阈值分割 / Otsu thresholding
        ADAPTIVE,           // 自适应阈值分割 / Adaptive thresholding
        REGION_GROWING,     // 区域生长 / Region growing
        KMEANS,             // K-means聚类 / K-means clustering
        WATERSHED,          // 分水岭算法 / Watershed algorithm
        MEAN_SHIFT,         // Mean Shift聚类 / Mean shift clustering
        GRAB_CUT            // GrabCut算法 / GrabCut algorithm
    }
    
    /**
     * 阈值类型枚举 / Threshold Type Enum
     */
    public enum ThresholdType {
        BINARY,             // 二值化 / Binary
        BINARY_INV,         // 反二值化 / Binary inverse
        TRUNC,              // 截断 / Truncate
        TOZERO,             // 零化 / To zero
        TOZERO_INV          // 反零化 / To zero inverse
    }
    
    /**
     * 分割结果类 / Segmentation Result Class
     */
    public static class SegmentationResult {
        private ImageData segmentedImage;    // 分割后的图像 / Segmented image
        private int numRegions;              // 区域数量 / Number of regions
        private double[] regionLabels;       // 区域标签 / Region labels
        private double threshold;            // 使用的阈值 / Used threshold
        
        public SegmentationResult(ImageData segmentedImage, int numRegions, double[] regionLabels, double threshold) {
            this.segmentedImage = segmentedImage;
            this.numRegions = numRegions;
            this.regionLabels = regionLabels;
            this.threshold = threshold;
        }
        
        public ImageData getSegmentedImage() { return segmentedImage; }
        public int getNumRegions() { return numRegions; }
        public double[] getRegionLabels() { return regionLabels; }
        public double getThreshold() { return threshold; }
    }
    
    /**
     * 阈值分割 / Threshold Segmentation
     * <p>
     * 使用固定阈值进行图像分割。
     * Performs image segmentation using fixed threshold.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param threshold 阈值 / Threshold
     * @param thresholdType 阈值类型 / Threshold type
     * @return 分割结果 / Segmentation result
     */
    public static SegmentationResult thresholdSegmentation(ImageData image, double threshold, ThresholdType thresholdType) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        
        IMatrix<Double> segmented = Linalg.zeros(channel.getRowNum(), channel.getColNum());
        
        // 应用阈值 / Apply threshold
        for (int y = 0; y < channel.getRowNum(); y++) {
            for (int x = 0; x < channel.getColNum(); x++) {
                double pixelValue = channel.get(y, x);
                double newValue = applyThreshold(pixelValue, threshold, thresholdType);
                segmented.set(y, x, newValue);
            }
        }
        
        ImageData segmentedImage = new ImageData(segmented, grayscale.getWidth(), grayscale.getHeight(), 1,
                                               ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
        
        // 计算区域数量 / Calculate number of regions
        int numRegions = countRegions(segmented);
        
        return new SegmentationResult(segmentedImage, numRegions, null, threshold);
    }
    
    /**
     * Otsu阈值分割 / Otsu Threshold Segmentation
     * <p>
     * 使用Otsu算法自动确定最优阈值进行图像分割。
     * Uses Otsu algorithm to automatically determine optimal threshold for image segmentation.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @return 分割结果 / Segmentation result
     */
    public static SegmentationResult otsuSegmentation(ImageData image) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        
        // 计算Otsu阈值 / Calculate Otsu threshold
        double otsuThreshold = calculateOtsuThreshold(channel);
        
        // 应用阈值分割 / Apply threshold segmentation
        return thresholdSegmentation(image, otsuThreshold, ThresholdType.BINARY);
    }
    
    /**
     * 自适应阈值分割 / Adaptive Threshold Segmentation
     * <p>
     * 使用自适应阈值进行图像分割，适用于光照不均匀的图像。
     * Uses adaptive threshold for image segmentation, suitable for images with uneven illumination.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param blockSize 块大小 / Block size
     * @param constant 常数 / Constant
     * @return 分割结果 / Segmentation result
     */
    public static SegmentationResult adaptiveThresholdSegmentation(ImageData image, int blockSize, double constant) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (blockSize <= 0 || blockSize % 2 == 0) {
            throw new IllegalArgumentException("块大小必须为正奇数 / Block size must be positive odd number");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        
        int height = channel.getRowNum();
        int width = channel.getColNum();
        IMatrix<Double> segmented = Linalg.zeros(height, width);
        
        int halfBlock = blockSize / 2;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 计算局部均值 / Calculate local mean
                double localMean = calculateLocalMean(channel, x, y, halfBlock, height, width);
                
                // 应用自适应阈值 / Apply adaptive threshold
                double pixelValue = channel.get(y, x);
                double threshold = localMean - constant;
                segmented.set(y, x, pixelValue > threshold ? 1.0 : 0.0);
            }
        }
        
        ImageData segmentedImage = new ImageData(segmented, grayscale.getWidth(), grayscale.getHeight(), 1,
                                               ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
        
        int numRegions = countRegions(segmented);
        
        return new SegmentationResult(segmentedImage, numRegions, null, 0.0);
    }
    
    /**
     * 区域生长分割 / Region Growing Segmentation
     * <p>
     * 使用区域生长算法进行图像分割。
     * Performs image segmentation using region growing algorithm.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param seedPoints 种子点 / Seed points
     * @param threshold 生长阈值 / Growing threshold
     * @return 分割结果 / Segmentation result
     */
    public static SegmentationResult regionGrowingSegmentation(ImageData image, int[][] seedPoints, double threshold) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (seedPoints == null || seedPoints.length == 0) {
            throw new IllegalArgumentException("种子点不能为空 / Seed points cannot be empty");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        
        int height = channel.getRowNum();
        int width = channel.getColNum();
        IMatrix<Double> segmented = Linalg.zeros(height, width);
        boolean[][] visited = new boolean[height][width];
        
        // 对每个种子点进行区域生长 / Perform region growing for each seed point
        int regionLabel = 1;
        for (int[] seed : seedPoints) {
            int x = seed[0];
            int y = seed[1];
            
            if (x >= 0 && x < width && y >= 0 && y < height && !visited[y][x]) {
                growRegion(channel, segmented, visited, x, y, threshold, regionLabel);
                regionLabel++;
            }
        }
        
        ImageData segmentedImage = new ImageData(segmented, grayscale.getWidth(), grayscale.getHeight(), 1,
                                               ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
        
        int numRegions = regionLabel - 1;
        
        return new SegmentationResult(segmentedImage, numRegions, null, threshold);
    }
    
    /**
     * K-means聚类分割 / K-means Clustering Segmentation
     * <p>
     * 使用K-means聚类算法进行图像分割。
     * Performs image segmentation using K-means clustering algorithm.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param k 聚类数量 / Number of clusters
     * @param maxIterations 最大迭代次数 / Maximum iterations
     * @return 分割结果 / Segmentation result
     */
    public static SegmentationResult kmeansSegmentation(ImageData image, int k, int maxIterations) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (k <= 0) {
            throw new IllegalArgumentException("聚类数量必须大于0 / Number of clusters must be greater than 0");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        
        int height = channel.getRowNum();
        int width = channel.getColNum();
        
        // 准备数据 / Prepare data
        IVector<Double> data = channel.flatten();
        
        // 执行K-means聚类 / Perform K-means clustering
        KMeansResult kmeansResult = performKMeans(data, k, maxIterations);
        
        // 创建分割图像 / Create segmented image
        IMatrix<Double> segmented = Linalg.zeros(height, width);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                segmented.set(y, x, (double) kmeansResult.labels[index]);
            }
        }
        
        ImageData segmentedImage = new ImageData(segmented, grayscale.getWidth(), grayscale.getHeight(), 1,
                                               ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
        
        return new SegmentationResult(segmentedImage, k, null, 0.0);
    }
    
    /**
     * 分水岭分割 / Watershed Segmentation
     * <p>
     * 使用分水岭算法进行图像分割。
     * Performs image segmentation using watershed algorithm.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param markers 标记图像 / Marker image
     * @return 分割结果 / Segmentation result
     */
    public static SegmentationResult watershedSegmentation(ImageData image, ImageData markers) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (markers == null) {
            throw new IllegalArgumentException("标记图像不能为null / Marker image cannot be null");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        IMatrix<Double> markerChannel = markers.toGrayscale().getChannel(0);
        
        // 简化的分水岭算法实现 / Simplified watershed algorithm implementation
        IMatrix<Double> segmented = performWatershed(channel, markerChannel);
        
        ImageData segmentedImage = new ImageData(segmented, grayscale.getWidth(), grayscale.getHeight(), 1,
                                               ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
        
        // 计算区域数量 / Calculate number of regions
        int numRegions = countRegions(segmented);
        
        return new SegmentationResult(segmentedImage, numRegions, null, 0.0);
    }
    
    /**
     * 边缘检测分割 / Edge Detection Segmentation
     * <p>
     * 使用边缘检测进行图像分割。
     * Performs image segmentation using edge detection.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param lowThreshold 低阈值 / Low threshold
     * @param highThreshold 高阈值 / High threshold
     * @return 分割结果 / Segmentation result
     */
    public static SegmentationResult edgeDetectionSegmentation(ImageData image, double lowThreshold, double highThreshold) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        // 使用Sobel边缘检测 / Use Sobel edge detection
        ImageData edgeImage = ImageFilter.sobelEdgeDetection(image, ImageFilter.EdgeDirection.ALL_DIRECTIONS);
        IMatrix<Double> edgeChannel = edgeImage.getChannel(0);
        
        int height = edgeChannel.getRowNum();
        int width = edgeChannel.getColNum();
        IMatrix<Double> segmented = Linalg.zeros(height, width);
        
        // 应用双阈值 / Apply double threshold
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double edgeValue = edgeChannel.get(y, x);
                if (edgeValue > highThreshold) {
                    segmented.set(y, x, 1.0);  // 强边缘 / Strong edge
                } else if (edgeValue > lowThreshold) {
                    segmented.set(y, x, 0.5);  // 弱边缘 / Weak edge
                } else {
                    segmented.set(y, x, 0.0);  // 非边缘 / Non-edge
                }
            }
        }
        
        ImageData segmentedImage = new ImageData(segmented, image.getWidth(), image.getHeight(), 1,
                                               ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
        
        int numRegions = countRegions(segmented);
        
        return new SegmentationResult(segmentedImage, numRegions, null, (lowThreshold + highThreshold) / 2);
    }
    
    // ========== 辅助方法 / Helper Methods ==========
    
    /**
     * 应用阈值 / Apply threshold
     */
    private static double applyThreshold(double pixelValue, double threshold, ThresholdType thresholdType) {
        switch (thresholdType) {
            case BINARY:
                return pixelValue > threshold ? 1.0 : 0.0;
            case BINARY_INV:
                return pixelValue > threshold ? 0.0 : 1.0;
            case TRUNC:
                return Math.min(pixelValue, threshold);
            case TOZERO:
                return pixelValue > threshold ? pixelValue : 0.0;
            case TOZERO_INV:
                return pixelValue > threshold ? 0.0 : pixelValue;
            default:
                return pixelValue > threshold ? 1.0 : 0.0;
        }
    }
    
    /**
     * 计算Otsu阈值 / Calculate Otsu threshold
     */
    private static double calculateOtsuThreshold(IMatrix<Double> image) {
        // 计算直方图 / Calculate histogram
        int bins = 256;
        IVector<Double> histogram = Linalg.zeros(bins);
        
        double min = image.min();
        double max = image.max();
        double range = max - min;
        
        if (range > 0) {
            for (int y = 0; y < image.getRowNum(); y++) {
                for (int x = 0; x < image.getColNum(); x++) {
                    double normalized = (image.get(y, x) - min) / range;
                    int bin = Math.min((int) (normalized * (bins - 1)), bins - 1);
                    histogram.set(bin, histogram.get(bin) + 1);
                }
            }
        }
        
        // 归一化直方图 / Normalize histogram
        double totalPixels = histogram.sum();
        if (totalPixels > 0) {
            histogram = histogram.multiplyScalar(1.0 / totalPixels);
        }
        
        // 计算Otsu阈值 / Calculate Otsu threshold
        double bestThreshold = 0;
        double maxVariance = 0;
        
        for (int t = 0; t < bins; t++) {
            // 计算前景和背景概率 / Calculate foreground and background probabilities
            double w0 = 0, w1 = 0;
            double u0 = 0, u1 = 0;
            
            for (int i = 0; i <= t; i++) {
                w0 += histogram.get(i);
                u0 += i * histogram.get(i);
            }
            
            for (int i = t + 1; i < bins; i++) {
                w1 += histogram.get(i);
                u1 += i * histogram.get(i);
            }
            
            if (w0 > 0) u0 /= w0;
            if (w1 > 0) u1 /= w1;
            
            // 计算类间方差 / Calculate between-class variance
            double variance = w0 * w1 * (u0 - u1) * (u0 - u1);
            
            if (variance > maxVariance) {
                maxVariance = variance;
                bestThreshold = t;
            }
        }
        
        // 转换回原始值域 / Convert back to original value range
        return min + (bestThreshold / (double) (bins - 1)) * range;
    }
    
    /**
     * 计算局部均值 / Calculate local mean
     */
    private static double calculateLocalMean(IMatrix<Double> image, int x, int y, int halfBlock, int height, int width) {
        double sum = 0.0;
        int count = 0;
        
        for (int dy = -halfBlock; dy <= halfBlock; dy++) {
            for (int dx = -halfBlock; dx <= halfBlock; dx++) {
                int ny = y + dy;
                int nx = x + dx;
                
                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    sum += image.get(ny, nx);
                    count++;
                }
            }
        }
        
        return count > 0 ? sum / count : 0.0;
    }
    
    /**
     * 区域生长 / Region growing
     */
    private static void growRegion(IMatrix<Double> image, IMatrix<Double> segmented, boolean[][] visited,
                                 int startX, int startY, double threshold, int regionLabel) {
        int height = image.getRowNum();
        int width = image.getColNum();
        double seedValue = image.get(startY, startX);
        
        java.util.Queue<int[]> queue = new java.util.LinkedList<>();
        queue.offer(new int[]{startX, startY});
        visited[startY][startX] = true;
        segmented.set(startY, startX, (double) regionLabel);
        
        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            int x = current[0];
            int y = current[1];
            
            for (int[] dir : directions) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height && !visited[ny][nx]) {
                    double pixelValue = image.get(ny, nx);
                    if (Math.abs(pixelValue - seedValue) <= threshold) {
                        visited[ny][nx] = true;
                        segmented.set(ny, nx, (double) regionLabel);
                        queue.offer(new int[]{nx, ny});
                    }
                }
            }
        }
    }
    
    /**
     * K-means聚类结果类 / K-means Clustering Result Class
     */
    private static class KMeansResult {
        public int[] labels;
        
        public KMeansResult(int[] labels) {
            this.labels = labels;
        }
    }
    
    /**
     * 执行K-means聚类 / Perform K-means clustering
     */
    private static KMeansResult performKMeans(IVector<Double> data, int k, int maxIterations) {
        int n = data.length();
        int[] labels = new int[n];
        double[] centroids = new double[k];
        
        // 初始化质心 / Initialize centroids
        double min = data.min();
        double max = data.max();
        for (int i = 0; i < k; i++) {
            centroids[i] = min + (max - min) * i / (k - 1);
        }
        
        // 迭代优化 / Iterative optimization
        for (int iter = 0; iter < maxIterations; iter++) {
            // 分配数据点到最近的质心 / Assign data points to nearest centroids
            for (int i = 0; i < n; i++) {
                double value = data.get(i);
                int bestCluster = 0;
                double minDistance = Math.abs(value - centroids[0]);
                
                for (int j = 1; j < k; j++) {
                    double distance = Math.abs(value - centroids[j]);
                    if (distance < minDistance) {
                        minDistance = distance;
                        bestCluster = j;
                    }
                }
                labels[i] = bestCluster;
            }
            
            // 更新质心 / Update centroids
            double[] newCentroids = new double[k];
            int[] counts = new int[k];
            
            for (int i = 0; i < n; i++) {
                int cluster = labels[i];
                newCentroids[cluster] += data.get(i);
                counts[cluster]++;
            }
            
            boolean converged = true;
            for (int j = 0; j < k; j++) {
                if (counts[j] > 0) {
                    newCentroids[j] /= counts[j];
                    if (Math.abs(newCentroids[j] - centroids[j]) > 1e-6) {
                        converged = false;
                    }
                }
                centroids[j] = newCentroids[j];
            }
            
            if (converged) break;
        }
        
        return new KMeansResult(labels);
    }
    
    /**
     * 执行分水岭算法 / Perform watershed algorithm
     */
    private static IMatrix<Double> performWatershed(IMatrix<Double> image, IMatrix<Double> markers) {
        int height = image.getRowNum();
        int width = image.getColNum();
        IMatrix<Double> segmented = markers.copy();
        
        // 简化的分水岭实现 / Simplified watershed implementation
        // 这里使用基于标记的分水岭算法 / Here uses marker-based watershed algorithm
        
        boolean[][] processed = new boolean[height][width];
        java.util.Queue<int[]> queue = new java.util.LinkedList<>();
        
        // 初始化队列 / Initialize queue
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (markers.get(y, x) > 0) {
                    queue.offer(new int[]{x, y});
                    processed[y][x] = true;
                }
            }
        }
        
        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            int x = current[0];
            int y = current[1];
            double currentLabel = segmented.get(y, x);
            
            for (int[] dir : directions) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height && !processed[ny][nx]) {
                    segmented.set(ny, nx, currentLabel);
                    processed[ny][nx] = true;
                    queue.offer(new int[]{nx, ny});
                }
            }
        }
        
        return segmented;
    }
    
    /**
     * 计算区域数量 / Count number of regions
     */
    private static int countRegions(IMatrix<Double> segmented) {
        int height = segmented.getRowNum();
        int width = segmented.getColNum();
        java.util.Set<Double> uniqueLabels = new java.util.HashSet<>();
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double label = segmented.get(y, x);
                if (label > 0) {
                    uniqueLabels.add(label);
                }
            }
        }
        
        return uniqueLabels.size();
    }
}
