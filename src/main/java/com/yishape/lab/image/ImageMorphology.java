package com.yishape.lab.image;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 图像形态学操作类 / Image Morphology Operations Class
 * <p>
 * 提供各种图像形态学操作功能，包括腐蚀、膨胀、开运算、闭运算、形态学梯度、顶帽、黑帽等。
 * 充分利用现有的线性代数功能。
 * </p>
 * <p>
 * Provides various image morphology operations including erosion, dilation, opening, closing,
 * morphological gradient, top hat, black hat, etc. Fully utilizes existing linear algebra functionality.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ImageMorphology {
    
    /**
     * 结构元素类型枚举 / Structuring Element Type Enum
     */
    public enum StructuringElementType {
        RECTANGLE,          // 矩形 / Rectangle
        ELLIPSE,            // 椭圆 / Ellipse
        CROSS,              // 十字形 / Cross
        DIAMOND,            // 菱形 / Diamond
        DISK,               // 圆形 / Disk
        CUSTOM              // 自定义 / Custom
    }
    
    /**
     * 形态学操作类型枚举 / Morphology Operation Type Enum
     */
    public enum MorphologyOperation {
        EROSION,            // 腐蚀 / Erosion
        DILATION,           // 膨胀 / Dilation
        OPENING,            // 开运算 / Opening
        CLOSING,            // 闭运算 / Closing
        GRADIENT,           // 形态学梯度 / Morphological gradient
        TOP_HAT,            // 顶帽 / Top hat
        BLACK_HAT,          // 黑帽 / Black hat
        HIT_OR_MISS         // 击中或击不中 / Hit or miss
    }
    
    /**
     * 结构元素类 / Structuring Element Class
     */
    public static class StructuringElement {
        private IMatrix<Double> kernel;      // 核矩阵 / Kernel matrix
        private int centerX;                 // 中心X坐标 / Center X coordinate
        private int centerY;                 // 中心Y坐标 / Center Y coordinate
        private StructuringElementType type; // 类型 / Type
        
        public StructuringElement(IMatrix<Double> kernel, int centerX, int centerY, StructuringElementType type) {
            this.kernel = kernel;
            this.centerX = centerX;
            this.centerY = centerY;
            this.type = type;
        }
        
        public IMatrix<Double> getKernel() { return kernel; }
        public int getCenterX() { return centerX; }
        public int getCenterY() { return centerY; }
        public StructuringElementType getType() { return type; }
    }
    
    /**
     * 创建矩形结构元素 / Create rectangular structuring element
     * 
     * @param width 宽度 / Width
     * @param height 高度 / Height
     * @return 结构元素 / Structuring element
     */
    public static StructuringElement createRectangle(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("宽度和高度必须大于0 / Width and height must be greater than 0");
        }
        
        IMatrix<Double> kernel = Linalg.ones(height, width);
        return new StructuringElement(kernel, width / 2, height / 2, StructuringElementType.RECTANGLE);
    }
    
    /**
     * 创建椭圆结构元素 / Create elliptical structuring element
     * 
     * @param width 宽度 / Width
     * @param height 高度 / Height
     * @return 结构元素 / Structuring element
     */
    public static StructuringElement createEllipse(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("宽度和高度必须大于0 / Width and height must be greater than 0");
        }
        
        IMatrix<Double> kernel = Linalg.zeros(height, width);
        double centerX = (width - 1) / 2.0;
        double centerY = (height - 1) / 2.0;
        double a = width / 2.0;
        double b = height / 2.0;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double dx = (x - centerX) / a;
                double dy = (y - centerY) / b;
                if (dx * dx + dy * dy <= 1.0) {
                    kernel.set(y, x, 1.0);
                }
            }
        }
        
        return new StructuringElement(kernel, width / 2, height / 2, StructuringElementType.ELLIPSE);
    }
    
    /**
     * 创建十字形结构元素 / Create cross structuring element
     * 
     * @param size 大小 / Size
     * @return 结构元素 / Structuring element
     */
    public static StructuringElement createCross(int size) {
        if (size <= 0 || size % 2 == 0) {
            throw new IllegalArgumentException("大小必须为正奇数 / Size must be positive odd number");
        }
        
        IMatrix<Double> kernel = Linalg.zeros(size, size);
        int center = size / 2;
        
        // 水平线 / Horizontal line
        for (int x = 0; x < size; x++) {
            kernel.set(center, x, 1.0);
        }
        
        // 垂直线 / Vertical line
        for (int y = 0; y < size; y++) {
            kernel.set(y, center, 1.0);
        }
        
        return new StructuringElement(kernel, center, center, StructuringElementType.CROSS);
    }
    
    /**
     * 创建菱形结构元素 / Create diamond structuring element
     * 
     * @param size 大小 / Size
     * @return 结构元素 / Structuring element
     */
    public static StructuringElement createDiamond(int size) {
        if (size <= 0 || size % 2 == 0) {
            throw new IllegalArgumentException("大小必须为正奇数 / Size must be positive odd number");
        }
        
        IMatrix<Double> kernel = Linalg.zeros(size, size);
        int center = size / 2;
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                if (Math.abs(x - center) + Math.abs(y - center) <= center) {
                    kernel.set(y, x, 1.0);
                }
            }
        }
        
        return new StructuringElement(kernel, center, center, StructuringElementType.DIAMOND);
    }
    
    /**
     * 创建圆形结构元素 / Create disk structuring element
     * 
     * @param radius 半径 / Radius
     * @return 结构元素 / Structuring element
     */
    public static StructuringElement createDisk(int radius) {
        if (radius <= 0) {
            throw new IllegalArgumentException("半径必须大于0 / Radius must be greater than 0");
        }
        
        int size = 2 * radius + 1;
        IMatrix<Double> kernel = Linalg.zeros(size, size);
        
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double dx = x - radius;
                double dy = y - radius;
                if (dx * dx + dy * dy <= radius * radius) {
                    kernel.set(y, x, 1.0);
                }
            }
        }
        
        return new StructuringElement(kernel, radius, radius, StructuringElementType.DISK);
    }
    
    /**
     * 腐蚀操作 / Erosion Operation
     * <p>
     * 对图像进行腐蚀操作，用于去除小的噪声和分离连接的对象。
     * Performs erosion operation on image, used to remove small noise and separate connected objects.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param se 结构元素 / Structuring element
     * @return 腐蚀后的图像 / Eroded image
     */
    public static ImageData erode(ImageData image, StructuringElement se) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (se == null) {
            throw new IllegalArgumentException("结构元素不能为null / Structuring element cannot be null");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        IMatrix<Double> kernel = se.getKernel();
        
        int height = channel.getRowNum();
        int width = channel.getColNum();
        int kernelHeight = kernel.getRowNum();
        int kernelWidth = kernel.getColNum();
        
        IMatrix<Double> result = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double minValue = Double.MAX_VALUE;
                boolean valid = true;
                
                for (int ky = 0; ky < kernelHeight && valid; ky++) {
                    for (int kx = 0; kx < kernelWidth && valid; kx++) {
                        if (kernel.get(ky, kx) > 0) {
                            int px = x + kx - se.getCenterX();
                            int py = y + ky - se.getCenterY();
                            
                            if (px >= 0 && px < width && py >= 0 && py < height) {
                                minValue = Math.min(minValue, channel.get(py, px));
                            } else {
                                valid = false;
                            }
                        }
                    }
                }
                
                result.set(y, x, valid ? minValue : 0.0);
            }
        }
        
        return new ImageData(result, grayscale.getWidth(), grayscale.getHeight(), 1,
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * 膨胀操作 / Dilation Operation
     * <p>
     * 对图像进行膨胀操作，用于填充小的空洞和连接断裂的对象。
     * Performs dilation operation on image, used to fill small holes and connect broken objects.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param se 结构元素 / Structuring element
     * @return 膨胀后的图像 / Dilated image
     */
    public static ImageData dilate(ImageData image, StructuringElement se) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (se == null) {
            throw new IllegalArgumentException("结构元素不能为null / Structuring element cannot be null");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        IMatrix<Double> kernel = se.getKernel();
        
        int height = channel.getRowNum();
        int width = channel.getColNum();
        int kernelHeight = kernel.getRowNum();
        int kernelWidth = kernel.getColNum();
        
        IMatrix<Double> result = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double maxValue = Double.MIN_VALUE;
                
                for (int ky = 0; ky < kernelHeight; ky++) {
                    for (int kx = 0; kx < kernelWidth; kx++) {
                        if (kernel.get(ky, kx) > 0) {
                            int px = x + kx - se.getCenterX();
                            int py = y + ky - se.getCenterY();
                            
                            if (px >= 0 && px < width && py >= 0 && py < height) {
                                maxValue = Math.max(maxValue, channel.get(py, px));
                            }
                        }
                    }
                }
                
                result.set(y, x, maxValue);
            }
        }
        
        return new ImageData(result, grayscale.getWidth(), grayscale.getHeight(), 1,
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * 开运算 / Opening Operation
     * <p>
     * 开运算是先腐蚀后膨胀的组合操作，用于去除小的噪声和分离连接的对象。
     * Opening is a combination of erosion followed by dilation, used to remove small noise and separate connected objects.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param se 结构元素 / Structuring element
     * @return 开运算后的图像 / Opened image
     */
    public static ImageData opening(ImageData image, StructuringElement se) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (se == null) {
            throw new IllegalArgumentException("结构元素不能为null / Structuring element cannot be null");
        }
        
        // 先腐蚀后膨胀 / Erosion followed by dilation
        ImageData eroded = erode(image, se);
        return dilate(eroded, se);
    }
    
    /**
     * 闭运算 / Closing Operation
     * <p>
     * 闭运算是先膨胀后腐蚀的组合操作，用于填充小的空洞和连接断裂的对象。
     * Closing is a combination of dilation followed by erosion, used to fill small holes and connect broken objects.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param se 结构元素 / Structuring element
     * @return 闭运算后的图像 / Closed image
     */
    public static ImageData closing(ImageData image, StructuringElement se) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (se == null) {
            throw new IllegalArgumentException("结构元素不能为null / Structuring element cannot be null");
        }
        
        // 先膨胀后腐蚀 / Dilation followed by erosion
        ImageData dilated = dilate(image, se);
        return erode(dilated, se);
    }
    
    /**
     * 形态学梯度 / Morphological Gradient
     * <p>
     * 形态学梯度是膨胀和腐蚀的差值，用于边缘检测。
     * Morphological gradient is the difference between dilation and erosion, used for edge detection.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param se 结构元素 / Structuring element
     * @return 形态学梯度图像 / Morphological gradient image
     */
    public static ImageData morphologicalGradient(ImageData image, StructuringElement se) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (se == null) {
            throw new IllegalArgumentException("结构元素不能为null / Structuring element cannot be null");
        }
        
        // 计算膨胀和腐蚀 / Calculate dilation and erosion
        ImageData dilated = dilate(image, se);
        ImageData eroded = erode(image, se);
        
        // 计算差值 / Calculate difference
        IMatrix<Double> dilatedChannel = dilated.getChannel(0);
        IMatrix<Double> erodedChannel = eroded.getChannel(0);
        IMatrix<Double> gradient = dilatedChannel.sub(erodedChannel);
        
        return new ImageData(gradient, image.getWidth(), image.getHeight(), 1,
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * 顶帽变换 / Top Hat Transform
     * <p>
     * 顶帽变换是原图像与开运算结果的差值，用于提取比结构元素小的亮特征。
     * Top hat transform is the difference between original image and opening result, used to extract bright features smaller than structuring element.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param se 结构元素 / Structuring element
     * @return 顶帽变换图像 / Top hat transform image
     */
    public static ImageData topHat(ImageData image, StructuringElement se) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (se == null) {
            throw new IllegalArgumentException("结构元素不能为null / Structuring element cannot be null");
        }
        
        // 计算开运算 / Calculate opening
        ImageData opened = opening(image, se);
        
        // 计算差值 / Calculate difference
        IMatrix<Double> originalChannel = image.toGrayscale().getChannel(0);
        IMatrix<Double> openedChannel = opened.getChannel(0);
        IMatrix<Double> topHat = originalChannel.sub(openedChannel);
        
        return new ImageData(topHat, image.getWidth(), image.getHeight(), 1,
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * 黑帽变换 / Black Hat Transform
     * <p>
     * 黑帽变换是闭运算结果与原图像的差值，用于提取比结构元素小的暗特征。
     * Black hat transform is the difference between closing result and original image, used to extract dark features smaller than structuring element.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param se 结构元素 / Structuring element
     * @return 黑帽变换图像 / Black hat transform image
     */
    public static ImageData blackHat(ImageData image, StructuringElement se) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (se == null) {
            throw new IllegalArgumentException("结构元素不能为null / Structuring element cannot be null");
        }
        
        // 计算闭运算 / Calculate closing
        ImageData closed = closing(image, se);
        
        // 计算差值 / Calculate difference
        IMatrix<Double> closedChannel = closed.getChannel(0);
        IMatrix<Double> originalChannel = image.toGrayscale().getChannel(0);
        IMatrix<Double> blackHat = closedChannel.sub(originalChannel);
        
        return new ImageData(blackHat, image.getWidth(), image.getHeight(), 1,
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * 击中或击不中变换 / Hit or Miss Transform
     * <p>
     * 击中或击不中变换用于检测特定的模式或形状。
     * Hit or miss transform is used to detect specific patterns or shapes.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param hitKernel 击中核 / Hit kernel
     * @param missKernel 击不中核 / Miss kernel
     * @return 击中或击不中变换图像 / Hit or miss transform image
     */
    public static ImageData hitOrMiss(ImageData image, IMatrix<Double> hitKernel, IMatrix<Double> missKernel) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        if (hitKernel == null || missKernel == null) {
            throw new IllegalArgumentException("核不能为null / Kernels cannot be null");
        }
        
        // 转换为二值图像 / Convert to binary image
        ImageData binary = image.toGrayscale();
        IMatrix<Double> channel = binary.getChannel(0);
        
        int height = channel.getRowNum();
        int width = channel.getColNum();
        int hitHeight = hitKernel.getRowNum();
        int hitWidth = hitKernel.getColNum();
        int missHeight = missKernel.getRowNum();
        int missWidth = missKernel.getColNum();
        
        IMatrix<Double> result = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                boolean hitMatch = true;
                boolean missMatch = true;
                
                // 检查击中条件 / Check hit condition
                for (int ky = 0; ky < hitHeight && hitMatch; ky++) {
                    for (int kx = 0; kx < hitWidth && hitMatch; kx++) {
                        if (hitKernel.get(ky, kx) > 0) {
                            int px = x + kx - hitWidth / 2;
                            int py = y + ky - hitHeight / 2;
                            
                            if (px >= 0 && px < width && py >= 0 && py < height) {
                                if (channel.get(py, px) <= 0.5) {
                                    hitMatch = false;
                                }
                            } else {
                                hitMatch = false;
                            }
                        }
                    }
                }
                
                // 检查击不中条件 / Check miss condition
                for (int ky = 0; ky < missHeight && missMatch; ky++) {
                    for (int kx = 0; kx < missWidth && missMatch; kx++) {
                        if (missKernel.get(ky, kx) > 0) {
                            int px = x + kx - missWidth / 2;
                            int py = y + ky - missHeight / 2;
                            
                            if (px >= 0 && px < width && py >= 0 && py < height) {
                                if (channel.get(py, px) > 0.5) {
                                    missMatch = false;
                                }
                            } else {
                                missMatch = false;
                            }
                        }
                    }
                }
                
                result.set(y, x, (hitMatch && missMatch) ? 1.0 : 0.0);
            }
        }
        
        return new ImageData(result, image.getWidth(), image.getHeight(), 1,
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * 骨架化 / Skeletonization
     * <p>
     * 骨架化用于提取对象的骨架结构。
     * Skeletonization is used to extract skeleton structure of objects.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @return 骨架化图像 / Skeletonized image
     */
    public static ImageData skeletonize(ImageData image) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        // 转换为二值图像 / Convert to binary image
        ImageData binary = image.toGrayscale();
        IMatrix<Double> channel = binary.getChannel(0);
        
        // 二值化 / Binarize
        IMatrix<Double> binaryChannel = channel.copy();
        for (int y = 0; y < binaryChannel.getRowNum(); y++) {
            for (int x = 0; x < binaryChannel.getColNum(); x++) {
                binaryChannel.set(y, x, binaryChannel.get(y, x) > 0.5 ? 1.0 : 0.0);
            }
        }
        
        IMatrix<Double> skeleton = binaryChannel.copy();
        boolean changed = true;
        int iteration = 0;
        int maxIterations = 1000;
        
        while (changed && iteration < maxIterations) {
            changed = false;
            IMatrix<Double> temp = skeleton.copy();
            
            // 细化算法 / Thinning algorithm
            for (int y = 1; y < skeleton.getRowNum() - 1; y++) {
                for (int x = 1; x < skeleton.getColNum() - 1; x++) {
                    if (skeleton.get(y, x) > 0) {
                        // 计算8邻域 / Calculate 8-neighborhood
                        double[] neighbors = {
                            skeleton.get(y-1, x-1), skeleton.get(y-1, x), skeleton.get(y-1, x+1),
                            skeleton.get(y, x+1), skeleton.get(y+1, x+1), skeleton.get(y+1, x),
                            skeleton.get(y+1, x-1), skeleton.get(y, x-1)
                        };
                        
                        // 计算连通数 / Calculate connectivity number
                        int connectivity = 0;
                        for (int i = 0; i < 8; i++) {
                            int next = (i + 1) % 8;
                            if (neighbors[i] == 0 && neighbors[next] == 1) {
                                connectivity++;
                            }
                        }
                        
                        // 计算前景像素数 / Calculate number of foreground pixels
                        int foreground = 0;
                        for (double neighbor : neighbors) {
                            if (neighbor > 0) foreground++;
                        }
                        
                        // 细化条件 / Thinning conditions
                        if (connectivity == 1 && foreground >= 2 && foreground <= 6) {
                            // 检查其他条件 / Check other conditions
                            boolean condition1 = neighbors[0] * neighbors[2] * neighbors[4] == 0;
                            boolean condition2 = neighbors[2] * neighbors[4] * neighbors[6] == 0;
                            
                            if (condition1 && condition2) {
                                temp.set(y, x, 0.0);
                                changed = true;
                            }
                        }
                    }
                }
            }
            
            skeleton = temp;
            iteration++;
        }
        
        return new ImageData(skeleton, image.getWidth(), image.getHeight(), 1,
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * 距离变换 / Distance Transform
     * <p>
     * 距离变换计算每个前景像素到最近背景像素的距离。
     * Distance transform calculates the distance from each foreground pixel to the nearest background pixel.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @return 距离变换图像 / Distance transform image
     */
    public static ImageData distanceTransform(ImageData image) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        // 转换为二值图像 / Convert to binary image
        ImageData binary = image.toGrayscale();
        IMatrix<Double> channel = binary.getChannel(0);
        
        int height = channel.getRowNum();
        int width = channel.getColNum();
        IMatrix<Double> distance = Linalg.zeros(height, width);
        
        // 初始化 / Initialize
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                distance.set(y, x, channel.get(y, x) > 0.5 ? Double.MAX_VALUE : 0.0);
            }
        }
        
        // 前向扫描 / Forward pass
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (distance.get(y, x) > 0) {
                    double minDist = distance.get(y, x);
                    
                    if (y > 0) {
                        minDist = Math.min(minDist, distance.get(y-1, x) + 1);
                    }
                    if (x > 0) {
                        minDist = Math.min(minDist, distance.get(y, x-1) + 1);
                    }
                    if (y > 0 && x > 0) {
                        minDist = Math.min(minDist, distance.get(y-1, x-1) + Math.sqrt(2));
                    }
                    if (y > 0 && x < width - 1) {
                        minDist = Math.min(minDist, distance.get(y-1, x+1) + Math.sqrt(2));
                    }
                    
                    distance.set(y, x, minDist);
                }
            }
        }
        
        // 后向扫描 / Backward pass
        for (int y = height - 1; y >= 0; y--) {
            for (int x = width - 1; x >= 0; x--) {
                if (distance.get(y, x) > 0) {
                    double minDist = distance.get(y, x);
                    
                    if (y < height - 1) {
                        minDist = Math.min(minDist, distance.get(y+1, x) + 1);
                    }
                    if (x < width - 1) {
                        minDist = Math.min(minDist, distance.get(y, x+1) + 1);
                    }
                    if (y < height - 1 && x < width - 1) {
                        minDist = Math.min(minDist, distance.get(y+1, x+1) + Math.sqrt(2));
                    }
                    if (y < height - 1 && x > 0) {
                        minDist = Math.min(minDist, distance.get(y+1, x-1) + Math.sqrt(2));
                    }
                    
                    distance.set(y, x, minDist);
                }
            }
        }
        
        return new ImageData(distance, image.getWidth(), image.getHeight(), 1,
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
}
