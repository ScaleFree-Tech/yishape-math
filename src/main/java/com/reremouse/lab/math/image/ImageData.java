package com.reremouse.lab.math.image;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;

/**
 * 图像数据类 / Image Data Class
 * <p>
 * 表示图像数据的基础类，支持灰度图像和彩色图像。
 * 使用IMatrix存储图像数据，充分利用现有的线性代数功能。
 * </p>
 * <p>
 * Base class representing image data, supporting grayscale and color images.
 * Uses IMatrix to store image data, fully utilizing existing linear algebra functionality.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ImageData {
    
    /**
     * 图像类型枚举 / Image Type Enum
     */
    public enum ImageType {
        GRAYSCALE,  // 灰度图像 / Grayscale image
        RGB,        // RGB彩色图像 / RGB color image
        RGBA,       // RGBA彩色图像 / RGBA color image
        HSV,        // HSV彩色图像 / HSV color image
        LAB         // LAB彩色图像 / LAB color image
    }
    
    /**
     * 像素格式枚举 / Pixel Format Enum
     */
    public enum PixelFormat {
        UINT8,      // 8位无符号整数 / 8-bit unsigned integer
        UINT16,     // 16位无符号整数 / 16-bit unsigned integer
        FLOAT32,    // 32位浮点数 / 32-bit float
        FLOAT64     // 64位浮点数 / 64-bit double
    }
    
    private IMatrix<Double> data;           // 图像数据矩阵 / Image data matrix
    private int width;                      // 图像宽度 / Image width
    private int height;                     // 图像高度 / Image height
    private int channels;                   // 通道数 / Number of channels
    private ImageType imageType;            // 图像类型 / Image type
    private PixelFormat pixelFormat;        // 像素格式 / Pixel format
    private double minValue;                // 最小值 / Minimum value
    private double maxValue;                // 最大值 / Maximum value
    
    /**
     * 构造函数 - 从矩阵创建图像 / Constructor - Create image from matrix
     * 
     * @param data 图像数据矩阵 / Image data matrix
     * @param width 图像宽度 / Image width
     * @param height 图像高度 / Image height
     * @param channels 通道数 / Number of channels
     * @param imageType 图像类型 / Image type
     * @param pixelFormat 像素格式 / Pixel format
     */
    public ImageData(IMatrix<Double> data, int width, int height, int channels, 
                    ImageType imageType, PixelFormat pixelFormat) {
        if (data == null) {
            throw new IllegalArgumentException("图像数据不能为null / Image data cannot be null");
        }
        if (width <= 0 || height <= 0 || channels <= 0) {
            throw new IllegalArgumentException("宽度、高度和通道数必须大于0 / Width, height and channels must be greater than 0");
        }
        if (data.getRowNum() != height || data.getColNum() != width * channels) {
            throw new IllegalArgumentException("矩阵维度与图像尺寸不匹配 / Matrix dimensions don't match image size");
        }
        
        this.data = data;
        this.width = width;
        this.height = height;
        this.channels = channels;
        this.imageType = imageType;
        this.pixelFormat = pixelFormat;
        
        // 计算值域 / Calculate value range
        calculateValueRange();
    }
    
    /**
     * 构造函数 - 从数组创建灰度图像 / Constructor - Create grayscale image from array
     * 
     * @param pixelData 像素数据数组 / Pixel data array
     * @param width 图像宽度 / Image width
     * @param height 图像高度 / Image height
     */
    public ImageData(double[][] pixelData, int width, int height) {
        this(pixelData, width, height, 1, ImageType.GRAYSCALE, PixelFormat.FLOAT64);
    }
    
    /**
     * 构造函数 - 从数组创建图像 / Constructor - Create image from array
     * 
     * @param pixelData 像素数据数组 / Pixel data array
     * @param width 图像宽度 / Image width
     * @param height 图像高度 / Image height
     * @param channels 通道数 / Number of channels
     * @param imageType 图像类型 / Image type
     * @param pixelFormat 像素格式 / Pixel format
     */
    public ImageData(double[][] pixelData, int width, int height, int channels, 
                    ImageType imageType, PixelFormat pixelFormat) {
        if (pixelData == null) {
            throw new IllegalArgumentException("像素数据不能为null / Pixel data cannot be null");
        }
        if (width <= 0 || height <= 0 || channels <= 0) {
            throw new IllegalArgumentException("宽度、高度和通道数必须大于0 / Width, height and channels must be greater than 0");
        }
        if (pixelData.length != height || pixelData[0].length != width * channels) {
            throw new IllegalArgumentException("数组维度与图像尺寸不匹配 / Array dimensions don't match image size");
        }
        
        this.data = Linalg.matrix(pixelData);
        this.width = width;
        this.height = height;
        this.channels = channels;
        this.imageType = imageType;
        this.pixelFormat = pixelFormat;
        
        // 计算值域 / Calculate value range
        calculateValueRange();
    }
    
    /**
     * 构造函数 - 从通道数组创建图像 / Constructor - Create image from channel array
     * 
     * @param channels 通道数组 / Channel array
     * @param width 图像宽度 / Image width
     * @param height 图像高度 / Image height
     * @param numChannels 通道数 / Number of channels
     * @param imageType 图像类型 / Image type
     * @param pixelFormat 像素格式 / Pixel format
     */
    public ImageData(IMatrix<Double>[] channels, int width, int height, int numChannels, 
                    ImageType imageType, PixelFormat pixelFormat) {
        if (channels == null || channels.length != numChannels) {
            throw new IllegalArgumentException("通道数组不能为null且长度必须匹配 / Channel array cannot be null and length must match");
        }
        if (width <= 0 || height <= 0 || numChannels <= 0) {
            throw new IllegalArgumentException("宽度、高度和通道数必须大于0 / Width, height and channels must be greater than 0");
        }
        
        this.width = width;
        this.height = height;
        this.channels = numChannels;
        this.imageType = imageType;
        this.pixelFormat = pixelFormat;
        
        // 合并通道数据 / Merge channel data
        this.data = Linalg.zeros(height, width * numChannels);
        for (int c = 0; c < numChannels; c++) {
            if (channels[c].getRowNum() != height || channels[c].getColNum() != width) {
                throw new IllegalArgumentException("通道维度与图像尺寸不匹配 / Channel dimensions don't match image size");
            }
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    this.data.set(y, x * numChannels + c, channels[c].get(y, x));
                }
            }
        }
        
        // 计算值域 / Calculate value range
        calculateValueRange();
    }
    
    /**
     * 构造函数 - 创建空图像 / Constructor - Create empty image
     * 
     * @param width 图像宽度 / Image width
     * @param height 图像高度 / Image height
     * @param channels 通道数 / Number of channels
     * @param imageType 图像类型 / Image type
     * @param pixelFormat 像素格式 / Pixel format
     */
    public ImageData(int width, int height, int channels, ImageType imageType, PixelFormat pixelFormat) {
        this.width = width;
        this.height = height;
        this.channels = channels;
        this.imageType = imageType;
        this.pixelFormat = pixelFormat;
        
        // 创建零矩阵 / Create zero matrix
        this.data = Linalg.zeros(height, width * channels);
        this.minValue = 0.0;
        this.maxValue = 0.0;
    }
    
    /**
     * 获取像素值 / Get pixel value
     * 
     * @param x X坐标 / X coordinate
     * @param y Y坐标 / Y coordinate
     * @param channel 通道索引 / Channel index
     * @return 像素值 / Pixel value
     */
    public double getPixel(int x, int y, int channel) {
        if (x < 0 || x >= width || y < 0 || y >= height || channel < 0 || channel >= channels) {
            throw new IndexOutOfBoundsException("坐标或通道索引超出范围 / Coordinate or channel index out of bounds");
        }
        return data.get(y, x * channels + channel);
    }
    
    /**
     * 设置像素值 / Set pixel value
     * 
     * @param x X坐标 / X coordinate
     * @param y Y坐标 / Y coordinate
     * @param channel 通道索引 / Channel index
     * @param value 像素值 / Pixel value
     */
    public void setPixel(int x, int y, int channel, double value) {
        if (x < 0 || x >= width || y < 0 || y >= height || channel < 0 || channel >= channels) {
            throw new IndexOutOfBoundsException("坐标或通道索引超出范围 / Coordinate or channel index out of bounds");
        }
        data.set(y, x * channels + channel, value);
        
        // 更新值域 / Update value range
        if (value < minValue) minValue = value;
        if (value > maxValue) maxValue = value;
    }
    
    /**
     * 获取指定通道的矩阵 / Get matrix for specified channel
     * 
     * @param channel 通道索引 / Channel index
     * @return 通道矩阵 / Channel matrix
     */
    public IMatrix<Double> getChannel(int channel) {
        if (channel < 0 || channel >= channels) {
            throw new IndexOutOfBoundsException("通道索引超出范围 / Channel index out of bounds");
        }
        
        IMatrix<Double> channelMatrix = Linalg.zeros(height, width);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                channelMatrix.set(y, x, getPixel(x, y, channel));
            }
        }
        return channelMatrix;
    }
    
    /**
     * 设置指定通道的矩阵 / Set matrix for specified channel
     * 
     * @param channel 通道索引 / Channel index
     * @param channelMatrix 通道矩阵 / Channel matrix
     */
    public void setChannel(int channel, IMatrix<Double> channelMatrix) {
        if (channel < 0 || channel >= channels) {
            throw new IndexOutOfBoundsException("通道索引超出范围 / Channel index out of bounds");
        }
        if (channelMatrix.getRowNum() != height || channelMatrix.getColNum() != width) {
            throw new IllegalArgumentException("通道矩阵尺寸不匹配 / Channel matrix size doesn't match");
        }
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                setPixel(x, y, channel, channelMatrix.get(y, x));
            }
        }
    }
    
    /**
     * 获取指定行的像素向量 / Get pixel vector for specified row
     * 
     * @param y Y坐标 / Y coordinate
     * @return 像素向量 / Pixel vector
     */
    public IVector<Double> getRow(int y) {
        if (y < 0 || y >= height) {
            throw new IndexOutOfBoundsException("Y坐标超出范围 / Y coordinate out of bounds");
        }
        return data.getRow(y);
    }
    
    /**
     * 获取指定列的像素向量 / Get pixel vector for specified column
     * 
     * @param x X坐标 / X coordinate
     * @return 像素向量 / Pixel vector
     */
    public IVector<Double> getColumn(int x) {
        if (x < 0 || x >= width) {
            throw new IndexOutOfBoundsException("X坐标超出范围 / X coordinate out of bounds");
        }
        
        IVector<Double> column = Linalg.zeros(height * channels);
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                column.set(y * channels + c, getPixel(x, y, c));
            }
        }
        return column;
    }
    
    /**
     * 获取图像子区域 / Get image subregion
     * 
     * @param x X起始坐标 / X start coordinate
     * @param y Y起始坐标 / Y start coordinate
     * @param w 宽度 / Width
     * @param h 高度 / Height
     * @return 子图像 / Sub-image
     */
    public ImageData getSubImage(int x, int y, int w, int h) {
        if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > width || y + h > height) {
            throw new IllegalArgumentException("子区域参数无效 / Subregion parameters invalid");
        }
        
        IMatrix<Double> subData = Linalg.zeros(h, w * channels);
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w * channels; col++) {
                subData.set(row, col, data.get(y + row, (x * channels) + col));
            }
        }
        
        return new ImageData(subData, w, h, channels, imageType, pixelFormat);
    }
    
    /**
     * 设置图像子区域 / Set image subregion
     * 
     * @param x X起始坐标 / X start coordinate
     * @param y Y起始坐标 / Y start coordinate
     * @param subImage 子图像 / Sub-image
     */
    public void setSubImage(int x, int y, ImageData subImage) {
        if (x < 0 || y < 0 || x + subImage.width > width || y + subImage.height > height) {
            throw new IllegalArgumentException("子区域超出图像边界 / Subregion exceeds image bounds");
        }
        if (subImage.channels != channels) {
            throw new IllegalArgumentException("子图像通道数不匹配 / Sub-image channel count doesn't match");
        }
        
        for (int row = 0; row < subImage.height; row++) {
            for (int col = 0; col < subImage.width * channels; col++) {
                data.set(y + row, (x * channels) + col, subImage.data.get(row, col));
            }
        }
        
        // 重新计算值域 / Recalculate value range
        calculateValueRange();
    }
    
    /**
     * 转换为灰度图像 / Convert to grayscale image
     * 
     * @return 灰度图像 / Grayscale image
     */
    public ImageData toGrayscale() {
        if (imageType == ImageType.GRAYSCALE) {
            return this.copy();
        }
        
        ImageData grayscale = new ImageData(width, height, 1, ImageType.GRAYSCALE, pixelFormat);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double grayValue = 0.0;
                
                switch (imageType) {
                    case RGB:
                        // RGB转灰度：0.299*R + 0.587*G + 0.114*B / RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
                        grayValue = 0.299 * getPixel(x, y, 0) + 0.587 * getPixel(x, y, 1) + 0.114 * getPixel(x, y, 2);
                        break;
                    case RGBA:
                        // RGBA转灰度，忽略Alpha通道 / RGBA to grayscale, ignore Alpha channel
                        grayValue = 0.299 * getPixel(x, y, 0) + 0.587 * getPixel(x, y, 1) + 0.114 * getPixel(x, y, 2);
                        break;
                    case HSV:
                        // HSV转灰度，使用V通道 / HSV to grayscale, use V channel
                        grayValue = getPixel(x, y, 2);
                        break;
                    case LAB:
                        // LAB转灰度，使用L通道 / LAB to grayscale, use L channel
                        grayValue = getPixel(x, y, 0);
                        break;
                    default:
                        // 默认取第一个通道 / Default to first channel
                        grayValue = getPixel(x, y, 0);
                        break;
                }
                
                grayscale.setPixel(x, y, 0, grayValue);
            }
        }
        
        return grayscale;
    }
    
    /**
     * 归一化图像值到[0,1]范围 / Normalize image values to [0,1] range
     * 
     * @return 归一化后的图像 / Normalized image
     */
    public ImageData normalize() {
        ImageData normalized = this.copy();
        
        if (maxValue > minValue) {
            double range = maxValue - minValue;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < channels; c++) {
                        double normalizedValue = (getPixel(x, y, c) - minValue) / range;
                        normalized.setPixel(x, y, c, normalizedValue);
                    }
                }
            }
        }
        
        return normalized;
    }
    
    /**
     * 图像复制 / Image copy
     * 
     * @return 图像副本 / Image copy
     */
    public ImageData copy() {
        return new ImageData(data.copy(), width, height, channels, imageType, pixelFormat);
    }
    
    /**
     * 计算值域 / Calculate value range
     */
    private void calculateValueRange() {
        minValue = data.min();
        maxValue = data.max();
    }
    
    // ========== Getter方法 / Getter Methods ==========
    
    public IMatrix<Double> getData() { return data; }
    public int getWidth() { return width; }
    public int getHeight() { return height; }
    public int getChannels() { return channels; }
    public ImageType getImageType() { return imageType; }
    public PixelFormat getPixelFormat() { return pixelFormat; }
    public double getMinValue() { return minValue; }
    public double getMaxValue() { return maxValue; }
    
    /**
     * 获取图像尺寸 / Get image dimensions
     * 
     * @return [宽度, 高度] / [width, height]
     */
    public int[] getDimensions() {
        return new int[]{width, height};
    }
    
    /**
     * 获取图像形状 / Get image shape
     * 
     * @return [高度, 宽度, 通道数] / [height, width, channels]
     */
    public int[] getShape() {
        return new int[]{height, width, channels};
    }
    
    /**
     * 检查是否为灰度图像 / Check if grayscale image
     * 
     * @return 是否为灰度图像 / Whether grayscale image
     */
    public boolean isGrayscale() {
        return imageType == ImageType.GRAYSCALE && channels == 1;
    }
    
    /**
     * 检查是否为彩色图像 / Check if color image
     * 
     * @return 是否为彩色图像 / Whether color image
     */
    public boolean isColor() {
        return channels > 1;
    }
    
    @Override
    public String toString() {
        return String.format("ImageData[%dx%d, %d channels, %s, %s, range=[%.3f, %.3f]]", 
                           width, height, channels, imageType, pixelFormat, minValue, maxValue);
    }
}
