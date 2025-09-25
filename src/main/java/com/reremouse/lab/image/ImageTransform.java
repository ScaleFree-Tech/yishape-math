package com.reremouse.lab.image;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.wavele.WaveletAnalysis;
import com.reremouse.lab.math.signal.core.Complex;
import com.reremouse.lab.math.signal.core.RereFFT;
import com.reremouse.lab.math.signal.wavele.WaveletCoefficients;

/**
 * 图像变换类 / Image Transform Class
 * <p>
 * 提供各种图像变换功能，包括傅里叶变换、小波变换、离散余弦变换等。
 * 充分利用现有的信号处理功能和线性代数功能。
 * </p>
 * <p>
 * Provides various image transform functionality including Fourier transform,
 * wavelet transform, discrete cosine transform, etc. Fully utilizes existing
 * signal processing and linear algebra functionality.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class ImageTransform {
    
    /**
     * 变换类型枚举 / Transform Type Enum
     */
    public enum TransformType {
        FFT,                // 快速傅里叶变换 / Fast Fourier Transform
        IFFT,               // 逆快速傅里叶变换 / Inverse Fast Fourier Transform
        DCT,                // 离散余弦变换 / Discrete Cosine Transform
        IDCT,               // 逆离散余弦变换 / Inverse Discrete Cosine Transform
        WAVELET,            // 小波变换 / Wavelet Transform
        IWAVELET,           // 逆小波变换 / Inverse Wavelet Transform
        HADAMARD,           // 哈达玛变换 / Hadamard Transform
        HARTLEY             // 哈特利变换 / Hartley Transform
    }
    
    /**
     * 频域滤波类型枚举 / Frequency Domain Filter Type Enum
     */
    public enum FrequencyFilterType {
        LOW_PASS,           // 低通滤波 / Low-pass filter
        HIGH_PASS,          // 高通滤波 / High-pass filter
        BAND_PASS,          // 带通滤波 / Band-pass filter
        BAND_STOP,          // 带阻滤波 / Band-stop filter
        GAUSSIAN,           // 高斯滤波 / Gaussian filter
        BUTTERWORTH,        // 巴特沃斯滤波 / Butterworth filter
        IDEAL               // 理想滤波 / Ideal filter
    }
    
    /**
     * 2D FFT变换 / 2D FFT Transform
     * <p>
     * 对图像进行2D快速傅里叶变换，将图像从空间域转换到频域。
     * Performs 2D Fast Fourier Transform on image, converting from spatial domain to frequency domain.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @return FFT变换结果 / FFT transform result
     */
    public static ImageFFTResult fft2D(ImageData image) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        
        int height = channel.getRowNum();
        int width = channel.getColNum();
        
        // 零填充到2的幂 / Zero-pad to power of 2
        int paddedHeight = nextPowerOfTwo(height);
        int paddedWidth = nextPowerOfTwo(width);
        
        // 创建复数矩阵 / Create complex matrix
        Complex[][] complexData = new Complex[paddedHeight][paddedWidth];
        
        // 初始化复数数据 / Initialize complex data
        for (int y = 0; y < paddedHeight; y++) {
            for (int x = 0; x < paddedWidth; x++) {
                if (y < height && x < width) {
                    complexData[y][x] = new Complex(channel.get(y, x), 0.0);
                } else {
                    complexData[y][x] = new Complex(0.0, 0.0);
                }
            }
        }
        
        // 执行2D FFT / Perform 2D FFT
        Complex[][] fftResult = fft2D(complexData, paddedHeight, paddedWidth);
        
        return new ImageFFTResult(fftResult, paddedHeight, paddedWidth, height, width);
    }
    
    /**
     * 2D IFFT逆变换 / 2D IFFT Inverse Transform
     * <p>
     * 对频域数据进行2D逆快速傅里叶变换，将图像从频域转换回空间域。
     * Performs 2D Inverse Fast Fourier Transform on frequency domain data, converting back to spatial domain.
     * </p>
     * 
     * @param fftResult FFT变换结果 / FFT transform result
     * @return 重建的图像 / Reconstructed image
     */
    public static ImageData ifft2D(ImageFFTResult fftResult) {
        if (fftResult == null) {
            throw new IllegalArgumentException("FFT结果不能为null / FFT result cannot be null");
        }
        
        Complex[][] fftData = fftResult.getFFTData();
        int paddedHeight = fftResult.getPaddedHeight();
        int paddedWidth = fftResult.getPaddedWidth();
        int originalHeight = fftResult.getOriginalHeight();
        int originalWidth = fftResult.getOriginalWidth();
        
        // 执行2D IFFT / Perform 2D IFFT
        Complex[][] ifftResult = ifft2D(fftData, paddedHeight, paddedWidth);
        
        // 提取原始尺寸的图像 / Extract original size image
        IMatrix<Double> imageData = Linalg.zeros(originalHeight, originalWidth);
        for (int y = 0; y < originalHeight; y++) {
            for (int x = 0; x < originalWidth; x++) {
                imageData.set(y, x, ifftResult[y][x].real);
            }
        }
        
        return new ImageData(imageData, originalWidth, originalHeight, 1, 
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * 频域滤波 / Frequency Domain Filtering
     * <p>
     * 在频域对图像进行滤波处理。
     * Performs filtering on image in frequency domain.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param filterType 滤波类型 / Filter type
     * @param cutoffFreq 截止频率 / Cutoff frequency
     * @param filterOrder 滤波器阶数 / Filter order
     * @return 滤波后的图像 / Filtered image
     */
    public static ImageData frequencyDomainFilter(ImageData image, FrequencyFilterType filterType, 
                                                double cutoffFreq, int filterOrder) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        // 执行FFT / Perform FFT
        ImageFFTResult fftResult = fft2D(image);
        
        // 创建滤波器 / Create filter
        Complex[][] filter = createFrequencyFilter(fftResult.getPaddedHeight(), fftResult.getPaddedWidth(), 
                                                 filterType, cutoffFreq, filterOrder);
        
        // 应用滤波器 / Apply filter
        Complex[][] fftData = fftResult.getFFTData();
        int height = fftData.length;
        int width = fftData[0].length;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                fftData[y][x] = fftData[y][x].multiply(filter[y][x]);
            }
        }
        
        // 执行IFFT / Perform IFFT
        return ifft2D(fftResult);
    }
    
    /**
     * 小波变换 / Wavelet Transform
     * <p>
     * 对图像进行小波变换，提供多尺度分析能力。
     * Performs wavelet transform on image, providing multi-scale analysis capability.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @param waveletType 小波类型 / Wavelet type
     * @param levels 分解层数 / Number of decomposition levels
     * @return 小波变换结果 / Wavelet transform result
     */
    public static ImageWaveletResult waveletTransform(ImageData image, WaveletAnalysis.WaveletType waveletType, int levels) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        
        int height = channel.getRowNum();
        int width = channel.getColNum();
        
        // 对每一行进行小波变换 / Perform wavelet transform on each row
        WaveletCoefficients[] rowCoeffs = new WaveletCoefficients[height];
        for (int y = 0; y < height; y++) {
            IVector<Double> row = channel.getRow(y);
            rowCoeffs[y] = WaveletAnalysis.discreteWaveletTransform(row, waveletType, levels, 0.0);
        }
        
        // 对每一列进行小波变换 / Perform wavelet transform on each column
        WaveletCoefficients[][] coeffs = new WaveletCoefficients[height][width];
        for (int y = 0; y < height; y++) {
            for (int level = 0; level <= levels; level++) {
                IVector<Double> column;
                if (level == levels) {
                    column = rowCoeffs[y].approximation;
                } else {
                    column = rowCoeffs[y].details[level];
                }
                
                WaveletCoefficients colCoeffs = WaveletAnalysis.discreteWaveletTransform(column, waveletType, levels, 0.0);
                coeffs[y][level] = colCoeffs;
            }
        }
        
        return new ImageWaveletResult(coeffs, height, width, levels, waveletType);
    }
    
    /**
     * 逆小波变换 / Inverse Wavelet Transform
     * <p>
     * 从小波系数重建原始图像。
     * Reconstructs original image from wavelet coefficients.
     * </p>
     * 
     * @param waveletResult 小波变换结果 / Wavelet transform result
     * @return 重建的图像 / Reconstructed image
     */
    public static ImageData inverseWaveletTransform(ImageWaveletResult waveletResult) {
        if (waveletResult == null) {
            throw new IllegalArgumentException("小波结果不能为null / Wavelet result cannot be null");
        }
        
        WaveletCoefficients[][] coeffs = waveletResult.getCoeffs();
        int height = waveletResult.getHeight();
        int width = waveletResult.getWidth();
        int levels = waveletResult.getLevels();
        WaveletAnalysis.WaveletType waveletType = waveletResult.getWaveletType();
        
        // 重建每一列 / Reconstruct each column
        IMatrix<Double> reconstructed = Linalg.zeros(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int level = 0; level <= levels; level++) {
                WaveletCoefficients colCoeffs = coeffs[y][level];
                IVector<Double> reconstructedColumn = WaveletAnalysis.inverseDiscreteWaveletTransform(colCoeffs, waveletType, 0.0);
                
                // 将重建的列数据放回矩阵 / Put reconstructed column data back to matrix
                for (int x = 0; x < Math.min(width, reconstructedColumn.length()); x++) {
                    reconstructed.set(y, x, reconstructedColumn.get(x));
                }
            }
        }
        
        return new ImageData(reconstructed, width, height, 1, 
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * 离散余弦变换 (DCT) / Discrete Cosine Transform (DCT)
     * <p>
     * 对图像进行DCT变换，常用于图像压缩。
     * Performs DCT transform on image, commonly used for image compression.
     * </p>
     * 
     * @param image 输入图像 / Input image
     * @return DCT变换结果 / DCT transform result
     */
    public static ImageData dct2D(ImageData image) {
        if (image == null) {
            throw new IllegalArgumentException("图像不能为null / Image cannot be null");
        }
        
        // 转换为灰度图像 / Convert to grayscale
        ImageData grayscale = image.toGrayscale();
        IMatrix<Double> channel = grayscale.getChannel(0);
        
        int height = channel.getRowNum();
        int width = channel.getColNum();
        
        IMatrix<Double> dctResult = Linalg.zeros(height, width);
        
        // 对每个8x8块进行DCT / Perform DCT on each 8x8 block
        for (int y = 0; y < height; y += 8) {
            for (int x = 0; x < width; x += 8) {
                IMatrix<Double> block = extractBlock(channel, x, y, 8, 8);
                IMatrix<Double> dctBlock = dctBlock(block);
                setBlock(dctResult, dctBlock, x, y);
            }
        }
        
        return new ImageData(dctResult, width, height, 1, 
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
    
    /**
     * 逆离散余弦变换 (IDCT) / Inverse Discrete Cosine Transform (IDCT)
     * <p>
     * 对DCT系数进行逆变换，重建原始图像。
     * Performs inverse DCT transform on DCT coefficients to reconstruct original image.
     * </p>
     * 
     * @param dctImage DCT变换图像 / DCT transform image
     * @return 重建的图像 / Reconstructed image
     */
    public static ImageData idct2D(ImageData dctImage) {
        if (dctImage == null) {
            throw new IllegalArgumentException("DCT图像不能为null / DCT image cannot be null");
        }
        
        IMatrix<Double> channel = dctImage.getChannel(0);
        int height = channel.getRowNum();
        int width = channel.getColNum();
        
        IMatrix<Double> idctResult = Linalg.zeros(height, width);
        
        // 对每个8x8块进行IDCT / Perform IDCT on each 8x8 block
        for (int y = 0; y < height; y += 8) {
            for (int x = 0; x < width; x += 8) {
                IMatrix<Double> block = extractBlock(channel, x, y, 8, 8);
                IMatrix<Double> idctBlock = idctBlock(block);
                setBlock(idctResult, idctBlock, x, y);
            }
        }
        
        return new ImageData(idctResult, width, height, 1, 
                           ImageData.ImageType.GRAYSCALE, ImageData.PixelFormat.FLOAT64);
    }
    
    // ========== 辅助方法 / Helper Methods ==========
    
    /**
     * 2D FFT实现 / 2D FFT Implementation
     */
    private static Complex[][] fft2D(Complex[][] data, int height, int width) {
        Complex[][] result = new Complex[height][width];
        
        // 对每一行进行FFT / Perform FFT on each row
        for (int y = 0; y < height; y++) {
            result[y] = RereFFT.fft(data[y]);
        }
        
        // 对每一列进行FFT / Perform FFT on each column
        for (int x = 0; x < width; x++) {
            Complex[] column = new Complex[height];
            for (int y = 0; y < height; y++) {
                column[y] = result[y][x];
            }
            
            Complex[] fftColumn = RereFFT.fft(column);
            for (int y = 0; y < height; y++) {
                result[y][x] = fftColumn[y];
            }
        }
        
        return result;
    }
    
    /**
     * 2D IFFT实现 / 2D IFFT Implementation
     */
    private static Complex[][] ifft2D(Complex[][] data, int height, int width) {
        Complex[][] result = new Complex[height][width];
        
        // 对每一列进行IFFT / Perform IFFT on each column
        for (int x = 0; x < width; x++) {
            Complex[] column = new Complex[height];
            for (int y = 0; y < height; y++) {
                column[y] = data[y][x];
            }
            
            Complex[] ifftColumn = RereFFT.ifft(column);
            for (int y = 0; y < height; y++) {
                result[y][x] = ifftColumn[y];
            }
        }
        
        // 对每一行进行IFFT / Perform IFFT on each row
        for (int y = 0; y < height; y++) {
            result[y] = RereFFT.ifft(result[y]);
        }
        
        return result;
    }
    
    /**
     * 创建频域滤波器 / Create frequency domain filter
     */
    private static Complex[][] createFrequencyFilter(int height, int width, FrequencyFilterType filterType, 
                                                   double cutoffFreq, int filterOrder) {
        Complex[][] filter = new Complex[height][width];
        int centerY = height / 2;
        int centerX = width / 2;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double distance = Math.sqrt(Math.pow(y - centerY, 2) + Math.pow(x - centerX, 2));
                double normalizedDistance = distance / Math.min(height, width);
                
                double filterValue = 1.0;
                
                switch (filterType) {
                    case LOW_PASS:
                        filterValue = (normalizedDistance <= cutoffFreq) ? 1.0 : 0.0;
                        break;
                    case HIGH_PASS:
                        filterValue = (normalizedDistance > cutoffFreq) ? 1.0 : 0.0;
                        break;
                    case BAND_PASS:
                        // 简化的带通滤波 / Simplified band-pass filter
                        filterValue = (normalizedDistance >= cutoffFreq * 0.5 && normalizedDistance <= cutoffFreq * 1.5) ? 1.0 : 0.0;
                        break;
                    case GAUSSIAN:
                        filterValue = Math.exp(-Math.pow(normalizedDistance, 2) / (2 * Math.pow(cutoffFreq, 2)));
                        break;
                    case BUTTERWORTH:
                        filterValue = 1.0 / (1.0 + Math.pow(normalizedDistance / cutoffFreq, 2 * filterOrder));
                        break;
                    default:
                        filterValue = 1.0;
                        break;
                }
                
                filter[y][x] = new Complex(filterValue, 0.0);
            }
        }
        
        return filter;
    }
    
    /**
     * 计算大于等于n的最小2的幂 / Calculate smallest power of 2 >= n
     */
    private static int nextPowerOfTwo(int n) {
        if (n <= 0) return 1;
        if ((n & (n - 1)) == 0) return n;
        
        int power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }
    
    /**
     * 提取图像块 / Extract image block
     */
    private static IMatrix<Double> extractBlock(IMatrix<Double> image, int startX, int startY, int blockWidth, int blockHeight) {
        IMatrix<Double> block = Linalg.zeros(blockHeight, blockWidth);
        
        for (int y = 0; y < blockHeight; y++) {
            for (int x = 0; x < blockWidth; x++) {
                int imageY = startY + y;
                int imageX = startX + x;
                
                if (imageY < image.getRowNum() && imageX < image.getColNum()) {
                    block.set(y, x, image.get(imageY, imageX));
                } else {
                    block.set(y, x, 0.0);
                }
            }
        }
        
        return block;
    }
    
    /**
     * 设置图像块 / Set image block
     */
    private static void setBlock(IMatrix<Double> image, IMatrix<Double> block, int startX, int startY) {
        int blockHeight = block.getRowNum();
        int blockWidth = block.getColNum();
        
        for (int y = 0; y < blockHeight; y++) {
            for (int x = 0; x < blockWidth; x++) {
                int imageY = startY + y;
                int imageX = startX + x;
                
                if (imageY < image.getRowNum() && imageX < image.getColNum()) {
                    image.set(imageY, imageX, block.get(y, x));
                }
            }
        }
    }
    
    /**
     * 8x8块的DCT变换 / DCT transform for 8x8 block
     */
    private static IMatrix<Double> dctBlock(IMatrix<Double> block) {
        int size = 8;
        IMatrix<Double> dctBlock = Linalg.zeros(size, size);
        
        for (int u = 0; u < size; u++) {
            for (int v = 0; v < size; v++) {
                double sum = 0.0;
                
                for (int x = 0; x < size; x++) {
                    for (int y = 0; y < size; y++) {
                        double cu = (u == 0) ? 1.0 / Math.sqrt(2) : 1.0;
                        double cv = (v == 0) ? 1.0 / Math.sqrt(2) : 1.0;
                        
                        double cosU = Math.cos((2 * x + 1) * u * Math.PI / (2 * size));
                        double cosV = Math.cos((2 * y + 1) * v * Math.PI / (2 * size));
                        
                        sum += block.get(y, x) * cu * cv * cosU * cosV;
                    }
                }
                
                dctBlock.set(u, v, sum / 4.0);
            }
        }
        
        return dctBlock;
    }
    
    /**
     * 8x8块的IDCT变换 / IDCT transform for 8x8 block
     */
    private static IMatrix<Double> idctBlock(IMatrix<Double> dctBlock) {
        int size = 8;
        IMatrix<Double> block = Linalg.zeros(size, size);
        
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                double sum = 0.0;
                
                for (int u = 0; u < size; u++) {
                    for (int v = 0; v < size; v++) {
                        double cu = (u == 0) ? 1.0 / Math.sqrt(2) : 1.0;
                        double cv = (v == 0) ? 1.0 / Math.sqrt(2) : 1.0;
                        
                        double cosU = Math.cos((2 * x + 1) * u * Math.PI / (2 * size));
                        double cosV = Math.cos((2 * y + 1) * v * Math.PI / (2 * size));
                        
                        sum += dctBlock.get(u, v) * cu * cv * cosU * cosV;
                    }
                }
                
                block.set(y, x, sum / 4.0);
            }
        }
        
        return block;
    }
    
    /**
     * FFT变换结果类 / FFT Transform Result Class
     */
    public static class ImageFFTResult {
        private Complex[][] fftData;
        private int paddedHeight;
        private int paddedWidth;
        private int originalHeight;
        private int originalWidth;
        
        public ImageFFTResult(Complex[][] fftData, int paddedHeight, int paddedWidth, 
                            int originalHeight, int originalWidth) {
            this.fftData = fftData;
            this.paddedHeight = paddedHeight;
            this.paddedWidth = paddedWidth;
            this.originalHeight = originalHeight;
            this.originalWidth = originalWidth;
        }
        
        public Complex[][] getFFTData() { return fftData; }
        public int getPaddedHeight() { return paddedHeight; }
        public int getPaddedWidth() { return paddedWidth; }
        public int getOriginalHeight() { return originalHeight; }
        public int getOriginalWidth() { return originalWidth; }
        
        /**
         * 获取幅度谱 / Get magnitude spectrum
         */
        public IMatrix<Double> getMagnitudeSpectrum() {
            IMatrix<Double> magnitude = Linalg.zeros(paddedHeight, paddedWidth);
            for (int y = 0; y < paddedHeight; y++) {
                for (int x = 0; x < paddedWidth; x++) {
                    magnitude.set(y, x, fftData[y][x].magnitude());
                }
            }
            return magnitude;
        }
        
        /**
         * 获取相位谱 / Get phase spectrum
         */
        public IMatrix<Double> getPhaseSpectrum() {
            IMatrix<Double> phase = Linalg.zeros(paddedHeight, paddedWidth);
            for (int y = 0; y < paddedHeight; y++) {
                for (int x = 0; x < paddedWidth; x++) {
                    phase.set(y, x, Math.atan2(fftData[y][x].imag, fftData[y][x].real));
                }
            }
            return phase;
        }
    }
    
    /**
     * 小波变换结果类 / Wavelet Transform Result Class
     */
    public static class ImageWaveletResult {
        private WaveletCoefficients[][] coeffs;
        private int height;
        private int width;
        private int levels;
        private WaveletAnalysis.WaveletType waveletType;
        
        public ImageWaveletResult(WaveletCoefficients[][] coeffs, int height, int width, 
                                int levels, WaveletAnalysis.WaveletType waveletType) {
            this.coeffs = coeffs;
            this.height = height;
            this.width = width;
            this.levels = levels;
            this.waveletType = waveletType;
        }
        
        public WaveletCoefficients[][] getCoeffs() { return coeffs; }
        public int getHeight() { return height; }
        public int getWidth() { return width; }
        public int getLevels() { return levels; }
        public WaveletAnalysis.WaveletType getWaveletType() { return waveletType; }
    }
}
