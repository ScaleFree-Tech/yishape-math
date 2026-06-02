package com.yishape.lab.math.signal.core;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 离散余弦变换 (DCT) 类 / Discrete Cosine Transform (DCT) Class
 * <p>
 * 提供各种类型的DCT变换，包括DCT-I、DCT-II、DCT-III、DCT-IV等。
 * 使用IVector接口进行向量操作，确保与现有代码库的兼容性。
 * </p>
 * <p>
 * Provides various types of DCT transforms including DCT-I, DCT-II, DCT-III, DCT-IV, etc.
 * Uses IVector interface for vector operations to ensure compatibility with existing codebase.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereDCT {

    /**
     * DCT-II变换 / DCT-II Transform
     * <p>
     * 最常用的DCT类型，广泛用于图像和音频压缩。
     * Most commonly used DCT type, widely used in image and audio compression.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return DCT-II变换结果 / DCT-II transform result
     */
    public static IVector<Double> dct2(IVector<Double> signal) {
        int n = signal.length();
        if (RereFastDct2.isPowerOfTwo(n)) {
            double[] buf = new double[n];
            for (int i = 0; i < n; i++) {
                buf[i] = signal.get(i);
            }
            RereFastDct2.dct2OrthoInPlace(buf, n);
            IVector<Double> result = Linalg.zeros(n);
            for (int i = 0; i < n; i++) {
                result.set(i, buf[i]);
            }
            return result;
        }
        IVector<Double> result = Linalg.zeros(n);
        for (int k = 0; k < n; k++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += signal.get(i) * Math.cos(Math.PI * k * (2 * i + 1) / (2 * n));
            }
            double alpha = (k == 0) ? Math.sqrt(1.0 / n) : Math.sqrt(2.0 / n);
            result.set(k, alpha * sum);
        }
        return result;
    }

    /**
     * DCT-II逆变换 / DCT-II Inverse Transform
     * <p>
     * DCT-II的逆变换，用于重建原始信号。
     * Inverse transform of DCT-II, used to reconstruct original signal.
     * </p>
     *
     * @param dctSignal DCT变换结果 / DCT transform result
     * @return 重建的原始信号 / Reconstructed original signal
     */
    public static IVector<Double> idct2(IVector<Double> dctSignal) {
        int n = dctSignal.length();
        if (RereFastDct2.isPowerOfTwo(n)) {
            double[] buf = new double[n];
            for (int i = 0; i < n; i++) {
                buf[i] = dctSignal.get(i);
            }
            RereFastDct2.idct2OrthoInPlace(buf, n);
            IVector<Double> result = Linalg.zeros(n);
            for (int i = 0; i < n; i++) {
                result.set(i, buf[i]);
            }
            return result;
        }
        IVector<Double> result = Linalg.zeros(n);
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                double alpha = (k == 0) ? Math.sqrt(1.0 / n) : Math.sqrt(2.0 / n);
                sum += alpha * dctSignal.get(k) * Math.cos(Math.PI * k * (2 * i + 1) / (2 * n));
            }
            result.set(i, sum);
        }
        return result;
    }

    /**
     * DCT-I变换 / DCT-I Transform
     * <p>
     * 对称边界条件的DCT变换。
     * DCT transform with symmetric boundary conditions.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return DCT-I变换结果 / DCT-I transform result
     */
    public static IVector<Double> dct1(IVector<Double> signal) {
        int n = signal.length();
        IVector<Double> result = Linalg.zeros(n);
        
        for (int k = 0; k < n; k++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += signal.get(i) * Math.cos(Math.PI * k * i / (n - 1));
            }
            double alpha = (k == 0 || k == n - 1) ? 0.5 : 1.0;
            result.set(k, alpha * sum);
        }
        
        return result;
    }

    /**
     * DCT-III变换 / DCT-III Transform
     * <p>
     * DCT-II的逆变换形式。
     * Inverse form of DCT-II.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return DCT-III变换结果 / DCT-III transform result
     */
    public static IVector<Double> dct3(IVector<Double> signal) {
        int n = signal.length();
        IVector<Double> result = Linalg.zeros(n);
        
        for (int k = 0; k < n; k++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                double alpha = (i == 0) ? Math.sqrt(1.0 / n) : Math.sqrt(2.0 / n);
                sum += alpha * signal.get(i) * Math.cos(Math.PI * i * (2 * k + 1) / (2 * n));
            }
            result.set(k, sum);
        }
        
        return result;
    }

    /**
     * DCT-IV变换 / DCT-IV Transform
     * <p>
     * 另一种常用的DCT类型。
     * Another commonly used DCT type.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @return DCT-IV变换结果 / DCT-IV transform result
     */
    public static IVector<Double> dct4(IVector<Double> signal) {
        int n = signal.length();
        IVector<Double> result = Linalg.zeros(n);
        
        for (int k = 0; k < n; k++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += signal.get(i) * Math.cos(Math.PI * (2 * k + 1) * (2 * i + 1) / (4 * n));
            }
            result.set(k, Math.sqrt(2.0 / n) * sum);
        }
        
        return result;
    }

    /**
     * 2D DCT变换 / 2D DCT Transform
     * <p>
     * 对矩阵进行2D DCT变换，常用于图像处理。
     * Perform 2D DCT transform on matrix, commonly used in image processing.
     * </p>
     *
     * @param matrix 输入矩阵 / Input matrix
     * @return 2D DCT变换结果 / 2D DCT transform result
     */
    public static double[][] dct2D(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];
        if (RereFastDct2.isPowerOfTwo(rows) && RereFastDct2.isPowerOfTwo(cols)) {
            for (int i = 0; i < rows; i++) {
                System.arraycopy(matrix[i], 0, result[i], 0, cols);
                RereFastDct2.dct2OrthoInPlace(result[i], cols);
            }
            double[] column = new double[rows];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    column[i] = result[i][j];
                }
                RereFastDct2.dct2OrthoInPlace(column, rows);
                for (int i = 0; i < rows; i++) {
                    result[i][j] = column[i];
                }
            }
            return result;
        }
        for (int i = 0; i < rows; i++) {
            IVector<Double> row = Linalg.vector(matrix[i]);
            IVector<Double> dctRow = dct2(row);
            for (int j = 0; j < cols; j++) {
                result[i][j] = dctRow.get(j);
            }
        }
        for (int j = 0; j < cols; j++) {
            double[] columnLegacy = new double[rows];
            for (int i = 0; i < rows; i++) {
                columnLegacy[i] = result[i][j];
            }
            IVector<Double> col = Linalg.vector(columnLegacy);
            IVector<Double> dctCol = dct2(col);
            for (int i = 0; i < rows; i++) {
                result[i][j] = dctCol.get(i);
            }
        }
        return result;
    }

    /**
     * 2D DCT逆变换 / 2D DCT Inverse Transform
     * <p>
     * 对2D DCT结果进行逆变换。
     * Perform inverse transform on 2D DCT result.
     * </p>
     *
     * @param dctMatrix 2D DCT变换结果 / 2D DCT transform result
     * @return 重建的原始矩阵 / Reconstructed original matrix
     */
    public static double[][] idct2D(double[][] dctMatrix) {
        int rows = dctMatrix.length;
        int cols = dctMatrix[0].length;
        double[][] result = new double[rows][cols];
        if (RereFastDct2.isPowerOfTwo(rows) && RereFastDct2.isPowerOfTwo(cols)) {
            for (int i = 0; i < rows; i++) {
                System.arraycopy(dctMatrix[i], 0, result[i], 0, cols);
            }
            double[] column = new double[rows];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    column[i] = result[i][j];
                }
                RereFastDct2.idct2OrthoInPlace(column, rows);
                for (int i = 0; i < rows; i++) {
                    result[i][j] = column[i];
                }
            }
            for (int i = 0; i < rows; i++) {
                RereFastDct2.idct2OrthoInPlace(result[i], cols);
            }
            return result;
        }
        for (int j = 0; j < cols; j++) {
            double[] columnLegacy = new double[rows];
            for (int i = 0; i < rows; i++) {
                columnLegacy[i] = dctMatrix[i][j];
            }
            IVector<Double> col = Linalg.vector(columnLegacy);
            IVector<Double> idctCol = idct2(col);
            for (int i = 0; i < rows; i++) {
                result[i][j] = idctCol.get(i);
            }
        }
        for (int i = 0; i < rows; i++) {
            IVector<Double> row = Linalg.vector(result[i]);
            IVector<Double> idctRow = idct2(row);
            for (int j = 0; j < cols; j++) {
                result[i][j] = idctRow.get(j);
            }
        }
        return result;
    }

    /**
     * 计算DCT的能量集中度 / Calculate DCT Energy Concentration
     * <p>
     * 计算DCT变换后前k个系数的能量占比。
     * Calculate energy ratio of first k coefficients after DCT transform.
     * </p>
     *
     * @param dctSignal DCT变换结果 / DCT transform result
     * @param k 前k个系数 / First k coefficients
     * @return 能量集中度 (0-1) / Energy concentration (0-1)
     */
    public static double energyConcentration(IVector<Double> dctSignal, int k) {
        if (k <= 0 || k > dctSignal.length()) {
            throw new IllegalArgumentException("k必须在1到信号长度之间");
        }
        
        double totalEnergy = dctSignal.multiply(dctSignal).sumValue();
        double partialEnergy = 0;
        
        for (int i = 0; i < k; i++) {
            double coeff = dctSignal.get(i);
            partialEnergy += coeff * coeff;
        }
        
        return partialEnergy / totalEnergy;
    }

    /**
     * DCT压缩 / DCT Compression
     * <p>
     * 通过保留前k个DCT系数进行信号压缩。
     * Compress signal by keeping only first k DCT coefficients.
     * </p>
     *
     * @param signal 输入信号向量 / Input signal vector
     * @param compressionRatio 压缩比 (0-1) / Compression ratio (0-1)
     * @return 压缩后的信号 / Compressed signal
     */
    public static IVector<Double> compress(IVector<Double> signal, double compressionRatio) {
        if (compressionRatio <= 0 || compressionRatio > 1) {
            throw new IllegalArgumentException("压缩比必须在(0,1]范围内");
        }
        
        // 计算DCT / Calculate DCT
        IVector<Double> dctSignal = dct2(signal);
        
        // 确定保留的系数数量 / Determine number of coefficients to keep
        int k = (int) (signal.length() * compressionRatio);
        k = Math.max(1, k); // 至少保留1个系数 / Keep at least 1 coefficient
        
        // 创建压缩的DCT信号 / Create compressed DCT signal
        IVector<Double> compressedDCT = Linalg.zeros(signal.length());
        for (int i = 0; i < k; i++) {
            compressedDCT.set(i, dctSignal.get(i));
        }
        
        // 逆变换重建信号 / Inverse transform to reconstruct signal
        return idct2(compressedDCT);
    }

    /**
     * 计算DCT的压缩误差 / Calculate DCT Compression Error
     * <p>
     * 计算压缩前后的均方误差。
     * Calculate mean squared error before and after compression.
     * </p>
     *
     * @param original 原始信号 / Original signal
     * @param compressed 压缩信号 / Compressed signal
     * @return 均方误差 / Mean squared error
     */
    public static double compressionError(IVector<Double> original, IVector<Double> compressed) {
        if (original.length() != compressed.length()) {
            throw new IllegalArgumentException("信号长度必须相同");
        }
        
        IVector<Double> error = original.sub(compressed);
        return error.multiply(error).meanValue();
    }
}
