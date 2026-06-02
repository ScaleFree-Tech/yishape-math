package com.yishape.lab.math.ml.dml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.util.Tuple2;

/**
 * 核函数工具类，供核化 DML 算法使用。
 */
public final class KernelDmlUtils {

    public enum KernelType {
        LINEAR,
        RBF,
        POLY,
        SIGMOID,
        COSINE
    }

    private KernelDmlUtils() {}

    /**
     * 计算核矩阵 K(X, Y)。
     */
    public static double[][] kernelMatrix(double[][] X, double[][] Y, KernelType type, double gamma, int degree, double coef0) {
        int nX = X.length;
        int nY = Y.length;
        int d = X[0].length;
        double[][] K = new double[nX][nY];

        if (type == KernelType.LINEAR) {
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nY; j++) {
                    double s = 0;
                    for (int k = 0; k < d; k++) {
                        s += X[i][k] * Y[j][k];
                    }
                    K[i][j] = s;
                }
            }
        } else if (type == KernelType.RBF) {
            if (gamma <= 0) gamma = 1.0 / d;
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nY; j++) {
                    double s = 0;
                    for (int k = 0; k < d; k++) {
                        double diff = X[i][k] - Y[j][k];
                        s += diff * diff;
                    }
                    K[i][j] = Math.exp(-gamma * s);
                }
            }
        } else if (type == KernelType.POLY) {
            if (gamma <= 0) gamma = 1.0;
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nY; j++) {
                    double s = 0;
                    for (int k = 0; k < d; k++) {
                        s += X[i][k] * Y[j][k];
                    }
                    K[i][j] = Math.pow(gamma * s + coef0, degree);
                }
            }
        } else if (type == KernelType.SIGMOID) {
            if (gamma <= 0) gamma = 1.0 / d;
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nY; j++) {
                    double s = 0;
                    for (int k = 0; k < d; k++) {
                        s += X[i][k] * Y[j][k];
                    }
                    K[i][j] = Math.tanh(gamma * s + coef0);
                }
            }
        } else if (type == KernelType.COSINE) {
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nY; j++) {
                    double dot = 0, normX = 0, normY = 0;
                    for (int k = 0; k < d; k++) {
                        dot += X[i][k] * Y[j][k];
                        normX += X[i][k] * X[i][k];
                        normY += Y[j][k] * Y[j][k];
                    }
                    K[i][j] = dot / (Math.sqrt(normX) * Math.sqrt(normY) + 1e-10);
                }
            }
        }
        return K;
    }

    /**
     * 中心化核矩阵。
     */
    public static double[][] centerKernelMatrix(double[][] K) {
        int n = K.length;
        double[][] Kc = new double[n][n];
        double[] colSum = new double[n];
        double[] rowSum = new double[n];
        double total = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                colSum[j] += K[i][j];
                rowSum[i] += K[i][j];
                total += K[i][j];
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Kc[i][j] = K[i][j] - rowSum[i] / n - colSum[j] / n + total / (n * n);
            }
        }
        return Kc;
    }

    /**
     * 核矩阵特征分解，返回 (特征值, 特征向量) 元组。
     */
    public static Tuple2<IVector<Double>, IMatrix<Double>> kernelEigen(double[][] K) {
        IMatrix<Double> km = IMatrix.of(K);
        return km.eigen();
    }
}
