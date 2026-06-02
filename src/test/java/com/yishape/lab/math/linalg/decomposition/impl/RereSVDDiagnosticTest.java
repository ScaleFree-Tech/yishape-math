package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 诊断测试：精确定位 RereSVDDecompBlas3 在大矩阵上的数值问题。
 */
public class RereSVDDiagnosticTest {

    @Test
    public void test300x300Reconstruction() {
        double[][] data = randomMatrix(300, 300, 42);
        double errFaer = reconstructionError(data, new RereSVDDecompBlas3());
        double errLegacy = reconstructionError(data, new RereSVDDecompBlas2());
        System.out.println("300x300 FaerStyle error: " + errFaer);
        System.out.println("300x300 Legacy error: " + errLegacy);
        assertTrue(errFaer < 1e-10, "FaerStyle error too large: " + errFaer);
        assertTrue(errLegacy < 1e-10, "Legacy error too large: " + errLegacy);
    }

    @Test
    public void test200x200Reconstruction() {
        double[][] data = randomMatrix(200, 200, 42);
        double errFaer = reconstructionError(data, new RereSVDDecompBlas3());
        double errLegacy = reconstructionError(data, new RereSVDDecompBlas2());
        System.out.println("200x200 FaerStyle error: " + errFaer);
        System.out.println("200x200 Legacy error: " + errLegacy);
        assertTrue(errFaer < 1e-10, "FaerStyle error too large: " + errFaer);
    }

    @Test
    public void test500x100Reconstruction() {
        double[][] data = randomMatrix(500, 100, 99);
        double errFaer = reconstructionError(data, new RereSVDDecompBlas3());
        double errLegacy = reconstructionError(data, new RereSVDDecompBlas2());
        System.out.println("500x100 FaerStyle error: " + errFaer);
        System.out.println("500x100 Legacy error: " + errLegacy);
        assertTrue(errFaer < 1e-10, "FaerStyle error too large: " + errFaer);
    }

    @Test
    public void test1000x200Reconstruction() {
        double[][] data = randomMatrix(1000, 200, 99);
        double errFaer = reconstructionError(data, new RereSVDDecompBlas3());
        double errLegacy = reconstructionError(data, new RereSVDDecompBlas2());
        System.out.println("1000x200 FaerStyle error: " + errFaer);
        System.out.println("1000x200 Legacy error: " + errLegacy);
        assertTrue(errFaer < 1e-10, "FaerStyle error too large: " + errFaer);
    }

    @Test
    public void testBlockedPathOnlyBidiagonal() {
        // 强制走 blocked 路径，检查 bidiagonal 后 A 的 bidiagonal 性质
        RereSVDDecompBlas3 svd = new RereSVDDecompBlas3();
        double[][] data = randomMatrix(300, 300, 42);
        double[][] A = copy(data);

        // 通过反射调用 blockedBidiagonalSVD，提取 d/e 后检查 Q^T A P = B
        // 这里简化：直接用 scalar 路径的 bidiagonalSVD 与 blocked 路径比较 singular values
        var r1 = svd.decompose(Linalg.matrix(data));
        double[] s1 = new double[300];
        for (int i = 0; i < 300; i++) s1[i] = r1.getSecond().get(i);

        RereSVDDecompBlas2 legacy = new RereSVDDecompBlas2();
        var r2 = legacy.decompose(Linalg.matrix(data));
        double[] s2 = new double[300];
        for (int i = 0; i < 300; i++) s2[i] = r2.getSecond().get(i);

        double maxDiff = 0.0;
        for (int i = 0; i < 300; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(s1[i] - s2[i]));
        }
        System.out.println("Max singular value diff (Faer vs Legacy) for 300x300: " + maxDiff);
        assertTrue(maxDiff < 1e-6, "Singular values diverge too much: " + maxDiff);
    }

    private double reconstructionError(double[][] data, com.yishape.lab.math.linalg.decomposition.ISVDDecomposition svd) {
        IMatrix<Double> A = Linalg.matrix(data);
        var r = svd.decompose(A);
        IMatrix<Double> U = r.getFirst();
        var S = r.getSecond();
        IMatrix<Double> VT = r.getThird();
        int m = A.rows(), n = A.cols(), k = Math.min(m, n);
        double maxErr = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += U.get(i, l) * S.get(l) * VT.get(l, j);
                }
                maxErr = Math.max(maxErr, Math.abs(sum - data[i][j]));
            }
        }
        return maxErr;
    }

    private static double[][] randomMatrix(int rows, int cols, long seed) {
        Random rng = new Random(seed);
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = rng.nextGaussian();
            }
        }
        return data;
    }

    private static double[][] copy(double[][] src) {
        double[][] dst = new double[src.length][];
        for (int i = 0; i < src.length; i++) {
            dst[i] = src[i].clone();
        }
        return dst;
    }
}
