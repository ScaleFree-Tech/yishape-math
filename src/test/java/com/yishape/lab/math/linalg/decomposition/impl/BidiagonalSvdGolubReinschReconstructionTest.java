package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link RereSVDDecompBlas2#bidiagonalSVD}（Golub–Reinsch 约化 + BD-SQR）对矩形矩阵
 * 的全矩阵重构误差应与 optimized 路径同量级，不应出现 O(1) 差错。
 */
class BidiagonalSvdGolubReinschReconstructionTest {

    @Test
    void tallWideRandomHasSmallRelativeFrobeniusError() {
        var rng = new java.util.Random(20260509);
        for (int trial = 0; trial < 8; trial++) {
            IMatrix<Double> a = randomMatrix(rng, 80, 60);
            assertSmallRecon("80x60 t=" + trial, a);
        }
    }

    @Test
    void sinCos80x60LikeOptimizedSvdTestHasSmallRelativeError() {
        double[][] data = new double[80][60];
        for (int i = 0; i < 80; i++) {
            for (int j = 0; j < 60; j++) {
                data[i][j] = Math.sin(0.1 * i + 0.2 * j) + 0.5 * Math.cos(0.15 * i - 0.1 * j);
            }
        }
        IMatrix<Double> a = Linalg.matrix(data);
        assertSmallRecon("sin/cos 80x60", a);
    }

    @Test
    void squareHasSmallRelativeFrobeniusError() {
        var rng = new java.util.Random(20260509);
        for (int trial = 0; trial < 8; trial++) {
            IMatrix<Double> a = randomMatrix(rng, 40, 40);
            assertSmallRecon("40x40 t=" + trial, a);
        }
    }

    private static IMatrix<Double> randomMatrix(java.util.Random rng, int rows, int cols) {
        double[][] d = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                d[i][j] = rng.nextGaussian();
            }
        }
        return Linalg.matrix(d);
    }

    private static void assertSmallRecon(String msg, IMatrix<Double> a) {
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> t = svd.decompose(a);
        IMatrix<Double> u = t.getFirst();
        IVector<Double> s = t.getSecond();
        IMatrix<Double> vt = t.getThird();

        int k = s.length();
        IMatrix<Double> sm = Linalg.zeros(k, k);
        for (int i = 0; i < k; i++) {
            sm.set(i, i, s.get(i));
        }
        IMatrix<Double> recon = u.mmul(sm).mmul(vt);

        double na = 0.0;
        double err = 0.0;
        for (int i = 0; i < a.rows(); i++) {
            for (int j = 0; j < a.cols(); j++) {
                double dij = a.get(i, j) - recon.get(i, j);
                err += dij * dij;
                na += a.get(i, j) * a.get(i, j);
            }
        }
        double rel = Math.sqrt(err / Math.max(1e-300, na));
        assertTrue(rel < 1e-8, () -> msg + " relative Frobenius error=" + rel);
    }
}
