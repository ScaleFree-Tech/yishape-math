package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * 随机双对角矩阵上分治合并（含非零连接元）与 QR 双对角 SVD 奇异值一致。
 */
class RereSVDBidiagonalDcMergeTest {

    @Test
    void divideConquerBidiagonalSingularValuesMatchQr() {
        Random rnd = new Random(7);
        RereSVDDecomposition svd = new RereSVDDecomposition(1e-14, 5000);
        for (int n = 9; n <= 36; n++) {
            for (int t = 0; t < 6; t++) {
                double[][] b = randomUpperBidiagonal(rnd, n);
                IMatrix<Double> B = Linalg.matrix(b);
                Tuple3<com.yishape.lab.math.linalg.IVector<Double>, IMatrix<Double>, IMatrix<Double>> dc =
                        svd.divideAndConquerBidiagonalSVDForTesting(B);
                Tuple3<com.yishape.lab.math.linalg.IVector<Double>, IMatrix<Double>, IMatrix<Double>> qr =
                        svd.qrBidiagonalForTesting(B);
                int m = dc.getFirst().length();
                double[] a = new double[m];
                double[] c = new double[m];
                for (int i = 0; i < m; i++) {
                    a[i] = Math.abs(dc.getFirst().get(i));
                    c[i] = Math.abs(qr.getFirst().get(i));
                }
                Arrays.sort(a);
                Arrays.sort(c);
                for (int i = 0; i < m; i++) {
                    assertEquals(c[i], a[i], 1e-7 * (1.0 + Math.abs(c[i])),
                            "n=" + n + " trial=" + t + " i=" + i);
                }
            }
        }
    }

    private static double[][] randomUpperBidiagonal(Random rnd, int n) {
        double[][] b = new double[n][n];
        for (int i = 0; i < n; i++) {
            b[i][i] = rnd.nextGaussian();
        }
        for (int i = 0; i < n - 1; i++) {
            b[i][i + 1] = rnd.nextGaussian();
        }
        return b;
    }
}
