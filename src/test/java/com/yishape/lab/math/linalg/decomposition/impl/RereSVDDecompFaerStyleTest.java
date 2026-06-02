package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.ISVDDecomposition;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Smoke tests for the BLAS-3 (Faer-style) SVD implementation. Verifies reconstruction
 * accuracy against the reference BLAS-2 path (RereSVDDecomposition).
 */
public class RereSVDDecompFaerStyleTest {

    private static final double RECON_TOL = 1e-9;

    @Test
    public void testTinySquare() {
        double[][] data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 10.0}
        };
        verifyReconstruction(data);
    }

    @Test
    public void test4x4() {
        double[][] data = {
            {1.0,  2.0,  3.0,  4.0},
            {5.0,  4.0,  3.0,  2.0},
            {1.5,  0.5,  6.5,  9.5},
            {7.0,  8.0,  10.0, 11.0}
        };
        verifyReconstruction(data);
    }

    @Test
    public void test10x10Random() {
        double[][] data = randomMatrix(10, 10, 42);
        verifyReconstruction(data);
    }

    @Test
    public void test50x50Random() {
        double[][] data = randomMatrix(50, 50, 7);
        verifyReconstruction(data);
    }

    @Test
    public void test64x64BlockBoundary() {
        // 触发 LABRD 分块路径（kMin >= 64）
        double[][] data = randomMatrix(64, 64, 11);
        verifyReconstruction(data);
    }

    @Test
    public void test100x100Blocked() {
        double[][] data = randomMatrix(100, 100, 13);
        verifyReconstruction(data);
    }

    @Test
    public void testTallMatrix100x50() {
        double[][] data = randomMatrix(100, 50, 17);
        verifyReconstruction(data);
    }

    @Test
    public void testWideMatrix50x100() {
        double[][] data = randomMatrix(50, 100, 19);
        verifyReconstruction(data);
    }

    private void verifyReconstruction(double[][] data) {
        IMatrix<Double> A = Linalg.matrix(data);
        ISVDDecomposition svd = new RereSVDDecompBlas3();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();

        assertNotNull(U);
        assertNotNull(S);
        assertNotNull(VT);

        int m = A.rows();
        int n = A.cols();
        int k = Math.min(m, n);
        assertEquals(m, U.rows(), "U rows");
        assertEquals(k, U.cols(), "U cols (thin SVD)");
        assertEquals(k, S.length(), "S length");
        assertEquals(n, VT.rows(), "VT rows");
        assertEquals(n, VT.cols(), "VT cols");

        // 重构：A_reconstructed[i][j] = sum_l U[i][l] * S[l] * VT[l][j]
        double[][] recon = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += U.get(i, l) * S.get(l) * VT.get(l, j);
                }
                recon[i][j] = sum;
            }
        }
        double maxErr = 0.0;
        double normA = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double diff = Math.abs(recon[i][j] - data[i][j]);
                if (diff > maxErr) maxErr = diff;
                normA += data[i][j] * data[i][j];
            }
        }
        normA = Math.sqrt(normA);
        double relErr = maxErr / Math.max(normA, 1e-300);
        assertTrue(relErr < RECON_TOL,
                "Reconstruction relative error too large: " + relErr + " (max abs " + maxErr + ", ||A||=" + normA + ")");

        // 奇异值非负且降序
        for (int i = 0; i < k; i++) {
            assertTrue(S.get(i) >= -1e-12, "singular value " + i + " negative: " + S.get(i));
        }
        for (int i = 1; i < k; i++) {
            assertTrue(S.get(i - 1) >= S.get(i) - 1e-12,
                    "singular values not sorted descending at " + i);
        }
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
}
