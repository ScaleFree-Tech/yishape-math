package com.yishape.lab.math.linalg;

import com.yishape.lab.math.linalg.decomposition.impl.RereBidiagonalDecomposition;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class BidiagonalizationVerificationTest {

    private static final double TOL = 1e-9;

    @Test
    public void testBidiagonalizationAccuracy() {
        // 4x3 test matrix (rank 2)
        double[][] testData = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {10.0, 11.0, 12.0}
        };

        IDoubleMatrix A = IDoubleMatrix.of(testData);

        // 执行双对角化
        RereBidiagonalDecomposition bidiag = new RereBidiagonalDecomposition();
        Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> result = bidiag.decompose(A);

        IDoubleMatrix U = (IDoubleMatrix) result.getFirst();
        IDoubleMatrix B = (IDoubleMatrix) result.getSecond();
        IDoubleMatrix V = (IDoubleMatrix) result.getThird();

        // 验证重构：A = U * B * V^T
        IDoubleMatrix VT = (IDoubleMatrix) V.transpose();
        IDoubleMatrix reconstructed = (IDoubleMatrix) ((IDoubleMatrix) U.mmul(B)).mmul(VT);

        // Assert reconstruction error is small
        double maxError = 0.0;
        for (int i = 0; i < A.getRowNum(); i++) {
            for (int j = 0; j < A.getColNum(); j++) {
                double error = Math.abs(A.get(i, j) - reconstructed.get(i, j));
                maxError = Math.max(maxError, error);
            }
        }
        assertTrue(maxError < 1e-6,
            "Bidiagonal reconstruction max error should be < 1e-6, got: " + maxError);

        // 验证U的正交性: U^T * U ≈ I
        IDoubleMatrix UT = (IDoubleMatrix) U.transpose();
        IDoubleMatrix UTU = (IDoubleMatrix) UT.mmul(U);
        int k = U.getColNum();
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, UTU.get(i, j), TOL,
                    "U^T*U[" + i + "," + j + "] should be " + expected);
            }
        }

        // 验证V的正交性: V^T * V ≈ I
        IDoubleMatrix VTV = (IDoubleMatrix) VT.mmul(V);
        int n = V.getColNum();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, VTV.get(i, j), TOL,
                    "V^T*V[" + i + "," + j + "] should be " + expected);
            }
        }

        // 验证B为双对角矩阵
        for (int i = 0; i < B.getRowNum(); i++) {
            for (int j = 0; j < B.getColNum(); j++) {
                if (j != i && j != i + 1) {
                    assertEquals(0.0, B.get(i, j), TOL,
                        "B[" + i + "," + j + "] should be zero for bidiagonal matrix");
                }
            }
        }
    }
}