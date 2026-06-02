package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * QR 矩阵/向量求解：深拷贝 RHS 后功能仍正确，且不得破坏调用方传入的 B / b。
 */
class QRDecompositionSolverRhsImmutabilityTest {

    @Test
    void solveMatrixRhsLeavesInputBUnchanged() {
        double[][] ad = {
            {1, 0},
            {0, 1},
            {1, 1}
        };
        double[] bd = {1, 0, 1};
        IMatrix<Double> A = Linalg.matrix(ad);
        IMatrix<Double> B = Linalg.matrix(new double[][]{bd}).transpose();

        var qr = Decomps.createQR();
        qr.decompose(A.copy());
        qr.getSolver().solve(B);

        assertEquals(1.0, B.get(0, 0), 0.0);
        assertEquals(0.0, B.get(1, 0), 0.0);
        assertEquals(1.0, B.get(2, 0), 0.0);
    }

    @Test
    void solveMatrixRhsLeastSquaresResidualAndReferenceMatch() {
        double[][] ad = {
            {1, 0},
            {0, 1},
            {1, 1}
        };
        // b = A * [1, 0]^T，该超定组在最小二乘意义下可精确达到
        double[] bd = {1, 0, 1};
        IMatrix<Double> A = Linalg.matrix(ad);
        IMatrix<Double> B = Linalg.matrix(new double[][]{bd}).transpose();

        var qr = Decomps.createQR();
        qr.decompose(A.copy());
        IMatrix<Double> X = qr.getSolver().solve(B);

        double r = ((Number) A.mmul(X).sub(B).frobeniusNorm()).doubleValue();
        assertTrue(r < 1e-10, "||AX-B||_F should be tiny for consistent RHS");

        assertEquals(1.0, X.get(0, 0), 1e-10);
        assertEquals(0.0, X.get(1, 0), 1e-10);
    }

    @Test
    void solveMultiColumnRhsPreservesBAndMatchesConsistentSystem() {
        double[][] ad = {
            {2, -1},
            {1, 3},
            {0, 1}
        };
        IMatrix<Double> A = Linalg.matrix(ad);
        IMatrix<Double> Xtrue = Linalg.rand(A.cols(), 2, 101L);
        IMatrix<Double> B = A.mmul(Xtrue);

        double b00 = B.get(0, 0);
        double b10 = B.get(1, 0);
        double b20 = B.get(2, 0);
        double b01 = B.get(0, 1);
        double b11 = B.get(1, 1);
        double b21 = B.get(2, 1);

        var qr = Decomps.createQR();
        qr.decompose(A.copy());
        IMatrix<Double> X = qr.getSolver().solve(B);

        assertEquals(b00, B.get(0, 0), 0.0);
        assertEquals(b10, B.get(1, 0), 0.0);
        assertEquals(b20, B.get(2, 0), 0.0);
        assertEquals(b01, B.get(0, 1), 0.0);
        assertEquals(b11, B.get(1, 1), 0.0);
        assertEquals(b21, B.get(2, 1), 0.0);

        double frob = ((Number) X.sub(Xtrue).frobeniusNorm()).doubleValue();
        assertTrue(frob < 1e-8, "batch solve should recover X with B in range(A)");

        double res = ((Number) A.mmul(X).sub(B).frobeniusNorm()).doubleValue();
        assertTrue(res < 1e-9);
    }

    @Test
    void solveVectorRhsUnchangedAndMatchesMatrixPath() {
        double[][] ad = {
            {1, 2},
            {3, 4},
            {5, 6}
        };
        IMatrix<Double> A = Linalg.matrix(ad);
        IVector<Double> b = Linalg.vector(new double[]{1.0, 0.0, -0.5});

        double b0 = b.get(0);
        double b1 = b.get(1);
        double b2 = b.get(2);

        var qr = Decomps.createQR();
        qr.decompose(A.copy());
        IVector<Double> xVec = qr.getSolver().solve(b);

        assertEquals(b0, b.get(0), 0.0);
        assertEquals(b1, b.get(1), 0.0);
        assertEquals(b2, b.get(2), 0.0);

        IMatrix<Double> bCol = Linalg.matrix(new double[][]{b.toDoubleArray()}).transpose();
        IMatrix<Double> xMat = qr.getSolver().solve(bCol);
        assertEquals(xVec.get(0), xMat.get(0, 0), 1e-10);
        assertEquals(xVec.get(1), xMat.get(1, 0), 1e-10);
    }
}
