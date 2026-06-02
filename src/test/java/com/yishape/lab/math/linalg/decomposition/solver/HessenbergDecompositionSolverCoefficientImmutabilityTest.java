package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Hessenberg 路径求解：拷贝 H 作消元工作区后，线性系统仍正确，且缓存的 H 不被破坏。
 */
class HessenbergDecompositionSolverCoefficientImmutabilityTest {

    @Test
    void repeatedSolvePreservesHAndMatchesReference() {
        double[][] ad = {
            {4, 1, -2},
            {1, 5, 3},
            {-2, 3, 6}
        };
        IMatrix<Double> A = Linalg.matrix(ad);
        IMatrix<Double> B = Linalg.matrix(new double[][]{{1}, {2}, {3}});

        var hess = Decomps.createHessenberg();
        var hq = hess.decompose(A.copy());
        IMatrix<Double> H = hq.getFirst();
        double h01snapshot = H.get(0, 1);
        double h12snapshot = H.get(1, 2);

        var solver = hess.getSolver();
        IMatrix<Double> x1 = solver.solve(B);
        assertEquals(h01snapshot, H.get(0, 1), 1e-14);
        assertEquals(h12snapshot, H.get(1, 2), 1e-14);

        IMatrix<Double> x2 = solver.solve(B);
        assertEquals(h01snapshot, H.get(0, 1), 1e-14);
        for (int i = 0; i < 3; i++) {
            assertEquals(x1.get(i, 0), x2.get(i, 0), 1e-10);
        }
    }

    @Test
    void solveSquareSystemResidualSmallVersusLuReference() {
        double[][] ad = {
            {4, 1, -2},
            {1, 5, 3},
            {-2, 3, 6}
        };
        IMatrix<Double> A = Linalg.matrix(ad);
        IMatrix<Double> B = Linalg.matrix(new double[][]{{1}, {2}, {3}});

        var lu = Decomps.createLU();
        lu.decompose(A.copy());
        IMatrix<Double> xRef = lu.getSolver().solve(B.copy());

        var hess = Decomps.createHessenberg();
        hess.decompose(A.copy());
        IMatrix<Double> xH = hess.getSolver().solve(B);

        double res = ((Number) A.mmul(xH).sub(B).frobeniusNorm()).doubleValue();
        assertTrue(res < 1e-9, "||A x - b||_F");

        for (int i = 0; i < 3; i++) {
            assertEquals(xRef.get(i, 0), xH.get(i, 0), 1e-8);
        }
    }

    @Test
    void solveMultiRhsPreservesHAndResidualSmall() {
        double[][] ad = {
            {10, 1, 0, 0},
            {1, 10, 1, 0},
            {0, 1, 10, 1},
            {0, 0, 1, 10}
        };
        int n = 4;
        IMatrix<Double> A = Linalg.matrix(ad);
        IMatrix<Double> B = Linalg.rand(n, 3, 2027L);

        var lu = Decomps.createLU();
        lu.decompose(A.copy());
        IMatrix<Double> xRef = lu.getSolver().solve(B.copy());

        var hess = Decomps.createHessenberg();
        var hq = hess.decompose(A.copy());
        IMatrix<Double> H = hq.getFirst();
        double[][] hSnap = H.toDoubleArray();

        var solver = hess.getSolver();
        IMatrix<Double> X = solver.solve(B);

        assertEquals(hSnap[0][1], H.get(0, 1), 1e-14);
        assertEquals(hSnap[n - 2][n - 1], H.get(n - 2, n - 1), 1e-14);

        double frobDiff = ((Number) X.sub(xRef).frobeniusNorm()).doubleValue();
        assertTrue(frobDiff < 1e-8, "Hessenberg vs LU on well-conditioned A");

        IMatrix<Double> r = A.mmul(X).sub(B);
        assertTrue(((Number) r.frobeniusNorm()).doubleValue() < 1e-10);
    }

    @Test
    void vectorRhsMatchesMatrixRhs() {
        double[][] ad = {
            {2, -1, 0},
            {-1, 2, -1},
            {0, -1, 2}
        };
        IMatrix<Double> A = Linalg.matrix(ad);
        IVector<Double> b = Linalg.vector(new double[]{1.0, 0.0, 2.0});

        var hess = Decomps.createHessenberg();
        hess.decompose(A.copy());
        IVector<Double> xV = hess.getSolver().solve(b);
        IMatrix<Double> bM = Linalg.matrix(new double[][]{b.toDoubleArray()}).transpose();
        IMatrix<Double> xM = hess.getSolver().solve(bM);

        assertEquals(xV.get(0), xM.get(0, 0), 1e-10);
        assertEquals(xV.get(1), xM.get(1, 0), 1e-10);
        assertEquals(xV.get(2), xM.get(2, 0), 1e-10);

        double res = ((Number) A.mmul(xM).sub(bM).frobeniusNorm()).doubleValue();
        assertTrue(res < 1e-9);
    }
}
