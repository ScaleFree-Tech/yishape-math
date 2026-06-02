package com.yishape.lab.math.linalg.sparse;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.sparse.impl.SparseConjugateGradientSolver;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SparseCGSolverTest {

    @Test
    public void solve_diagonalSPD_convergesInOneIteration() {
        ISparseMatrix A = ISparseMatrix.diag(new double[]{4, 9, 16});
        IVector<Double> b = Linalg.vector(new double[]{8, 27, 64});
        SparseConjugateGradientSolver cg = new SparseConjugateGradientSolver();

        IVector<Double> x = cg.solve(A, b);
        assertEquals(2.0, x.get(0), 1e-6);
        assertEquals(3.0, x.get(1), 1e-6);
        assertEquals(4.0, x.get(2), 1e-6);
        assertTrue(cg.getIterationCount() <= 3, "CG should converge quickly on diagonal SPD");
        assertTrue(cg.getResidual() < 1e-7);
    }

    @Test
    public void solve_identitySPD_convergesInOneIteration() {
        ISparseMatrix A = ISparseMatrix.eye(5);
        IVector<Double> b = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        SparseConjugateGradientSolver cg = new SparseConjugateGradientSolver();

        IVector<Double> x = cg.solve(A, b);
        for (int i = 0; i < 5; i++) {
            assertEquals(b.get(i), x.get(i), 1e-6);
        }
        assertTrue(cg.getIterationCount() <= 1, "Identity should converge in at most 1 iteration");
    }

    @Test
    public void solve_smallSPD_matchesDenseLU() {
        double[][] Ad = {{4, 1, 0}, {1, 4, 1}, {0, 1, 4}};
        ISparseMatrix A = ISparseMatrix.fromDense(Ad);
        IVector<Double> b = Linalg.vector(new double[]{5, 6, 5});
        SparseConjugateGradientSolver cg = new SparseConjugateGradientSolver();

        IVector<Double> x = cg.solve(A, b);

        IDoubleMatrix Adense = IDoubleMatrix.of(Ad);
        IVector<Double> xRef = Adense.solve(b);

        assertEquals(xRef.get(0), x.get(0), 1e-6);
        assertEquals(xRef.get(1), x.get(1), 1e-6);
        assertEquals(xRef.get(2), x.get(2), 1e-6);
    }

    @Test
    public void solve_noPreconditioner_smallSPD_converges() {
        double[][] Ad = {{4, 1}, {1, 3}};
        ISparseMatrix A = ISparseMatrix.fromDense(Ad);
        IVector<Double> b = Linalg.vector(new double[]{1, 2});
        SparseConjugateGradientSolver cg = new SparseConjugateGradientSolver(1e-8, 100, false);

        IVector<Double> x = cg.solve(A, b);
        assertTrue(cg.getResidual() < 1e-7);

        IDoubleMatrix Adense = IDoubleMatrix.of(Ad);
        IVector<Double> xRef = Adense.solve(b);
        assertEquals(xRef.get(0), x.get(0), 1e-6);
        assertEquals(xRef.get(1), x.get(1), 1e-6);
    }

    @Test
    public void solve_withInitialGuess_improvesConvergence() {
        double[][] Ad = {{5, 2}, {2, 3}};
        ISparseMatrix A = ISparseMatrix.fromDense(Ad);
        IVector<Double> b = Linalg.vector(new double[]{7, 5});
        IVector<Double> x0 = Linalg.vector(new double[]{0.9, 0.7});
        SparseConjugateGradientSolver cg = new SparseConjugateGradientSolver();

        IVector<Double> x = cg.solve(A, b, x0);

        IDoubleMatrix Adense = IDoubleMatrix.of(Ad);
        IVector<Double> xRef = Adense.solve(b);
        assertEquals(xRef.get(0), x.get(0), 1e-6);
        assertEquals(xRef.get(1), x.get(1), 1e-6);
    }

    @Test
    public void solve_zeroRightHandSide_producesZero() {
        ISparseMatrix A = ISparseMatrix.diag(new double[]{2, 3, 4});
        IVector<Double> b = Linalg.vector(new double[]{0, 0, 0});
        SparseConjugateGradientSolver cg = new SparseConjugateGradientSolver();

        IVector<Double> x = cg.solve(A, b);
        assertEquals(0.0, x.get(0), 1e-10);
        assertEquals(0.0, x.get(1), 1e-10);
        assertEquals(0.0, x.get(2), 1e-10);
    }
}
