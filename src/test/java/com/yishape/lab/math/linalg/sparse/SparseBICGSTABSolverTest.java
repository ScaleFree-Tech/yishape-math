package com.yishape.lab.math.linalg.sparse;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.sparse.impl.SparseBICGSTABSolver;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SparseBICGSTABSolverTest {

    @Test
    public void solve_diagonalMatrix_convergesImmediately() {
        ISparseMatrix A = ISparseMatrix.diag(new double[]{2, 5, 10});
        IVector<Double> b = Linalg.vector(new double[]{6, 15, 40});
        SparseBICGSTABSolver bicg = new SparseBICGSTABSolver();

        IVector<Double> x = bicg.solve(A, b);
        assertEquals(3.0, x.get(0), 1e-6);
        assertEquals(3.0, x.get(1), 1e-6);
        assertEquals(4.0, x.get(2), 1e-6);
    }

    @Test
    public void solve_symmetricPositiveDefinite_matchesDirectSolver() {
        double[][] Ad = {{4, 1, 0}, {1, 4, 1}, {0, 1, 4}};
        ISparseMatrix A = ISparseMatrix.fromDense(Ad);
        IVector<Double> b = Linalg.vector(new double[]{5, 6, 5});
        SparseBICGSTABSolver bicg = new SparseBICGSTABSolver();

        IVector<Double> x = bicg.solve(A, b);
        assertTrue(bicg.getResidual() < 1e-7);

        IDoubleMatrix Adense = IDoubleMatrix.of(Ad);
        IVector<Double> xRef = Adense.solve(b);
        assertEquals(xRef.get(0), x.get(0), 1e-6);
        assertEquals(xRef.get(1), x.get(1), 1e-6);
        assertEquals(xRef.get(2), x.get(2), 1e-6);
    }

    @Test
    public void solve_nonsymmetricMatrix_converges() {
        double[][] Ad = {{2, 1, 0}, {0, 2, 1}, {1, 0, 3}};
        ISparseMatrix A = ISparseMatrix.fromDense(Ad);
        IVector<Double> b = Linalg.vector(new double[]{3, 3, 4});
        SparseBICGSTABSolver bicg = new SparseBICGSTABSolver();

        IVector<Double> x = bicg.solve(A, b);
        assertTrue(bicg.getResidual() < 1e-6);

        IDoubleMatrix Adense = IDoubleMatrix.of(Ad);
        IVector<Double> xRef = Adense.solve(b);
        assertEquals(xRef.get(0), x.get(0), 1e-5);
        assertEquals(xRef.get(1), x.get(1), 1e-5);
        assertEquals(xRef.get(2), x.get(2), 1e-5);
    }

    @Test
    public void solve_withInitialGuess_converges() {
        double[][] Ad = {{3, 1}, {1, 3}};
        ISparseMatrix A = ISparseMatrix.fromDense(Ad);
        IVector<Double> b = Linalg.vector(new double[]{4, 4});
        IVector<Double> x0 = Linalg.vector(new double[]{0.5, 0.5});
        SparseBICGSTABSolver bicg = new SparseBICGSTABSolver();

        IVector<Double> x = bicg.solve(A, b, x0);
        assertTrue(bicg.getResidual() < 1e-7);
        assertEquals(1.0, x.get(0), 1e-6);
        assertEquals(1.0, x.get(1), 1e-6);
    }

    @Test
    public void solve_identityMatrix_trivialCase() {
        ISparseMatrix A = ISparseMatrix.eye(4);
        IVector<Double> b = Linalg.vector(new double[]{1, 2, 3, 4});
        SparseBICGSTABSolver bicg = new SparseBICGSTABSolver();

        IVector<Double> x = bicg.solve(A, b);
        assertEquals(1.0, x.get(0));
        assertEquals(2.0, x.get(1));
        assertEquals(3.0, x.get(2));
        assertEquals(4.0, x.get(3));
    }
}
