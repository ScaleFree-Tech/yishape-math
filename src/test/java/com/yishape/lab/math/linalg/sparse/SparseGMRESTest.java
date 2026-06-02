package com.yishape.lab.math.linalg.sparse;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.sparse.impl.SparseGMRESSolver;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SparseGMRESTest {

    @Test
    public void solve_diagonal_convergesQuickly() {
        ISparseMatrix A = ISparseMatrix.diag(new double[]{4, 9, 16, 25});
        IVector<Double> b = Linalg.vector(new double[]{8, 27, 64, 125});
        SparseGMRESSolver gmres = new SparseGMRESSolver();

        IVector<Double> x = gmres.solve(A, b);
        assertEquals(2.0, x.get(0), 1e-6);
        assertEquals(3.0, x.get(1), 1e-6);
        assertEquals(4.0, x.get(2), 1e-6);
        assertEquals(5.0, x.get(3), 1e-6);
        assertTrue(gmres.getResidual() < 1e-7);
    }

    @Test
    public void solve_nonsymmetric_matchesDenseLU() {
        double[][] Ad = {{4, 1, 2}, {0, 3, 1}, {1, 0, 5}};
        ISparseMatrix A = ISparseMatrix.fromDense(Ad);
        IVector<Double> b = Linalg.vector(new double[]{10, 7, 12});
        SparseGMRESSolver gmres = new SparseGMRESSolver();

        IVector<Double> x = gmres.solve(A, b);
        IDoubleMatrix Adense = IDoubleMatrix.of(Ad);
        IVector<Double> xRef = Adense.solve(b);

        for (int i = 0; i < xRef.length(); i++) {
            assertEquals(xRef.get(i), x.get(i), 1e-5);
        }
    }

    @Test
    public void solve_withRestart_solvesCorrectly() {
        double[][] Ad = {{5, 2, 1}, {0, 4, 2}, {1, 1, 6}};
        ISparseMatrix A = ISparseMatrix.fromDense(Ad);
        IVector<Double> b = Linalg.vector(new double[]{10, 8, 15});
        SparseGMRESSolver gmres = new SparseGMRESSolver(1e-8, 100, 2); // small restart

        IVector<Double> x = gmres.solve(A, b);
        IDoubleMatrix Adense = IDoubleMatrix.of(Ad);
        IVector<Double> xRef = Adense.solve(b);

        for (int i = 0; i < xRef.length(); i++) {
            assertEquals(xRef.get(i), x.get(i), 1e-5);
        }
    }

    @Test
    public void solve_withILUPreconditioner_reducesIterations() {
        int n = 20;
        double[][] dense = new double[n][n];
        for (int i = 0; i < n; i++) {
            dense[i][i] = 4.0;
            if (i > 0) dense[i][i - 1] = -1.0;
            if (i < n - 1) dense[i][i + 1] = -1.0;
        }
        ISparseMatrix A = ISparseMatrix.fromDense(dense);
        IVector<Double> b = Linalg.vector(new double[n]);
        b.set(0, 3.0);
        b.set(n - 1, 3.0);
        for (int i = 1; i < n - 1; i++) b.set(i, 2.0);

        SparseGMRESSolver gmresNoPrecond = new SparseGMRESSolver(1e-8, 200, 20, null);
        SparseILUPreconditioner ilu = new SparseILUPreconditioner();
        ilu.factor(A);
        SparseGMRESSolver gmresWithPrecond = new SparseGMRESSolver(1e-8, 200, 20, ilu);

        IVector<Double> xNo = gmresNoPrecond.solve(A, b);
        IVector<Double> xWith = gmresWithPrecond.solve(A, b);

        for (int i = 0; i < n; i++) {
            assertEquals(xNo.get(i), xWith.get(i), 1e-5);
        }
        assertTrue(gmresWithPrecond.getIterationCount() <= gmresNoPrecond.getIterationCount(),
                "ILU preconditioned GMRES should converge in fewer or equal iterations");
    }

    @Test
    public void solve_identity_convergesInOneIteration() {
        ISparseMatrix A = ISparseMatrix.eye(10);
        IVector<Double> b = Linalg.ones(10);
        SparseGMRESSolver gmres = new SparseGMRESSolver();

        IVector<Double> x = gmres.solve(A, b);
        for (int i = 0; i < 10; i++) {
            assertEquals(1.0, x.get(i), 1e-8);
        }
        assertTrue(gmres.getIterationCount() <= 2);
    }
}
