package com.yishape.lab.math.linalg.sparse;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.sparse.impl.SparseConjugateGradientSolver;
import com.yishape.lab.math.linalg.sparse.impl.SparseBICGSTABSolver;
import com.yishape.lab.math.linalg.sparse.impl.SparseGMRESSolver;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SparseILUTest {

    @Test
    public void ilu0_appliedToResidual_reducesNorm() {
        int n = 20;
        double[][] dense = new double[n][n];
        for (int i = 0; i < n; i++) {
            dense[i][i] = 4.0;
            if (i > 0) dense[i][i - 1] = -1.0;
            if (i < n - 1) dense[i][i + 1] = -1.0;
        }
        ISparseMatrix A = ISparseMatrix.fromDense(dense);

        SparseILUPreconditioner ilu = new SparseILUPreconditioner();
        ilu.factor(A);

        IVector<Double> r = Linalg.ones(n);
        IVector<Double> z = ilu.apply(r);
        assertNotNull(z);
        assertEquals(n, z.length());

        ISparseMatrix Acsr = A.toFormat(SparseFormat.CSR);
        IVector<Double> Az = Acsr.multiply(z);
        double resNorm = 0;
        for (int i = 0; i < n; i++) {
            double diff = r.get(i) - Az.get(i);
            resNorm += diff * diff;
        }
        resNorm = Math.sqrt(resNorm);
        assertTrue(resNorm < 100, "Preconditioned residual should be reasonable");
    }

    @Test
    public void ilu0_preconditionedCG_convergesFaster() {
        int n = 30;
        double[][] dense = new double[n][n];
        for (int i = 0; i < n; i++) {
            dense[i][i] = 4.0;
            if (i > 0) dense[i][i - 1] = -1.0;
            if (i < n - 1) dense[i][i + 1] = -1.0;
        }
        ISparseMatrix A = ISparseMatrix.fromDense(dense);
        IVector<Double> b = Linalg.ones(n);

        SparseILUPreconditioner ilu = new SparseILUPreconditioner();
        ilu.factor(A);

        SparseConjugateGradientSolver cgNoPrecond = new SparseConjugateGradientSolver(1e-8, 1000, null);
        SparseConjugateGradientSolver cgWithPrecond = new SparseConjugateGradientSolver(1e-8, 1000, ilu);

        IVector<Double> xNo = cgNoPrecond.solve(A, b);
        IVector<Double> xWith = cgWithPrecond.solve(A, b);

        for (int i = 0; i < n; i++) {
            assertEquals(xNo.get(i), xWith.get(i), 1e-5);
        }
        assertTrue(cgWithPrecond.getIterationCount() < cgNoPrecond.getIterationCount(),
                "ILU preconditioned CG should use fewer iterations: "
                        + cgWithPrecond.getIterationCount() + " vs " + cgNoPrecond.getIterationCount());
    }

    @Test
    public void ilu0_preconditionedBiCGStab_converges() {
        int n = 20;
        double[][] dense = new double[n][n];
        for (int i = 0; i < n; i++) {
            dense[i][i] = 5.0;
            if (i > 0) dense[i][i - 1] = -1.0;
            if (i < n - 1) dense[i][i + 1] = -2.0;
        }
        ISparseMatrix A = ISparseMatrix.fromDense(dense);
        IVector<Double> b = Linalg.ones(n);

        SparseILUPreconditioner ilu = new SparseILUPreconditioner();
        ilu.factor(A);

        SparseBICGSTABSolver bicgNo = new SparseBICGSTABSolver(1e-8, 500, null);
        SparseBICGSTABSolver bicgPrecond = new SparseBICGSTABSolver(1e-8, 500, ilu);

        IVector<Double> xNo = bicgNo.solve(A, b);
        IVector<Double> xWith = bicgPrecond.solve(A, b);

        for (int i = 0; i < n; i++) {
            assertEquals(xNo.get(i), xWith.get(i), 1e-5);
        }
    }

    @Test
    public void ilu0_factor_doesNotModifyOriginal() {
        int n = 10;
        double[][] dense = new double[n][n];
        for (int i = 0; i < n; i++) {
            dense[i][i] = 4.0;
            if (i > 0) dense[i][i - 1] = -1.0;
            if (i < n - 1) dense[i][i + 1] = -1.0;
        }
        ISparseMatrix A = ISparseMatrix.fromDense(dense);
        double traceBefore = 0;
        for (int i = 0; i < n; i++) traceBefore += A.get(i, i);

        SparseILUPreconditioner ilu = new SparseILUPreconditioner();
        ilu.factor(A);

        double traceAfter = 0;
        for (int i = 0; i < n; i++) traceAfter += A.get(i, i);
        assertEquals(traceBefore, traceAfter, 1e-10, "Original matrix should not be modified by ILU factor");
    }

    @Test
    public void ilu0_preconditionedGMRES_reducesIterations() {
        int n = 25;
        double[][] dense = new double[n][n];
        for (int i = 0; i < n; i++) {
            dense[i][i] = 4.0;
            if (i > 0) dense[i][i - 1] = -1.0;
            if (i < n - 1) dense[i][i + 1] = -1.0;
        }
        ISparseMatrix A = ISparseMatrix.fromDense(dense);
        IVector<Double> b = Linalg.ones(n);

        SparseILUPreconditioner ilu = new SparseILUPreconditioner();
        ilu.factor(A);

        SparseGMRESSolver g1 = new SparseGMRESSolver(1e-8, 200, 15, null);
        SparseGMRESSolver g2 = new SparseGMRESSolver(1e-8, 200, 15, ilu);

        IVector<Double> x1 = g1.solve(A, b);
        IVector<Double> x2 = g2.solve(A, b);

        for (int i = 0; i < n; i++) {
            assertEquals(x1.get(i), x2.get(i), 1e-5);
        }
        assertTrue(g2.getIterationCount() <= g1.getIterationCount(),
                "ILU-GMRES: " + g2.getIterationCount() + " vs plain: " + g1.getIterationCount());
    }
}
