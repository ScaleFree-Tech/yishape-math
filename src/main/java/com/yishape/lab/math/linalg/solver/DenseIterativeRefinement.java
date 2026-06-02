package com.yishape.lab.math.linalg.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;

/**
 * Standard iterative refinement for triangular solves (LAPACK GERFS-style fixed-point refinement,
 * all operations in double precision).
 */
public final class DenseIterativeRefinement {

    private static final int MAX_STEPS = 10;
    private static final double BASE_TOL = 1e-14;

    private DenseIterativeRefinement() {
    }

    /**
     * Refines {@code X} for {@code A×X ≈ B} using the same factorization-backed {@code solver}
     * to solve correction systems: solve {@code A×Δ = B − A×X}, then {@code X ← X + Δ}.
     */
    public static IMatrix<Double> refine(
            IMatrix<Double> a,
            IMatrix<Double> b,
            IMatrix<Double> x0,
            IDecompositionSolver solver) {
        if (x0 == null || !solver.isNonSingular()) {
            return x0;
        }
        IMatrix<Double> x = x0;
        int nRow = b.rows();
        int nRhs = b.cols();
        double normB = Math.max(1.0, b.frobeniusNorm());
        double tol = BASE_TOL * Math.max(1, nRow) * Math.max(1, nRhs);

        for (int k = 0; k < MAX_STEPS; k++) {
            IMatrix<Double> ax;
            try {
                ax = a.mmul(x);
            } catch (Exception e) {
                break;
            }
            IMatrix<Double> res = b.sub(ax);
            double nres = res.frobeniusNorm();
            if (nres <= tol * normB) {
                break;
            }
            IMatrix<Double> dx;
            try {
                dx = solver.solve(res);
            } catch (Exception e) {
                break;
            }
            double ndx = dx.frobeniusNorm();
            double nx = x.frobeniusNorm();
            if (ndx > 1e12 * Math.max(1.0, nx) || Double.isNaN(ndx)) {
                break;
            }
            x = x.add(dx);
            if (ndx <= 1e-18 * Math.max(1.0, nx)) {
                break;
            }
        }
        return x;
    }

    /**
     * ‖AX − B‖_F / max(1, ‖B‖_F) — for monitoring least-squares / solve quality.
     */
    public static double relativeResidualFrobenius(IMatrix<Double> a, IMatrix<Double> b, IMatrix<Double> x) {
        double nb = Math.max(1.0, b.frobeniusNorm());
        IMatrix<Double> r = a.mmul(x).sub(b);
        return r.frobeniusNorm() / nb;
    }
}
