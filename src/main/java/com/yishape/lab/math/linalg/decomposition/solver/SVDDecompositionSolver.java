package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * Solver for singular value decomposition.
 * <p>
 * This solver uses the SVD decomposition A = U * S * V^T to solve
 * linear systems. For a non-singular matrix, the solution to A * X = B is
 * X = V * S^(-1) * U^T * B.
 * </p>
 * 
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 * </ul>
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class SVDDecompositionSolver implements IDecompositionSolver {
    
    /** The U matrix. */
    private final IMatrix<Double> uMatrix;
    /** The singular values. */
    private final IVector<Double> singularValues;
    /** The V^T matrix. */
    private final IMatrix<Double> vtMatrix;
    /** Epsilon for numerical comparisons. */
    private final double epsilon;
    
    /**
     * Create a solver from SVD decomposition results.
     * 
     * @param uMatrix the U matrix
     * @param singularValues the singular values
     * @param vtMatrix the V^T matrix
     * @param epsilon threshold for considering an element as zero
     */
    public SVDDecompositionSolver(IMatrix<Double> uMatrix, IVector<Double> singularValues, IMatrix<Double> vtMatrix, double epsilon) {
        this.uMatrix = uMatrix;
        this.singularValues = singularValues;
        this.vtMatrix = vtMatrix;
        this.epsilon = epsilon;
    }
    
    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        return pseudoinverse().mmul(b);
    }
    
    @Override
    public IVector<Double> solve(IVector<Double> b) {
        // Convert vector to matrix, solve, then convert back
        IMatrix<Double> bMatrix = Linalg.matrix(new double[][]{b.toDoubleArray()}).transpose();
        IMatrix<Double> xMatrix = solve(bMatrix);
        double[] xArray = new double[xMatrix.rows()];
        for (int i = 0; i < xArray.length; i++) {
            xArray[i] = xMatrix.get(i, 0);
        }
        return Linalg.vector(xArray);
    }
    
    @Override
    public boolean isNonSingular() {
        // Check if all singular values are non-zero
        for (int i = 0; i < singularValues.length(); i++) {
            if (Math.abs(singularValues.get(i)) < epsilon) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public IMatrix<Double> getInverse() {
        if (!isNonSingular()) {
            throw new RuntimeException("Matrix is singular: smallest singular value is " +
                singularValues.get(singularValues.length() - 1));
        }
        return pseudoinverse();
    }

    /**
     * Moore–Penrose pseudoinverse A⁺ for A = UΣVᵀ with <b>thin</b> U (m×k), k = min(m,n),
     * Σ as length-k vector, Vᵀ full n×n as in {@link com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2}.
     * <p>A⁺ = Vₖ Σ⁺ Uᵀ with Vₖ the first k columns of V (columns of V are rows of Vᵀ transposed).</p>
     */
    private IMatrix<Double> pseudoinverse() {
        int m = uMatrix.rows();
        int k = uMatrix.cols();
        int n = vtMatrix.cols();
        if (singularValues.length() != k) {
            throw new IllegalStateException(
                    "SVD shape mismatch: U is " + m + "×" + k + " but S has length " + singularValues.length());
        }
        IMatrix<Double> ut = uMatrix.transpose();
        IMatrix<Double> sInvK = Linalg.zeros(k, k);
        for (int i = 0; i < k; i++) {
            double sv = singularValues.get(i);
            if (Math.abs(sv) > epsilon) {
                sInvK.put(i, i, 1.0 / sv);
            }
        }
        IMatrix<Double> sInvUt = sInvK.mmul(ut);
        IMatrix<Double> vk = Linalg.zeros(n, k);
        for (int j = 0; j < k; j++) {
            for (int r = 0; r < n; r++) {
                vk.put(r, j, vtMatrix.get(j, r));
            }
        }
        return vk.mmul(sInvUt);
    }
    
    /**
     * Get the U matrix.
     * 
     * @return the U matrix
     */
    public IMatrix<Double> getU() {
        return uMatrix;
    }
    
    /**
     * Get the singular values.
     * 
     * @return the singular values
     */
    public IVector<Double> getSingularValues() {
        return singularValues;
    }
    
    /**
     * Get the V^T matrix.
     * 
     * @return the V^T matrix
     */
    public IMatrix<Double> getVT() {
        return vtMatrix;
    }
}