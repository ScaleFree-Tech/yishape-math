package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.ISchurDecomposition;
import com.yishape.lab.math.linalg.decomposition.IMatrixDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.SchurDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.NonSquareMatrixException;
import com.yishape.lab.math.linalg.decomposition.DecompositionFailedException;
import com.yishape.lab.util.Tuple2;

/**
 * Implementation of Schur decomposition using the Francis QR algorithm with double shift.
 * <p>
 * This implementation computes the real Schur decomposition of a square matrix,
 * which decomposes the matrix A as: A = U T U^T where U is orthogonal and T
 * is quasi-upper triangular (block upper triangular with 1x1 and 2x2 blocks).
 * </p>
 * 
 * <p>
 * The algorithm uses the Francis QR step with double shift for better convergence:
 * <ol>
 *   <li>Reduce matrix to upper Hessenberg form</li>
 *   <li>Apply Francis QR steps with double shift until convergence</li>
 *   <li>Accumulate orthogonal transformations to form U</li>
 * </ol>
 * </p>
 * 
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 *   <li>Hessenberg, K. (1940). Behandlung linearer Eigenwertaufgaben mit Hilfe der Hamilton-Cayleyschen Gleichung.</li>
 *   <li>Francis, J. G. F. (1961). The QR Transformation A Unitary Analogue to the LR Transformation—Part 1.</li>
 *   <li>Watkins, D. S. (2007). The matrix eigenvalue problem: GR and Krylov subspace methods. SIAM.</li>
 * </ul>
 * 
 * @since 2.0
 */
public class RereSchurDecomposition implements ISchurDecomposition {
    
    /** Cached value of U. */
    private IMatrix<Double> cachedU;
    /** Cached value of T. */
    private IMatrix<Double> cachedT;
    /** Cached value of UT. */
    private IMatrix<Double> cachedUT;
    /** The matrix data in Hessenberg form. */
    private double[][] hData;
    /** The orthogonal transformation matrix. */
    private double[][] uData;
    /** Epsilon for numerical comparisons. */
    private double epsilon;
    /** Maximum number of iterations. */
    private static final int MAX_ITERATIONS = 1000;
    
    /**
     * Default constructor with default epsilon.
     */
    public RereSchurDecomposition() {
        this.epsilon = DEFAULT_EPSILON;
    }
    
    /**
     * Constructor with configurable epsilon.
     * 
     * @param epsilon threshold for considering an element as zero
     */
    public RereSchurDecomposition(double epsilon) {
        this.epsilon = epsilon;
    }
    
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix) {
        return decompose(matrix, epsilon);
    }
    
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon) {
        // Reset cached values
        cachedU = null;
        cachedT = null;
        cachedUT = null;
        this.epsilon = epsilon;
        
        // Get matrix dimensions
        int rows = matrix.rows();
        int cols = matrix.cols();
        
        if (rows != cols) {
            throw new NonSquareMatrixException(
                "Only square matrices can be decomposed using Schur decomposition",
                "Schur Decomposition", 
                "Matrix " + rows + "x" + cols,
                rows, cols);
        }
        
        int n = rows;
        
        // Copy data
        hData = matrix.toDoubleArray();
        uData = new double[n][n];
        
        // Initialize U as identity matrix
        for (int i = 0; i < n; i++) {
            uData[i][i] = 1.0;
        }
        
        // Reduce to upper Hessenberg form
        reduceToHessenberg();
        
        // Apply Francis QR algorithm with double shift
        francisQRStep();
        
        // Create result matrices
        cachedT = Linalg.matrix(hData);
        cachedU = Linalg.matrix(uData);
        cachedUT = cachedU.transpose();
        
        return new Tuple2<>(cachedU, cachedT);
    }
    
    /**
     * Reduce matrix to upper Hessenberg form using Householder reflections.
     */
    private void reduceToHessenberg() {
        int n = hData.length;
        
        for (int k = 0; k < n - 2; k++) {
            // Find the Householder reflection to zero out elements below the subdiagonal
            double[] h = new double[n - k - 1];
            double norm = 0.0;
            
            for (int i = k + 1; i < n; i++) {
                h[i - k - 1] = hData[i][k];
                norm += h[i - k - 1] * h[i - k - 1];
            }
            
            if (norm > epsilon) {
                norm = Math.sqrt(norm);
                if (h[0] != 0) {
                    norm *= Math.signum(h[0]);
                }
                
                // Normalize the vector
                h[0] += norm;
                double hNorm = 0.0;
                for (int i = 0; i < h.length; i++) {
                    hNorm += h[i] * h[i];
                }
                hNorm = Math.sqrt(hNorm);
                
                if (hNorm > epsilon) {
                    for (int i = 0; i < h.length; i++) {
                        h[i] /= hNorm;
                    }
                    
                    // Apply the Householder reflection from the right
                    for (int j = k + 1; j < n; j++) {
                        double dotProduct = 0.0;
                        for (int i = k + 1; i < n; i++) {
                            dotProduct += hData[j][i] * h[i - k - 1];
                        }
                        for (int i = k + 1; i < n; i++) {
                            hData[j][i] -= 2.0 * dotProduct * h[i - k - 1];
                        }
                    }
                    
                    // Apply the Householder reflection from the left
                    for (int i = 0; i < n; i++) {
                        double dotProduct = 0.0;
                        for (int j = k + 1; j < n; j++) {
                            dotProduct += hData[j][i] * h[j - k - 1];
                        }
                        for (int j = k + 1; j < n; j++) {
                            hData[j][i] -= 2.0 * dotProduct * h[j - k - 1];
                        }
                    }
                    
                    // Accumulate the transformation in U
                    for (int i = 0; i < n; i++) {
                        double dotProduct = 0.0;
                        for (int j = k + 1; j < n; j++) {
                            dotProduct += uData[i][j] * h[j - k - 1];
                        }
                        for (int j = k + 1; j < n; j++) {
                            uData[i][j] -= 2.0 * dotProduct * h[j - k - 1];
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Apply Francis QR step with double shift until convergence.
     */
    private void francisQRStep() {
        int n = hData.length;
        int iterations = 0;
        
        while (iterations < MAX_ITERATIONS) {
            // Check for convergence
            boolean converged = true;
            for (int i = 1; i < n; i++) {
                if (Math.abs(hData[i][i-1]) > epsilon) {
                    converged = false;
                    break;
                }
            }
            
            if (converged) {
                break;
            }
            
            // Determine shift
            double s, t;
            if (n >= 2) {
                // Use double shift from bottom 2x2 block
                double tr = hData[n-2][n-2] + hData[n-1][n-1];
                double det = hData[n-2][n-2] * hData[n-1][n-1] - hData[n-2][n-1] * hData[n-1][n-2];
                // Compute eigenvalues of the 2x2 matrix
                double discriminant = tr * tr - 4 * det;
                if (discriminant >= 0) {
                    // Real eigenvalues
                    double sqrtDisc = Math.sqrt(discriminant);
                    s = (tr + sqrtDisc) / 2.0;
                    t = (tr - sqrtDisc) / 2.0;
                } else {
                    // Complex eigenvalues, use real part
                    s = tr / 2.0;
                    t = s;
                }
            } else {
                // Use single shift
                s = hData[n-1][n-1];
                t = s;
            }
            
            // Apply QR step with shift
            qrStepWithShift(s, t);
            
            iterations++;
        }
        
        if (iterations >= MAX_ITERATIONS) {
            throw new DecompositionFailedException(
                "Schur decomposition failed to converge after " + MAX_ITERATIONS + " iterations",
                "Schur Decomposition",
                "Matrix " + n + "x" + n,
                iterations,
                epsilon);
        }
    }
    
    /**
     * Apply a QR step with shift to the Hessenberg matrix.
     * 
     * @param s first shift value
     * @param t second shift value
     */
    private void qrStepWithShift(double s, double t) {
        int n = hData.length;
        
        // Compute the first column of (H - s*I)(H - t*I)
        double x = hData[0][0] * hData[0][0] + hData[0][1] * hData[1][0] 
                  - (s + t) * hData[0][0] + s * t;
        double y = hData[1][0] * (hData[0][0] + hData[1][1] - s - t);
        double z = hData[1][0] * hData[2][1];
        
        for (int k = 0; k < n - 1; k++) {
            // Compute Givens rotation
            double r = Math.sqrt(x * x + y * y);
            if (r < epsilon) {
                x = hData[k+1][k];
                y = hData[k+2][k];
                if (k < n - 2) {
                    z = hData[k+3][k];
                } else {
                    z = 0.0;
                }
                continue;
            }
            
            double c = x / r;
            double sd = y / r;
            
            // Apply Givens rotation from the left
            for (int j = k; j < n; j++) {
                double tmp = c * hData[k][j] + sd * hData[k+1][j];
                hData[k+1][j] = -sd * hData[k][j] + c * hData[k+1][j];
                hData[k][j] = tmp;
            }
            
            // Apply Givens rotation from the right
            for (int i = 0; i <= Math.min(k+2, n-1); i++) {
                double tmp = c * hData[i][k] + sd * hData[i][k+1];
                hData[i][k+1] = -sd * hData[i][k] + c * hData[i][k+1];
                hData[i][k] = tmp;
            }
            
            // Accumulate the transformation in U
            for (int i = 0; i < n; i++) {
                double tmp = c * uData[i][k] + sd * uData[i][k+1];
                uData[i][k+1] = -sd * uData[i][k] + c * uData[i][k+1];
                uData[i][k] = tmp;
            }
            
            // Update x, y, z for next iteration
            x = hData[k+1][k];
            if (k < n - 2) {
                y = hData[k+2][k];
            } else {
                y = 0.0;
            }
            if (k < n - 3) {
                z = hData[k+3][k];
            } else {
                z = 0.0;
            }
        }
    }
    
    @Override
    public IMatrix<Double> getU() {
        if (cachedU == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        return cachedU;
    }
    
    @Override
    public IMatrix<Double> getT() {
        if (cachedT == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        return cachedT;
    }
    
    @Override
    public IMatrix<Double> getUT() {
        if (cachedUT == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        return cachedUT;
    }
    
    @Override
    public IDecompositionSolver getSolver() {
        if (hData == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        // Return the standalone solver
        return new SchurDecompositionSolver(Linalg.matrix(hData), Linalg.matrix(uData), epsilon);
    }
    
    @Override
    public double getEpsilon() {
        return epsilon;
    }

    /**
     * Perform Schur decomposition with configurable parameters.
     * 
     * @param matrix The matrix to decompose (must be square)
     * @param epsilon Threshold for considering an element as zero
     * @param maxIterations Maximum number of iterations for iterative algorithms
     * @return A tuple containing U (orthogonal matrix) and T (quasi-upper triangular matrix)
     * @throws NonSquareMatrixException if the matrix is not square
     * @throws DecompositionFailedException if the decomposition fails to converge
     */
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations) {
        // Save original maxIterations value
        // Note: This implementation doesn't use maxIterations directly, but we maintain the interface contract
        return decompose(matrix, epsilon);
    }

    /**
     * Calculate the determinant of the matrix.
     * 
     * @return determinant of the matrix
     * @throws IllegalStateException if decomposition has not been performed
     */
    @Override
    public double getDeterminant() {
        if (hData == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        
        // For Schur decomposition A = U T U^T, det(A) = det(T)
        // Since T is quasi-upper triangular, its determinant is the product of diagonal elements
        // For 2x2 blocks, we need to compute the determinant of each block
        double det = 1.0;
        int n = hData.length;
        
        for (int i = 0; i < n; i++) {
            if (i < n - 1 && Math.abs(hData[i+1][i]) > epsilon) {
                // 2x2 block
                double a = hData[i][i];
                double b = hData[i][i+1];
                double c = hData[i+1][i];
                double d = hData[i+1][i+1];
                det *= (a * d - b * c);
                i++; // Skip the next element as it's part of this block
            } else {
                // 1x1 block
                det *= hData[i][i];
            }
        }
        
        return det;
    }

    /**
     * Check if the decomposed matrix is non-singular.
     * 
     * @return true if the decomposed matrix is non-singular
     * @throws IllegalStateException if decomposition has not been performed
     */
    @Override
    public boolean isNonSingular() {
        return Math.abs(getDeterminant()) > epsilon;
    }

    /**
     * Get the condition number of the matrix.
     * 
     * @return condition number of the matrix
     * @throws IllegalStateException if decomposition has not been performed
     */
    @Override
    public double getConditionNumber() {
        if (hData == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        
        // Estimate the condition number from the Schur form
        // The condition number is the ratio of the largest to smallest singular values
        // For a triangular matrix, we can estimate from the diagonal elements
        double maxDiag = 0.0;
        double minDiag = Double.POSITIVE_INFINITY;
        
        for (int i = 0; i < hData.length; i++) {
            double diagAbs = Math.abs(hData[i][i]);
            maxDiag = Math.max(maxDiag, diagAbs);
            if (diagAbs > epsilon) {
                minDiag = Math.min(minDiag, diagAbs);
            }
        }
        
        if (minDiag == Double.POSITIVE_INFINITY || minDiag < epsilon) {
            return Double.POSITIVE_INFINITY; // Singular matrix
        }
        
        return maxDiag / minDiag;
    }

    /**
     * Get the rank of the matrix.
     * 
     * @return rank of the matrix
     * @throws IllegalStateException if decomposition has not been performed
     */
    @Override
    public int getRank() {
        if (hData == null) {
            throw new IllegalStateException("Decomposition not yet performed");
        }
        
        // Count non-zero diagonal elements in the Schur form
        int rank = 0;
        for (int i = 0; i < hData.length; i++) {
            if (Math.abs(hData[i][i]) > epsilon) {
                rank++;
            }
        }
        
        return rank;
    }

    /**
     * Get the maximum number of iterations allowed.
     * 
     * @return the maximum number of iterations
     */
    @Override
    public int getMaxIterations() {
        return IMatrixDecomposition.DEFAULT_MAX_ITERATIONS;
    }
}