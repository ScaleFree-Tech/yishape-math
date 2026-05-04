package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.decomposition.impl.RereBidiagonalDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereCholeskyDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereHessenbergDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereLUDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereQRDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereTridiagonalDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereSchurDecomposition;

/**
 * Unified factory for creating matrix decomposition instances.
 * <p>
 * This factory provides a consistent interface for creating all types of matrix
 * decompositions with configurable parameters for numerical stability and performance.
 * </p>
 *
 * <h3>Usage Examples</h3>
 * <pre>
 * {@code
 // Create a QR decomposition with default parameters
 IQRDecomposition qr = Decomps.createQR();

 // Create a Cholesky decomposition with custom epsilon
 ICholeskyDecomposition chol = Decomps.createCholesky(1e-15, 1000);

 // Create a Schur decomposition
 ISchurDecomposition schur = Decomps.createSchur();
 }
 * </pre>
 *
 * @author lteb2
 * @version 1.0
 * @since 2.0
 */
public class Decomps {
    
    /**
     * Private constructor to prevent instantiation.
     */
    private Decomps() {
        // Factory class should not be instantiated
    }
    
    // ========== Bidiagonal Decomposition ==========
    
    /**
     * Create a bidiagonal decomposition with default parameters.
     * 
     * @return a bidiagonal decomposition instance
     */
    public static IBidiagonalDecomposition createBidiagonal() {
        return new RereBidiagonalDecomposition();
    }
    
    /**
     * Create a bidiagonal decomposition with configurable parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     * @return a bidiagonal decomposition instance
     */
    public static IBidiagonalDecomposition createBidiagonal(double epsilon, int maxIterations) {
        // For now, we return the default implementation as it doesn't have configurable parameters
        return new RereBidiagonalDecomposition();
    }
    
    // ========== Cholesky Decomposition ==========
    
    /**
     * Create a Cholesky decomposition with default parameters.
     * 
     * @return a Cholesky decomposition instance
     */
    public static ICholeskyDecomposition createCholesky() {
        return new RereCholeskyDecomposition();
    }
    
    /**
     * Create a Cholesky decomposition with configurable thresholds.
     * 
     * @param relativeSymmetryThreshold threshold above which off-diagonal elements 
     *        are considered too different and matrix not symmetric
     * @param absolutePositivityThreshold threshold below which diagonal elements 
     *        are considered null and matrix not positive definite
     * @return a Cholesky decomposition instance
     */
    public static ICholeskyDecomposition createCholesky(
            double relativeSymmetryThreshold, double absolutePositivityThreshold) {
        return new RereCholeskyDecomposition(relativeSymmetryThreshold, absolutePositivityThreshold);
    }
    
    /**
     * Create a Cholesky decomposition with unified parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     * @return a Cholesky decomposition instance
     */
    public static ICholeskyDecomposition createCholesky(double epsilon, int maxIterations) {
        return new RereCholeskyDecomposition(epsilon, maxIterations);
    }
    
    // ========== Eigen Decomposition ==========
    
    /**
     * Create an eigen decomposition with default parameters.
     * 
     * @return an eigen decomposition instance
     */
    public static IEigenDecomposition createEigen() {
        return new RereEigenDecomposition();
    }
    
    /**
     * Create an eigen decomposition with configurable parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     * @return an eigen decomposition instance
     */
    public static IEigenDecomposition createEigen(double epsilon, int maxIterations) {
        return new RereEigenDecomposition(epsilon, maxIterations);
    }
    
    // ========== Hessenberg Decomposition ==========
    
    /**
     * Create a Hessenberg decomposition with default parameters.
     * 
     * @return a Hessenberg decomposition instance
     */
    public static IHessenbergDecomposition createHessenberg() {
        return new RereHessenbergDecomposition();
    }
    
    /**
     * Create a Hessenberg decomposition with configurable parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     * @return a Hessenberg decomposition instance
     */
    public static IHessenbergDecomposition createHessenberg(double epsilon, int maxIterations) {
        return new RereHessenbergDecomposition(epsilon, maxIterations);
    }
    
    // ========== LU Decomposition ==========
    
    /**
     * Create an LU decomposition with default parameters.
     * 
     * @return an LU decomposition instance
     */
    public static ILUDecomposition createLU() {
        return new RereLUDecomposition();
    }
    
    /**
     * Create an LU decomposition with configurable threshold.
     * 
     * @param singularityThreshold threshold for considering a matrix singular
     * @return an LU decomposition instance
     */
    public static ILUDecomposition createLU(double singularityThreshold) {
        return new RereLUDecomposition(singularityThreshold);
    }
    
    // ========== QR Decomposition ==========
    
    /**
     * Create a QR decomposition with default parameters.
     * 
     * @return a QR decomposition instance
     */
    public static IQRDecomposition createQR() {
        return new RereQRDecomposition();
    }
    
    /**
     * Create a QR decomposition with configurable threshold.
     * 
     * @param threshold singularity threshold
     * @return a QR decomposition instance
     */
    public static IQRDecomposition createQR(double threshold) {
        return new RereQRDecomposition(threshold);
    }
    
    /**
     * Create a QR decomposition with unified parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     * @return a QR decomposition instance
     */
    public static IQRDecomposition createQR(double epsilon, int maxIterations) {
        return new RereQRDecomposition(epsilon, maxIterations);
    }
    
    // ========== SVD Decomposition ==========
    
    /**
     * Create an SVD decomposition with default parameters.
     * 
     * @return an SVD decomposition instance
     */
    public static ISVDDecomposition createSVD() {
        return new RereSVDDecomposition();
    }
    
    /**
     * Create an SVD decomposition with configurable parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     * @return an SVD decomposition instance
     */
    public static ISVDDecomposition createSVD(double epsilon, int maxIterations) {
        return new RereSVDDecomposition(epsilon, maxIterations);
    }
    
    // ========== Tridiagonal Decomposition ==========
    
    /**
     * Create a tridiagonal decomposition with default parameters.
     * 
     * @return a tridiagonal decomposition instance
     */
    public static ITridiagonalDecomposition createTridiagonal() {
        return new RereTridiagonalDecomposition();
    }
    
    /**
     * Create a tridiagonal decomposition with configurable parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     * @return a tridiagonal decomposition instance
     */
    public static ITridiagonalDecomposition createTridiagonal(double epsilon, int maxIterations) {
        return new RereTridiagonalDecomposition(epsilon, maxIterations);
    }
    
    // ========== Schur Decomposition ==========
    
    /**
     * Create a Schur decomposition with default parameters.
     * 
     * @return a Schur decomposition instance
     */
    public static ISchurDecomposition createSchur() {
        return new RereSchurDecomposition();
    }
    
    /**
     * Create a Schur decomposition with configurable epsilon.
     * 
     * @param epsilon threshold for considering an element as zero
     * @return a Schur decomposition instance
     */
    public static ISchurDecomposition createSchur(double epsilon) {
        return new RereSchurDecomposition(epsilon);
    }
    
    /**
     * Create a Schur decomposition with unified parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     * @return a Schur decomposition instance
     */
    public static ISchurDecomposition createSchur(double epsilon, int maxIterations) {
        return new RereSchurDecomposition(epsilon);
    }
}