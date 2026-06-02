package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.decomposition.impl.RereBidiagonalDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereBunchKaufmanDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereCholeskyDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereHessenbergDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereLUDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereQRDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereQrcpDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereQrcpDgeqp3Decomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereQrcpDlaqpsDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2;
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

    // ========== Bunch–Kaufman (symmetric indefinite) ==========

    /**
     * 对称不定 Bunch–Kaufman L·D·Lᵀ（下三角，对齐 LAPACK DSYTF2/DSYTRS 之 L 路径）。
     */
    public static IBunchKaufmanDecomposition createBunchKaufman() {
        return new RereBunchKaufmanDecomposition();
    }

    /**
     * @param relativeSymmetryTolerance 非对称判据（同 Cholesky 风格相对阈值）
     * @param singularityEps 行列式块判零阈值
     */
    public static IBunchKaufmanDecomposition createBunchKaufman(double relativeSymmetryTolerance, double singularityEps) {
        return new RereBunchKaufmanDecomposition(relativeSymmetryTolerance, singularityEps);
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

    // ========== QR 列主元（QRCP / RRQR） ==========

    /**
     * 列主元 QR：{@code A·P = Q·R}，适合秩亏损或接近秩亏损的最小二乘数值稳定路径。
     */
    public static IQrcpDecomposition createQrcp() {
        return new RereQrcpDecomposition();
    }

    public static IQrcpDecomposition createQrcp(double threshold) {
        return new RereQrcpDecomposition(threshold);
    }

    public static IQrcpDecomposition createQrcp(double epsilon, int maxIterations) {
        return new RereQrcpDecomposition(epsilon, maxIterations);
    }

    /**
     * 列主元 QR（DGEQP3 式列范数递推选主元 + 相同 Householder），大矩形阵时选主元阶段渐近快于 {@link #createQrcp()}。
     */
    public static IQrcpDecomposition createQrcpDgeqp3() {
        return new RereQrcpDgeqp3Decomposition();
    }

    public static IQrcpDecomposition createQrcpDgeqp3(double threshold) {
        return new RereQrcpDgeqp3Decomposition(threshold);
    }

    public static IQrcpDecomposition createQrcpDgeqp3(double epsilon, int maxIterations) {
        return new RereQrcpDgeqp3Decomposition(epsilon, maxIterations);
    }

    public static IQrcpDecomposition createQrcpDgeqp3(double threshold, double normRecurrenceEpsilon) {
        return new RereQrcpDgeqp3Decomposition(threshold, normRecurrenceEpsilon);
    }

    public static IQrcpDecomposition createQrcpDgeqp3(double epsilon, int maxIterations, double normRecurrenceEpsilon) {
        return new RereQrcpDgeqp3Decomposition(epsilon, maxIterations, normRecurrenceEpsilon);
    }

    /**
     * 列主元 QR（DGEQP3 递推 + DLAQPS 风格的尾随列范数分条更新），适合大 {@code n} 时略优的缓存遍历顺序；
     * 与 {@link #createQrcpDgeqp3()} 数值一致。
     */
    public static IQrcpDecomposition createQrcpDlaqps() {
        return new RereQrcpDlaqpsDecomposition();
    }

    public static IQrcpDecomposition createQrcpDlaqps(int columnTileSize) {
        return new RereQrcpDlaqpsDecomposition(columnTileSize);
    }

    public static IQrcpDecomposition createQrcpDlaqps(double threshold, int columnTileSize) {
        return new RereQrcpDlaqpsDecomposition(threshold, columnTileSize);
    }

    public static IQrcpDecomposition createQrcpDlaqps(double epsilon, int maxIterations, int columnTileSize) {
        return new RereQrcpDlaqpsDecomposition(epsilon, maxIterations, columnTileSize);
    }

    public static IQrcpDecomposition createQrcpDlaqps(double threshold, double normRecurrenceEpsilon, int columnTileSize) {
        return new RereQrcpDlaqpsDecomposition(threshold, normRecurrenceEpsilon, columnTileSize);
    }

    public static IQrcpDecomposition createQrcpDlaqps(double epsilon, int maxIterations, double normRecurrenceEpsilon, int columnTileSize) {
        return new RereQrcpDlaqpsDecomposition(epsilon, maxIterations, normRecurrenceEpsilon, columnTileSize);
    }

    // ========== SVD Decomposition ==========
    
    /**
     * Create an SVD decomposition with default parameters.
     * 
     * @return an SVD decomposition instance
     */
    public static ISVDDecomposition createSVD() {
        return new RereSVDDecompBlas2();
    }
    
    /**
     * Create an SVD decomposition with configurable parameters.
     * 
     * @param epsilon threshold for considering an element as zero
     * @param maxIterations maximum number of iterations
     * @return an SVD decomposition instance
     */
    public static ISVDDecomposition createSVD(double epsilon, int maxIterations) {
        return new RereSVDDecompBlas2(epsilon, maxIterations);
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