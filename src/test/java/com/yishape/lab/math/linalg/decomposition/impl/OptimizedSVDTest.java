package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.util.Tuple3;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

/**
 * Test to analyze and validate the optimizedSVD method implementation.
 * This test focuses on large matrices that trigger the optimized algorithm path.
 */
public class OptimizedSVDTest {
    
    private RereSVDDecompBlas2 svd;
    
    @BeforeEach
    public void setUp() {
        svd = new RereSVDDecompBlas2();
    }
    
    @Test
    public void testLargeMatrixOptimizedSVD() {
        System.out.println("=== Testing Optimized SVD for Large Matrix ===");
        
        // Create a matrix with > 10000 elements to trigger optimized SVD
        int rows = 120;
        int cols = 90;  // 120 * 90 = 10800 > 10000
        
        System.out.printf("Creating %dx%d matrix (%d elements) to trigger optimizedSVD\n", 
                         rows, cols, rows * cols);
        
        // Create a well-conditioned test matrix
        IMatrix<Double> matrix = createTestMatrix(rows, cols);
        
        System.out.println("Matrix created. Running SVD decomposition...");
        
        long startTime = System.currentTimeMillis();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
        long endTime = System.currentTimeMillis();
        
        System.out.printf("SVD completed in %d ms\n", endTime - startTime);
        
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        
        System.out.printf("Results: U(%dx%d), S(%d), VT(%dx%d)\n", 
                         U.rows(), U.cols(), S.length(), VT.rows(), VT.cols());
        
        // Verify dimensions
        verifyDimensions(matrix, U, S, VT);
        
        // Verify orthogonality (sample checks for large matrices)
        verifyOrthogonality(U, VT);
        
        // Verify reconstruction (sample elements)
        verifyReconstruction(matrix, U, S, VT);
        
        // Verify singular values are sorted
        verifySingularValuesSorted(S);
        
        // Check for numerical stability issues
        checkNumericalStability(S);
    }
    
    @Test
    public void testMediumMatrixBidiagonalSVD() {
        System.out.println("\n=== Testing Bidiagonal SVD for Medium Matrix ===");
        
        // Create a matrix with < 10000 elements to trigger bidiagonal SVD
        int rows = 80;
        int cols = 60;  // 80 * 60 = 4800 < 10000
        
        System.out.printf("Creating %dx%d matrix (%d elements) to trigger bidiagonalSVD\n", 
                         rows, cols, rows * cols);
        
        IMatrix<Double> matrix = createTestMatrix(rows, cols);
        
        long startTime = System.currentTimeMillis();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
        long endTime = System.currentTimeMillis();
        
        System.out.printf("SVD completed in %d ms\n", endTime - startTime);
        
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        
        // Same verification as optimized SVD
        verifyDimensions(matrix, U, S, VT);
        verifyOrthogonality(U, VT);
        verifyReconstruction(matrix, U, S, VT);
        verifySingularValuesSorted(S);
        checkNumericalStability(S);
    }
    
    @Test
    public void testRankDeficientLargeMatrix() {
        System.out.println("\n=== Testing Rank-Deficient Large Matrix ===");
        
        int rows = 110;
        int cols = 100;  // 110 * 100 = 11000 > 10000
        
        // Create a rank-deficient matrix
        IMatrix<Double> matrix = createRankDeficientMatrix(rows, cols, 20); // rank = 20
        
        System.out.printf("Created rank-deficient %dx%d matrix (rank ≈ 20)\n", rows, cols);
        
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
        
        IVector<Double> S = result.getSecond();
        
        // Count non-zero singular values
        int effectiveRank = 0;
        for (int i = 0; i < S.length(); i++) {
            if (S.get(i) > 1e-10) {
                effectiveRank++;
            }
        }
        
        System.out.printf("Detected rank: %d (expected: ≈20)\n", effectiveRank);
        
        // Print top 10 singular values
        System.out.println("Top 10 singular values:");
        for (int i = 0; i < Math.min(10, S.length()); i++) {
            System.out.printf("σ[%d] = %.6e\n", i, S.get(i));
        }
    }
    
    private IMatrix<Double> createTestMatrix(int rows, int cols) {
        // Create a well-conditioned test matrix with known structure
        double[][] data = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Use a combination of trigonometric functions for interesting structure
                data[i][j] = Math.sin(0.1 * i + 0.2 * j) + 0.5 * Math.cos(0.15 * i - 0.1 * j);
            }
        }
        
        return Linalg.matrix(data);
    }
    
    private IMatrix<Double> createRankDeficientMatrix(int rows, int cols, int targetRank) {
        // Create a rank-deficient matrix by constructing it as A = U * S * V^T
        // where S has only 'targetRank' non-zero elements
        
        double[][] uData = new double[rows][targetRank];
        double[][] vData = new double[cols][targetRank];
        
        // Create random U and V matrices
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < targetRank; j++) {
                uData[i][j] = Math.random() - 0.5;
            }
        }
        
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < targetRank; j++) {
                vData[i][j] = Math.random() - 0.5;
            }
        }
        
        IMatrix<Double> U = Linalg.matrix(uData);
        IMatrix<Double> V = Linalg.matrix(vData);
        
        // Create diagonal singular values matrix
        double[] singularValues = new double[targetRank];
        for (int i = 0; i < targetRank; i++) {
            singularValues[i] = 10.0 / (i + 1); // Decreasing singular values
        }
        
        IMatrix<Double> S = Linalg.zeros(targetRank, targetRank);
        for (int i = 0; i < targetRank; i++) {
            S.put(i, i, singularValues[i]);
        }
        
        // Construct the matrix: A = U * S * V^T
        return U.mmul(S).mmul(V.transposeNew());
    }
    
    private void verifyDimensions(IMatrix<Double> original, IMatrix<Double> U, 
                                 IVector<Double> S, IMatrix<Double> VT) {
        int m = original.rows();
        int n = original.cols();
        int minDim = Math.min(m, n);
        
        boolean dimensionsCorrect = true;
        
        if (U.rows() != m || U.cols() != minDim) {
            System.err.printf("❌ U dimensions incorrect: expected %dx%d, got %dx%d\n", 
                             m, minDim, U.rows(), U.cols());
            dimensionsCorrect = false;
        }
        
        if (S.length() != minDim) {
            System.err.printf("❌ S length incorrect: expected %d, got %d\n", 
                             minDim, S.length());
            dimensionsCorrect = false;
        }
        
        if (VT.rows() != minDim || VT.cols() != n) {
            System.err.printf("❌ VT dimensions incorrect: expected %dx%d, got %dx%d\n", 
                             minDim, n, VT.rows(), VT.cols());
            dimensionsCorrect = false;
        }
        
        if (dimensionsCorrect) {
            System.out.println("✓ All matrix dimensions are correct");
        }
    }
    
    private void verifyOrthogonality(IMatrix<Double> U, IMatrix<Double> VT) {
        // Check U^T * U ≈ I (sample check for large matrices)
        double maxError = 0.0;
        int checkSize = Math.min(5, U.cols()); // Check first 5x5 block
        
        IMatrix<Double> UTU = U.transposeNew().mmul(U);
        
        for (int i = 0; i < checkSize; i++) {
            for (int j = 0; j < checkSize; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double actual = UTU.get(i, j);
                double error = Math.abs(actual - expected);
                maxError = Math.max(maxError, error);
            }
        }
        
        System.out.printf("Orthogonality check (U^T*U): max error in %dx%d block = %.2e", 
                         checkSize, checkSize, maxError);
        if (maxError < 1e-10) {
            System.out.println(" ✓");
        } else {
            System.out.println(" ❌");
        }
        
        // Similar check for VT * VT^T ≈ I
        maxError = 0.0;
        checkSize = Math.min(5, VT.rows());
        
        IMatrix<Double> VTSVT = VT.mmul(VT.transposeNew());
        
        for (int i = 0; i < checkSize; i++) {
            for (int j = 0; j < checkSize; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double actual = VTSVT.get(i, j);
                double error = Math.abs(actual - expected);
                maxError = Math.max(maxError, error);
            }
        }
        
        System.out.printf("Orthogonality check (VT*VT^T): max error in %dx%d block = %.2e", 
                         checkSize, checkSize, maxError);
        if (maxError < 1e-10) {
            System.out.println(" ✓");
        } else {
            System.out.println(" ❌");
        }
    }
    
    private void verifyReconstruction(IMatrix<Double> original, IMatrix<Double> U, 
                                    IVector<Double> S, IMatrix<Double> VT) {
        // Sample-based reconstruction check for large matrices
        int checkRows = Math.min(5, original.rows());
        int checkCols = Math.min(5, original.cols());
        
        // Create S matrix
        IMatrix<Double> SMatrix = Linalg.zeros(S.length(), S.length());
        for (int i = 0; i < S.length(); i++) {
            SMatrix.put(i, i, S.get(i));
        }
        
        IMatrix<Double> reconstructed = U.mmul(SMatrix).mmul(VT);
        
        double maxError = 0.0;
        for (int i = 0; i < checkRows; i++) {
            for (int j = 0; j < checkCols; j++) {
                double error = Math.abs(original.get(i, j) - reconstructed.get(i, j));
                maxError = Math.max(maxError, error);
            }
        }
        
        System.out.printf("Reconstruction check (%dx%d sample): max error = %.2e", 
                         checkRows, checkCols, maxError);
        if (maxError < 1e-10) {
            System.out.println(" ✓");
        } else {
            System.out.println(" ❌");
        }
    }
    
    private void verifySingularValuesSorted(IVector<Double> S) {
        boolean sorted = true;
        for (int i = 0; i < S.length() - 1; i++) {
            if (S.get(i) < S.get(i + 1)) {
                sorted = false;
                break;
            }
        }
        
        System.out.println("Singular values sorted (descending): " + (sorted ? "✓" : "❌"));
    }
    
    private void checkNumericalStability(IVector<Double> S) {
        double maxSingularValue = S.get(0);
        double minNonZeroSingularValue = maxSingularValue;
        
        for (int i = 0; i < S.length(); i++) {
            if (S.get(i) > 1e-15) {
                minNonZeroSingularValue = Math.min(minNonZeroSingularValue, S.get(i));
            }
        }
        
        double conditionNumber = maxSingularValue / minNonZeroSingularValue;
        
        System.out.printf("Numerical stability: condition number = %.2e", conditionNumber);
        if (conditionNumber < 1e12) {
            System.out.println(" ✓");
        } else {
            System.out.println(" ⚠️ (high condition number)");
        }
        
        System.out.printf("Largest singular value: %.6e\n", maxSingularValue);
        System.out.printf("Smallest non-zero singular value: %.6e\n", minNonZeroSingularValue);
    }
}