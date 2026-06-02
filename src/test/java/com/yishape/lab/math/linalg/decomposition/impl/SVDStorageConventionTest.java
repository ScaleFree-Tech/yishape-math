package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.util.RerePrecision;
import com.yishape.lab.util.Tuple3;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

/**
 * Test to verify singular vector storage convention and identify storage issues.
 * According to standard SVD: A = U * S * V^T
 * - U contains left singular vectors as COLUMNS
 * - V^T contains right singular vectors as ROWS
 * - V contains right singular vectors as COLUMNS
 */
public class SVDStorageConventionTest {
    
    private RereSVDDecompBlas2 svd;
    
    @BeforeEach
    public void setUp() {
        svd = new RereSVDDecompBlas2();
    }
    
    @Test
    public void testDiagonalMatrixStorageConvention() {
        System.out.println("=== Testing Singular Vector Storage Convention ===");
        
        // Test Case 1: Simple diagonal matrix
        // Expected: singular vectors should be standard basis vectors (or permutations)
        double[][] data = {
            {3.0, 0.0},
            {0.0, 1.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        System.out.println("Input diagonal matrix:");
        printMatrix(matrix);
        
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
        
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        IMatrix<Double> V = VT.transposeNew();
        
        System.out.println("\nU matrix (left singular vectors as columns):");
        printMatrix(U);
        System.out.println("\nSingular values:");
        for (int i = 0; i < S.length(); i++) {
            System.out.printf("σ[%d] = %.6f\n", i, S.get(i));
        }
        System.out.println("\nV matrix (right singular vectors as columns):");
        printMatrix(V);
        System.out.println("\nV^T matrix (right singular vectors as rows):");
        printMatrix(VT);
        
        // Verify storage convention by checking if we can reconstruct correctly
        verifyReconstruction(matrix, U, S, VT);
        
        // Check expected behavior for diagonal matrix
        analyzeExpectedBehavior(matrix, U, S, V);
    }
    
    @Test
    public void testPermutedDiagonalMatrix() {
        System.out.println("\n=== Testing Permuted Diagonal Matrix ===");
        
        // Test Case 2: Permuted diagonal matrix
        double[][] data = {
            {0.0, 2.0},
            {1.0, 0.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        System.out.println("Input permuted diagonal matrix:");
        printMatrix(matrix);
        
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
        
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        IMatrix<Double> V = VT.transposeNew();
        
        System.out.println("\nU matrix:");
        printMatrix(U);
        System.out.println("\nSingular values:");
        for (int i = 0; i < S.length(); i++) {
            System.out.printf("σ[%d] = %.6f\n", i, S.get(i));
        }
        System.out.println("\nV matrix:");
        printMatrix(V);
        
        verifyReconstruction(matrix, U, S, VT);
        analyzeExpectedBehavior(matrix, U, S, V);
    }
    
    @Test
    public void testRankOneMatrix() {
        System.out.println("\n=== Testing Rank-1 Matrix ===");
        
        // Test Case 3: Rank-1 matrix u * v^T
        double[][] data = {
            {2.0, 4.0},
            {1.0, 2.0},
            {3.0, 6.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        System.out.println("Input rank-1 matrix:");
        printMatrix(matrix);
        
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
        
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        
        System.out.println("\nU matrix:");
        printMatrix(U);
        System.out.println("\nSingular values:");
        for (int i = 0; i < S.length(); i++) {
            System.out.printf("σ[%d] = %.6f\n", i, S.get(i));
        }
        System.out.println("\nV^T matrix:");
        printMatrix(VT);
        
        verifyReconstruction(matrix, U, S, VT);
        
        // For rank-1 matrix, only first singular value should be non-zero
        int nonZeroCount = 0;
        for (int i = 0; i < S.length(); i++) {
            if (!RerePrecision.equalsZero(S.get(i), 1e-10)) {
                nonZeroCount++;
            }
        }
        System.out.printf("\nRank verification: Found %d non-zero singular values (expected: 1)\n", nonZeroCount);
    }
    
    private void verifyReconstruction(IMatrix<Double> original, IMatrix<Double> U, 
                                    IVector<Double> S, IMatrix<Double> VT) {
        // Reconstruct: A = U * S * VT
        IMatrix<Double> SMatrix = Linalg.zeros(S.length(), S.length());
        for (int i = 0; i < S.length(); i++) {
            SMatrix.put(i, i, S.get(i));
        }
        
        IMatrix<Double> reconstructed = U.mmul(SMatrix).mmul(VT);
        
        // Calculate reconstruction error
        double maxError = 0.0;
        for (int i = 0; i < original.rows(); i++) {
            for (int j = 0; j < original.cols(); j++) {
                double error = Math.abs(original.get(i, j) - reconstructed.get(i, j));
                maxError = Math.max(maxError, error);
            }
        }
        
        System.out.printf("\nReconstruction verification: Max error = %.2e", maxError);
        if (maxError < 1e-10) {
            System.out.println(" ✓ PASS");
        } else {
            System.out.println(" ✗ FAIL");
        }
    }
    
    private void analyzeExpectedBehavior(IMatrix<Double> original, IMatrix<Double> U, 
                                       IVector<Double> S, IMatrix<Double> V) {
        System.out.println("\n--- Expected Behavior Analysis ---");
        
        // For diagonal matrices, we expect simple relationships
        if (isDiagonalOrPermutedDiagonal(original)) {
            System.out.println("Matrix is diagonal or permuted diagonal.");
            System.out.println("Expected: U and V should be orthogonal matrices with simple structure");
            
            // Check if U columns are approximately unit vectors
            boolean uColumnsNormalized = true;
            for (int j = 0; j < U.cols(); j++) {
                double norm = 0.0;
                for (int i = 0; i < U.rows(); i++) {
                    norm += U.get(i, j) * U.get(i, j);
                }
                norm = Math.sqrt(norm);
                if (!RerePrecision.equals(norm, 1.0, 1e-10)) {
                    uColumnsNormalized = false;
                    System.out.printf("U column %d norm: %.6f (expected: 1.0)\n", j, norm);
                }
            }
            
            // Check if V columns are approximately unit vectors
            boolean vColumnsNormalized = true;
            for (int j = 0; j < V.cols(); j++) {
                double norm = 0.0;
                for (int i = 0; i < V.rows(); i++) {
                    norm += V.get(i, j) * V.get(i, j);
                }
                norm = Math.sqrt(norm);
                if (!RerePrecision.equals(norm, 1.0, 1e-10)) {
                    vColumnsNormalized = false;
                    System.out.printf("V column %d norm: %.6f (expected: 1.0)\n", j, norm);
                }
            }
            
            System.out.println("U columns normalized: " + (uColumnsNormalized ? "✓" : "✗"));
            System.out.println("V columns normalized: " + (vColumnsNormalized ? "✓" : "✗"));
        }
    }
    
    private boolean isDiagonalOrPermutedDiagonal(IMatrix<Double> matrix) {
        int rows = matrix.rows();
        int cols = matrix.cols();
        
        // Check if it's a permutation of a diagonal matrix
        int nonZeroCount = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (!RerePrecision.equalsZero(matrix.get(i, j), 1e-12)) {
                    nonZeroCount++;
                }
            }
        }
        
        // For a permuted diagonal matrix, non-zero count should equal min(rows, cols)
        return nonZeroCount <= Math.min(rows, cols);
    }
    
    private void printMatrix(IMatrix<Double> matrix) {
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                System.out.printf("%8.4f ", matrix.get(i, j));
            }
            System.out.println();
        }
    }
}