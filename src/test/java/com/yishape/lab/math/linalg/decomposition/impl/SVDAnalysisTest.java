package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.util.Tuple3;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

public class SVDAnalysisTest {
    
    private RereSVDDecompBlas2 svd;
    
    @BeforeEach
    public void setUp() {
        svd = new RereSVDDecompBlas2();
    }
    
    @Test
    public void testSmallMatrix() {
        System.out.println("\n--- Test 1: Small Matrix (3x2) ---");
        double[][] data1 = {
            {3.0, 2.0},
            {2.0, 3.0},
            {2.0, 1.0}
        };
        testSVDImplementation(data1, "Traditional SVD");
    }
    
    @Test
    public void testMediumMatrix() {
        System.out.println("\n--- Test 2: Medium Matrix (5x4) ---");
        double[][] data2 = {
            {1.0, 2.0, 3.0, 4.0},
            {2.0, 4.0, 6.0, 8.0},
            {3.0, 6.0, 9.0, 12.0},
            {1.0, 1.0, 1.0, 1.0},
            {2.0, 1.0, 3.0, 2.0}
        };
        testSVDImplementation(data2, "Bidiagonal SVD");
    }
    
    @Test
    public void testSingularVectorStorage() {
        System.out.println("\n--- Test 3: Singular Vector Storage Verification ---");
        
        // Create a simple matrix to test singular vector storage
        double[][] data = {
            {1.0, 0.0},
            {0.0, 2.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        System.out.println("Test matrix (should have known singular vectors):");
        printMatrix(matrix);
        
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
        
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        IMatrix<Double> V = VT.transposeNew();
        
        System.out.println("\nExpected: U should be identity or permutation, V should be identity or permutation");
        System.out.println("U:");
        printMatrix(U);
        System.out.println("V:");
        printMatrix(V);
        System.out.println("Singular values:");
        for (int i = 0; i < S.length(); i++) {
            System.out.printf("σ[%d] = %.6f\n", i, S.get(i));
        }
    }
    
    @Test
    public void testMatrixReconstruction() {
        System.out.println("\n--- Test 4: Matrix Reconstruction Test ---");
        double[][] data = {
            {3.0, 2.0},
            {2.0, 3.0},
            {2.0, 1.0}
        };
        
        IMatrix<Double> original = Linalg.matrix(data);
        
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(original);
        
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        
        // Reconstruct: A = U * S * VT
        IMatrix<Double> SMatrix = Linalg.zeros(S.length(), S.length());
        for (int i = 0; i < S.length(); i++) {
            SMatrix.put(i, i, S.get(i));
        }
        
        IMatrix<Double> reconstructed = U.mmul(SMatrix).mmul(VT);
        
        System.out.println("Original matrix:");
        printMatrix(original);
        System.out.println("\nReconstructed matrix (U * S * V^T):");
        printMatrix(reconstructed);
        
        // Calculate reconstruction error
        double maxError = 0.0;
        for (int i = 0; i < original.rows(); i++) {
            for (int j = 0; j < original.cols(); j++) {
                double error = Math.abs(original.get(i, j) - reconstructed.get(i, j));
                maxError = Math.max(maxError, error);
            }
        }
        System.out.printf("\nMax reconstruction error: %.10f\n", maxError);
    }
    
    @Test
    public void testSVDProperties() {
        System.out.println("\n--- Test 5: SVD Property Verification ---");
        double[][] data = {
            {3.0, 2.0},
            {2.0, 3.0},
            {2.0, 1.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
        
        IMatrix<Double> U = result.getFirst();
        IMatrix<Double> VT = result.getThird();
        IMatrix<Double> V = VT.transposeNew();
        
        System.out.println("Testing orthogonality properties:");
        
        // Test U^T * U = I
        IMatrix<Double> UTU = U.transposeNew().mmul(U);
        System.out.println("\nU^T * U (should be close to identity):");
        printMatrix(UTU);
        
        // Test V^T * V = I (since we have V, we test V * V^T = I)
        IMatrix<Double> VVT = V.mmul(V.transposeNew());
        System.out.println("\nV * V^T (should be close to identity):");
        printMatrix(VVT);
        
        // Check if singular values are sorted in descending order
        IVector<Double> S = result.getSecond();
        boolean sorted = true;
        for (int i = 0; i < S.length() - 1; i++) {
            if (S.get(i) < S.get(i + 1)) {
                sorted = false;
                break;
            }
        }
        System.out.println("\nSingular values are sorted in descending order: " + sorted);
    }
    
    private void testSVDImplementation(double[][] data, String method) {
        try {
            IMatrix<Double> matrix = Linalg.matrix(data);
            System.out.println("Input matrix (" + matrix.rows() + "x" + matrix.cols() + "):");
            printMatrix(matrix);
            
            Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
            
            IMatrix<Double> U = result.getFirst();
            IVector<Double> S = result.getSecond();
            IMatrix<Double> VT = result.getThird();
            
            System.out.println("\nU matrix (" + U.rows() + "x" + U.cols() + "):");
            printMatrix(U);
            
            System.out.println("\nSingular values:");
            for (int i = 0; i < S.length(); i++) {
                System.out.printf("σ[%d] = %.6f\n", i, S.get(i));
            }
            
            System.out.println("\nV^T matrix (" + VT.rows() + "x" + VT.cols() + "):");
            printMatrix(VT);
            
            System.out.println("\nV matrix (transpose of V^T):");
            IMatrix<Double> V = VT.transposeNew();
            printMatrix(V);
            
        } catch (Exception e) {
            System.err.println("Error in " + method + ": " + e.getMessage());
            e.printStackTrace();
        }
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