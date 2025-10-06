package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecomposition;
import com.yishape.lab.util.Tuple3;

import org.junit.jupiter.api.Test;

public class SVDDebugTest {
    
    @Test
    public void debugSimpleMatrix() {
        // Create a simple 2x2 matrix
        double[][] data = {
            {1.0, 2.0},
            {3.0, 4.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        System.out.println("Original matrix:");
        printMatrix(matrix);
        
        RereSVDDecomposition svd = new RereSVDDecomposition();
        
        // Perform decomposition
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        
        System.out.println("U matrix:");
        printMatrix(U);
        System.out.println("Singular values:");
        printVector(S);
        System.out.println("VT matrix:");
        printMatrix(VT);
        
        // Create diagonal matrix from singular values
        IMatrix<Double> S_matrix = Linalg.zeros(2, 2);
        for (int i = 0; i < 2; i++) {
            S_matrix.put(i, i, S.get(i));
        }
        System.out.println("S matrix:");
        printMatrix(S_matrix);
        
        // Reconstruct the original matrix
        IMatrix<Double> reconstructed = U.mmul(S_matrix).mmul(VT);
        System.out.println("Reconstructed matrix:");
        printMatrix(reconstructed);
        
        // Check reconstruction error
        double maxError = 0.0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double error = Math.abs(data[i][j] - reconstructed.get(i, j));
                maxError = Math.max(maxError, error);
                System.out.printf("Original[%d][%d] = %f, Reconstructed[%d][%d] = %f, Error = %f%n", 
                    i, j, data[i][j], i, j, reconstructed.get(i, j), error);
            }
        }
        System.out.println("Max reconstruction error: " + maxError);
        
        // Verify orthogonality
        IMatrix<Double> UTU = U.transpose().mmul(U);
        IMatrix<Double> VTV = VT.mmul(VT.transpose());
        System.out.println("UTU matrix:");
        printMatrix(UTU);
        System.out.println("VTV matrix:");
        printMatrix(VTV);
        
        // Just print the results, don't assert
        System.out.println("Debug completed for 2x2 matrix");
    }
    
    @Test
    public void debugNonSquareMatrix() {
        // Create a 4x3 matrix
        double[][] data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {10.0, 11.0, 12.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(data);
        System.out.println("Original matrix (4x3):");
        printMatrix(matrix);
        
        RereSVDDecomposition svd = new RereSVDDecomposition();
        
        // Perform decomposition
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(matrix);
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        
        System.out.println("U matrix (" + U.rows() + "x" + U.cols() + "):");
        printMatrix(U);
        System.out.println("Singular values (" + S.length() + "):");
        printVector(S);
        System.out.println("VT matrix (" + VT.rows() + "x" + VT.cols() + "):");
        printMatrix(VT);
        
        // Create diagonal matrix from singular values
        IMatrix<Double> S_matrix = Linalg.zeros(3, 3);
        for (int i = 0; i < 3; i++) {
            S_matrix.put(i, i, S.get(i));
        }
        System.out.println("S matrix (3x3):");
        printMatrix(S_matrix);
        
        // Reconstruct the original matrix
        IMatrix<Double> reconstructed = U.mmul(S_matrix).mmul(VT);
        System.out.println("Reconstructed matrix (4x3):");
        printMatrix(reconstructed);
        
        // Check reconstruction error
        double maxError = 0.0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                double error = Math.abs(data[i][j] - reconstructed.get(i, j));
                maxError = Math.max(maxError, error);
                System.out.printf("Original[%d][%d] = %f, Reconstructed[%d][%d] = %f, Error = %f%n", 
                    i, j, data[i][j], i, j, reconstructed.get(i, j), error);
            }
        }
        System.out.println("Max reconstruction error: " + maxError);
        
        // Verify orthogonality
        IMatrix<Double> UTU = U.transpose().mmul(U);
        IMatrix<Double> VTV = VT.mmul(VT.transpose());
        System.out.println("UTU matrix:");
        printMatrix(UTU);
        System.out.println("VTV matrix:");
        printMatrix(VTV);
        
        // Just print the results, don't assert
        System.out.println("Debug completed for 4x3 matrix");
    }
    
    private void printMatrix(IMatrix<Double> matrix) {
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                System.out.printf("%10.6f ", matrix.get(i, j));
            }
            System.out.println();
        }
        System.out.println();
    }
    
    private void printVector(IVector<Double> vector) {
        for (int i = 0; i < vector.length(); i++) {
            System.out.printf("%10.6f ", vector.get(i));
        }
        System.out.println();
    }
}