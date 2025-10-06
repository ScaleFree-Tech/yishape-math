package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.util.Tuple2;

public class TridiagonalizationDebugTest {
    public static void main(String[] args) {
        // Create the same test matrix
        double[][] testData = {
            { 4.0, 2.0, 1.0},
            { 2.0, 3.0, 1.0},
            { 1.0, 1.0, 2.0}
        };
        
        System.out.println("Input matrix:");
        printMatrix(testData);
        
        // Test the tridiagonalization process step by step
        IMatrix<Double> matrix = Linalg.matrix(testData);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        
        // Test the full decomposition
        try {
            Tuple2<IVector<Double>,
                    IMatrix<Double>> result =
                eigen.decompose(matrix);
            
            System.out.println("\nDecomposition result:");
            System.out.println("Eigenvalues: " + java.util.Arrays.toString(result._1.toDoubleArray()));
            
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void printMatrix(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.printf("%.2f ", matrix[i][j]);
            }
            System.out.println();
        }
    }
}