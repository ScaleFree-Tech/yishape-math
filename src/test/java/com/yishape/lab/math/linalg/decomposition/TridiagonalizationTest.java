package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.math.linalg.IVector;

public class TridiagonalizationTest {
    public static void main(String[] args) {
        // Create the same test matrix
        double[][] testData = {
            { 4.0, 2.0, 1.0},
            { 2.0, 3.0, 1.0},
            { 1.0, 1.0, 2.0}
        };
        
        System.out.println("Input matrix:");
        for (int i = 0; i < testData.length; i++) {
            for (int j = 0; j < testData[i].length; j++) {
                System.out.printf("%.2f ", testData[i][j]);
            }
            System.out.println();
        }
        
        // Test the tridiagonalization process step by step
        IMatrix<Double> matrix = Linalg.matrix(testData);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        
        // Call the tridiagonalReduction method directly
        try {
            // We need to access the private method, so let's just run the full decomposition
            // and see what happens at each step
            Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(matrix);
            
            System.out.println("\nFull decomposition result:");
            System.out.println("Eigenvalues: " + java.util.Arrays.toString(result._1.toDoubleArray()));
            
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}