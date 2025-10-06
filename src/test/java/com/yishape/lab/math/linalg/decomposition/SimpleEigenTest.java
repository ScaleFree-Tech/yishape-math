package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.math.linalg.IVector;

/**
 * Simple test to isolate eigenvalue computation issue
 */
public class SimpleEigenTest {

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
        
        // RereMouse computation
        IMatrix<Double> matrix = Linalg.matrix(testData);
        RereEigenDecomposition rereEigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = rereEigen.decompose(matrix);
        
        System.out.println("\nRereMouse results:");
        System.out.println("Eigenvalues: " + java.util.Arrays.toString(result._1.toDoubleArray()));
        System.out.println("Eigenvectors:");
        for (int i = 0; i < result._2.rows(); i++) {
            for (int j = 0; j < result._2.cols(); j++) {
                System.out.printf("%.6f ", result._2.get(i, j).doubleValue());
            }
            System.out.println();
        }
        
        // Check sorting
        double[] eigenvalues = result._1.toDoubleArray();
        System.out.println("\nEigenvalues before sorting should be:");
        System.out.println("In descending order: " + java.util.Arrays.toString(eigenvalues));
        
        // Sort manually to see what the correct order should be
        Double[] manualSort = new Double[eigenvalues.length];
        for (int i = 0; i < eigenvalues.length; i++) {
            manualSort[i] = eigenvalues[i];
        }
        java.util.Arrays.sort(manualSort, (a, b) -> Double.compare(b, a)); // Descending order
        System.out.println("Manually sorted (descending): " + java.util.Arrays.toString(manualSort));
    }
}