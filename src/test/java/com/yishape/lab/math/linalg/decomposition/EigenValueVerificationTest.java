package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.math.linalg.IVector;
import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.EigenDecomposition;
import org.junit.jupiter.api.Test;

/**
 * Test to verify eigenvalue computation and investigate differences between implementations
 */
public class EigenValueVerificationTest {

    @Test
    void testEigenValueComputation() {
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
        
        // Verify eigenvalue equation: A * v = λ * v
        System.out.println("\nVerifying eigenvalue equation for RereMouse:");
        for (int i = 0; i < result._1.length(); i++) {
            double eigenvalue = result._1.get(i);
            System.out.println("Eigenvalue " + i + ": " + eigenvalue);
            
            // Get eigenvector
            double[] eigenvector = new double[result._2.rows()];
            for (int j = 0; j < result._2.rows(); j++) {
                eigenvector[j] = result._2.get(j, i).doubleValue();
            }
            
            System.out.println("Eigenvector " + i + ": " + java.util.Arrays.toString(eigenvector));
            
            // Compute A * v
            double[] Av = new double[3];
            for (int row = 0; row < 3; row++) {
                Av[row] = 0;
                for (int col = 0; col < 3; col++) {
                    Av[row] += testData[row][col] * eigenvector[col];
                }
            }
            
            // Compute λ * v
            double[] lambdaV = new double[3];
            for (int j = 0; j < 3; j++) {
                lambdaV[j] = eigenvalue * eigenvector[j];
            }
            
            System.out.println("A * v: " + java.util.Arrays.toString(Av));
            System.out.println("λ * v: " + java.util.Arrays.toString(lambdaV));
            
            // Check if they're close
            boolean close = true;
            for (int j = 0; j < 3; j++) {
                if (Math.abs(Av[j] - lambdaV[j]) > 1e-10) {
                    close = false;
                    break;
                }
            }
            System.out.println("Equation satisfied: " + close);
            System.out.println();
        }
        
        // Apache Commons Math computation
        Array2DRowRealMatrix commonsMatrix = new Array2DRowRealMatrix(testData);
        EigenDecomposition commonsEigen = new EigenDecomposition(commonsMatrix);
        
        System.out.println("Apache Commons Math results:");
        System.out.println("Eigenvalues: " + java.util.Arrays.toString(commonsEigen.getRealEigenvalues()));
        System.out.println("Eigenvectors:");
        double[][] vMatrix = commonsEigen.getV().getData();
        for (int i = 0; i < vMatrix.length; i++) {
            for (int j = 0; j < vMatrix[i].length; j++) {
                System.out.printf("%.6f ", vMatrix[i][j]);
            }
            System.out.println();
        }
        
        // Verify eigenvalue equation: A * v = λ * v for Commons Math
        System.out.println("\nVerifying eigenvalue equation for Commons Math:");
        double[] commonsEigenvalues = commonsEigen.getRealEigenvalues();
        for (int i = 0; i < commonsEigenvalues.length; i++) {
            double eigenvalue = commonsEigenvalues[i];
            System.out.println("Eigenvalue " + i + ": " + eigenvalue);
            
            // Get eigenvector (column i)
            double[] eigenvector = new double[vMatrix.length];
            for (int j = 0; j < vMatrix.length; j++) {
                eigenvector[j] = vMatrix[j][i];
            }
            
            System.out.println("Eigenvector " + i + ": " + java.util.Arrays.toString(eigenvector));
            
            // Compute A * v
            double[] Av = new double[3];
            for (int row = 0; row < 3; row++) {
                Av[row] = 0;
                for (int col = 0; col < 3; col++) {
                    Av[row] += testData[row][col] * eigenvector[col];
                }
            }
            
            // Compute λ * v
            double[] lambdaV = new double[3];
            for (int j = 0; j < 3; j++) {
                lambdaV[j] = eigenvalue * eigenvector[j];
            }
            
            System.out.println("A * v: " + java.util.Arrays.toString(Av));
            System.out.println("λ * v: " + java.util.Arrays.toString(lambdaV));
            
            // Check if they're close
            boolean close = true;
            for (int j = 0; j < 3; j++) {
                if (Math.abs(Av[j] - lambdaV[j]) > 1e-10) {
                    close = false;
                    break;
                }
            }
            System.out.println("Equation satisfied: " + close);
            System.out.println();
        }
    }
}