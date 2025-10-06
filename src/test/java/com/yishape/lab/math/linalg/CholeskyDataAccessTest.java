package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CholeskyDataAccessTest {

    @Test
    public void testCholeskyDataAccess() {
        // Create a 3x3 symmetric positive definite test matrix (same as in RereCholeskyDecompositionTest)
        double[][] testData = {
            { 4.0, 12.0, -16.0},
            { 12.0, 37.0, -43.0},
            { -16.0, -43.0, 98.0}
        };
        IMatrix<Double> matrix = Linalg.matrix(testData);
        
        // Cast to IDoubleMatrix and check getData
        IDoubleMatrix doubleMatrix = (IDoubleMatrix) matrix;
        double[][] data = doubleMatrix.getData();
        
        // Check that the data is correctly retrieved
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(testData[i][j], data[i][j], 1e-10, 
                    "Data should be correctly retrieved at (" + i + "," + j + ")");
            }
        }
        
        // Manually perform the symmetry check that Cholesky decomposition does
        double relativeSymmetryThreshold = 1.0e-15;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double diff = Math.abs(data[i][j] - data[j][i]);
                double maxAbs = Math.max(Math.abs(data[i][j]), Math.abs(data[j][i]));
                if (diff > relativeSymmetryThreshold * maxAbs) {
                    fail("Matrix is not symmetric at (" + i + "," + j + 
                        ") but diff=" + diff + ", threshold=" + (relativeSymmetryThreshold * maxAbs));
                }
            }
        }
    }
}