package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MatrixDataTest {

    @Test
    public void testMatrixDataAccess() {
        // Create a 3x3 symmetric positive definite test matrix (same as in RereCholeskyDecompositionTest)
        double[][] testData = {
            { 4.0, 12.0, -16.0},
            { 12.0, 37.0, -43.0},
            { -16.0, -43.0, 98.0}
        };
        IMatrix<Double> matrix = Linalg.matrix(testData);
        
        // Check that the matrix is correctly created
        assertEquals(3, matrix.rows());
        assertEquals(3, matrix.cols());
        
        // Check symmetry manually
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(testData[i][j], testData[j][i], 1e-10, 
                    "Test data should be symmetric at (" + i + "," + j + ")");
            }
        }
        
        // Check that the matrix returns the correct values
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(testData[i][j], matrix.get(i, j).doubleValue(), 1e-10, 
                    "Matrix should return correct value at (" + i + "," + j + ")");
            }
        }
        
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
    }
}