package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SymmetryCheckTest {

    @Test
    public void testMatrixSymmetry() {
        // Create a 3x3 symmetric positive definite test matrix (same as in RereCholeskyDecompositionTest)
        double[][] testData = {
            { 4.0, 12.0, -16.0},
            { 12.0, 37.0, -43.0},
            { -16.0, -43.0, 98.0}
        };
        
        // Check symmetry manually
        double relativeSymmetryThreshold = 1.0e-15;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double diff = Math.abs(testData[i][j] - testData[j][i]);
                double maxAbs = Math.max(Math.abs(testData[i][j]), Math.abs(testData[j][i]));
                boolean isSymmetric = (diff <= relativeSymmetryThreshold * maxAbs);
                
                assertTrue(isSymmetric, 
                    "Matrix should be symmetric at (" + i + "," + j + 
                    ") but diff=" + diff + ", threshold=" + (relativeSymmetryThreshold * maxAbs));
            }
        }
    }
}