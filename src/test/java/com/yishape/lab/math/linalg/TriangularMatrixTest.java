package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TriangularMatrixTest {
    
    @Test
    public void testLowerTriMatrix() {
        // Test 3x3 lower triangular matrix
        IMatrix<Double> lowerTri = Linalg.lowerTriMatrix(3);
        
        assertNotNull(lowerTri, "Lower triangular matrix should not be null");
        assertEquals(3, lowerTri.getRowNum(), "Matrix should be 3x3");
        assertEquals(3, lowerTri.getColNum(), "Matrix should be 3x3");
        
        // Check that lower triangular elements are 1.0
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j <= i; j++) {
                assertEquals(1.0, lowerTri.get(i, j), 1e-10, "Element at (" + i + "," + j + ") should be 1.0");
            }
        }
        
        // Check that upper triangular elements (excluding diagonal) are 0.0
        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
                assertEquals(0.0, lowerTri.get(i, j), 1e-10, "Element at (" + i + "," + j + ") should be 0.0");
            }
        }
    }
    
    @Test
    public void testUpperTriMatrix() {
        // Test 3x3 upper triangular matrix
        IMatrix<Double> upperTri = Linalg.upperTriMatrix(3);
        
        assertNotNull(upperTri, "Upper triangular matrix should not be null");
        assertEquals(3, upperTri.getRowNum(), "Matrix should be 3x3");
        assertEquals(3, upperTri.getColNum(), "Matrix should be 3x3");
        
        // Check that upper triangular elements are 1.0
        for (int i = 0; i < 3; i++) {
            for (int j = i; j < 3; j++) {
                assertEquals(1.0, upperTri.get(i, j), 1e-10, "Element at (" + i + "," + j + ") should be 1.0");
            }
        }
        
        // Check that lower triangular elements (excluding diagonal) are 0.0
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < i; j++) {
                assertEquals(0.0, upperTri.get(i, j), 1e-10, "Element at (" + i + "," + j + ") should be 0.0");
            }
        }
    }
    
    @Test
    public void testLowerTriMatrixSize1() {
        // Test 1x1 lower triangular matrix
        IMatrix<Double> lowerTri = Linalg.lowerTriMatrix(1);
        
        assertNotNull(lowerTri, "Lower triangular matrix should not be null");
        assertEquals(1, lowerTri.getRowNum(), "Matrix should be 1x1");
        assertEquals(1, lowerTri.getColNum(), "Matrix should be 1x1");
        assertEquals(1.0, lowerTri.get(0, 0), 1e-10, "Element should be 1.0");
    }
    
    @Test
    public void testUpperTriMatrixSize1() {
        // Test 1x1 upper triangular matrix
        IMatrix<Double> upperTri = Linalg.upperTriMatrix(1);
        
        assertNotNull(upperTri, "Upper triangular matrix should not be null");
        assertEquals(1, upperTri.getRowNum(), "Matrix should be 1x1");
        assertEquals(1, upperTri.getColNum(), "Matrix should be 1x1");
        assertEquals(1.0, upperTri.get(0, 0), 1e-10, "Element should be 1.0");
    }
}