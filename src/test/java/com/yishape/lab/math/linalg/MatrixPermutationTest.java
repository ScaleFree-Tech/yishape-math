package com.yishape.lab.math.linalg;

import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;

/**
 * Test class for matrix permutation methods
 */
public class MatrixPermutationTest {
    
    @Test
    void testPermuteRows() {
        System.out.println("Testing matrix row permutation...");
        
        // Create a 3x3 test matrix
        double[][] testData = {
            { 1.0, 2.0, 3.0},
            { 4.0, 5.0, 6.0},
            { 7.0, 8.0, 9.0}
        };
        IMatrix<Double> matrix = Linalg.matrix(testData);
        
        // Permute rows: [0, 1, 2] -> [2, 0, 1]
        int[] rowPermutation = {2, 0, 1};
        IMatrix<Double> permutedMatrix = matrix.permuteRows(rowPermutation);
        
        // Check that the rows have been permuted correctly
        assertEquals(7.0, permutedMatrix.get(0, 0).doubleValue(), 1e-10);
        assertEquals(8.0, permutedMatrix.get(0, 1).doubleValue(), 1e-10);
        assertEquals(9.0, permutedMatrix.get(0, 2).doubleValue(), 1e-10);
        
        assertEquals(1.0, permutedMatrix.get(1, 0).doubleValue(), 1e-10);
        assertEquals(2.0, permutedMatrix.get(1, 1).doubleValue(), 1e-10);
        assertEquals(3.0, permutedMatrix.get(1, 2).doubleValue(), 1e-10);
        
        assertEquals(4.0, permutedMatrix.get(2, 0).doubleValue(), 1e-10);
        assertEquals(5.0, permutedMatrix.get(2, 1).doubleValue(), 1e-10);
        assertEquals(6.0, permutedMatrix.get(2, 2).doubleValue(), 1e-10);
        
        System.out.println("Matrix row permutation test passed.");
    }
    
    @Test
    void testPermuteColumns() {
        System.out.println("Testing matrix column permutation...");
        
        // Create a 3x3 test matrix
        double[][] testData = {
            { 1.0, 2.0, 3.0},
            { 4.0, 5.0, 6.0},
            { 7.0, 8.0, 9.0}
        };
        IMatrix<Double> matrix = Linalg.matrix(testData);
        
        // Permute columns: [0, 1, 2] -> [2, 0, 1]
        int[] colPermutation = {2, 0, 1};
        IMatrix<Double> permutedMatrix = matrix.permuteColumns(colPermutation);
        
        // Check that the columns have been permuted correctly
        assertEquals(3.0, permutedMatrix.get(0, 0).doubleValue(), 1e-10);
        assertEquals(1.0, permutedMatrix.get(0, 1).doubleValue(), 1e-10);
        assertEquals(2.0, permutedMatrix.get(0, 2).doubleValue(), 1e-10);
        
        assertEquals(6.0, permutedMatrix.get(1, 0).doubleValue(), 1e-10);
        assertEquals(4.0, permutedMatrix.get(1, 1).doubleValue(), 1e-10);
        assertEquals(5.0, permutedMatrix.get(1, 2).doubleValue(), 1e-10);
        
        assertEquals(9.0, permutedMatrix.get(2, 0).doubleValue(), 1e-10);
        assertEquals(7.0, permutedMatrix.get(2, 1).doubleValue(), 1e-10);
        assertEquals(8.0, permutedMatrix.get(2, 2).doubleValue(), 1e-10);
        
        System.out.println("Matrix column permutation test passed.");
    }
    
    @Test
    void testLinalgPermuteMethods() {
        System.out.println("Testing Linalg permutation methods...");
        
        // Create a 3x3 test matrix
        double[][] testData = {
            { 1.0, 2.0, 3.0},
            { 4.0, 5.0, 6.0},
            { 7.0, 8.0, 9.0}
        };
        IMatrix<Double> matrix = Linalg.matrix(testData);
        
        // Test Linalg.permuteRows method
        int[] rowPermutation = {1, 2, 0};
        IMatrix<Double> permutedRows = Linalg.permuteRows(matrix, rowPermutation);
        
        // Check that the rows have been permuted correctly
        assertEquals(4.0, permutedRows.get(0, 0).doubleValue(), 1e-10);
        assertEquals(7.0, permutedRows.get(1, 0).doubleValue(), 1e-10);
        assertEquals(1.0, permutedRows.get(2, 0).doubleValue(), 1e-10);
        
        // Test Linalg.permuteColumns method
        int[] colPermutation = {1, 2, 0};
        IMatrix<Double> permutedCols = Linalg.permuteColumns(matrix, colPermutation);
        
        // Check that the columns have been permuted correctly
        assertEquals(2.0, permutedCols.get(0, 0).doubleValue(), 1e-10);
        assertEquals(3.0, permutedCols.get(0, 1).doubleValue(), 1e-10);
        assertEquals(1.0, permutedCols.get(0, 2).doubleValue(), 1e-10);
        
        System.out.println("Linalg permutation methods test passed.");
    }
}