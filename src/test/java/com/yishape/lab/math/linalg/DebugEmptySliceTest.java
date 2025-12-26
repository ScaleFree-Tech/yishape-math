package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class DebugEmptySliceTest {

    @Test
    public void debugEmptySlice() {
        float[][] testData = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {5.0f, 6.0f, 7.0f, 8.0f},
            {9.0f, 10.0f, 11.0f, 12.0f},
            {13.0f, 14.0f, 15.0f, 16.0f}
        };

        IMatrix<Float> matrix = IFloatMatrix.of(testData);
        
        System.out.println("Original matrix dimensions: " + matrix.rows() + "x" + matrix.cols());

        // Test the problematic slice
        IMatrix<Float> result = matrix.slice("2:2", "1:3");
        
        System.out.println("Result matrix dimensions: " + result.rows() + "x" + result.cols());
        System.out.println("Expected: 0x2");
        
        // Debug: Check if the result is using our special empty matrix
        if (result instanceof RereFloatMatrix) {
            RereFloatMatrix rereResult = (RereFloatMatrix) result;
            System.out.println("Result data length: " + rereResult.getData().length);
            if (rereResult.getData().length == 0) {
                System.out.println("Empty matrix cols field: " + rereResult.emptyMatrixCols);
            }
        }
        
        // Let's also test creating an empty matrix directly
        float[][] emptyData = new float[0][2];
        IMatrix<Float> emptyMatrix = IFloatMatrix.of(emptyData);
        System.out.println("Direct empty matrix dimensions: " + emptyMatrix.rows() + "x" + emptyMatrix.cols());
        
        assertEquals(0, result.rows());
        assertEquals(2, result.cols());
    }
}