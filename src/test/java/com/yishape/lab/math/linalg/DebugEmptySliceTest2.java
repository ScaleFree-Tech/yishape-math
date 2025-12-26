package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class DebugEmptySliceTest2 {

    @Test
    public void debugEmptySlice2() {
        float[][] testData = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {5.0f, 6.0f, 7.0f, 8.0f},
            {9.0f, 10.0f, 11.0f, 12.0f},
            {13.0f, 14.0f, 15.0f, 16.0f}
        };

        IMatrix<Float> matrix = IFloatMatrix.of(testData);
        
        // Test the slice expression "2:2" for rows
        SliceExpressionParser.SliceResult rowResult = SliceExpressionParser.parse("2:2", 4);
        System.out.println("Row slice result: " + rowResult);
        System.out.println("Row actualStart: " + rowResult.actualStart);
        System.out.println("Row actualEnd: " + rowResult.actualEnd);
        System.out.println("Row step: " + rowResult.step);
        
        // Test the slice expression "1:3" for columns
        SliceExpressionParser.SliceResult colResult = SliceExpressionParser.parse("1:3", 4);
        System.out.println("Col slice result: " + colResult);
        System.out.println("Col actualStart: " + colResult.actualStart);
        System.out.println("Col actualEnd: " + colResult.actualEnd);
        System.out.println("Col step: " + colResult.step);
        
        // Calculate slice sizes
        int resultRows = calculateSliceSize(rowResult.actualStart, rowResult.actualEnd, rowResult.step);
        int resultCols = calculateSliceSize(colResult.actualStart, colResult.actualEnd, colResult.step);
        
        System.out.println("Result rows: " + resultRows);
        System.out.println("Result cols: " + resultCols);
        
        // Test the actual slice
        IMatrix<Float> result = matrix.slice("2:2", "1:3");
        System.out.println("Actual result rows: " + result.rows());
        System.out.println("Actual result cols: " + result.cols());
        
        // Test what happens when we create a matrix with 0 rows and 2 cols
        IMatrix<Float> testMatrix = IFloatMatrix.of(new float[0][2]);
        System.out.println("Test matrix rows: " + testMatrix.rows());
        System.out.println("Test matrix cols: " + testMatrix.cols());
    }
    
    // Copy the calculateSliceSize method for testing
    private int calculateSliceSize(int start, int end, int step) {
        if (step > 0) {
            return Math.max(0, (end - start + step - 1) / step);
        } else {
            // 对于负数步长，我们需要特殊处理
            // 当end为-1时，表示到开头
            int absStep = Math.abs(step);
            if (start < end) {
                return 0; // 如果start < end且step为负，没有元素
            }
            if (end == -1) {
                // 表示到开头，元素个数 = (start - 0) / abs(step) + 1
                return Math.max(0, start / absStep + 1);
            } else {
                // 一般情况：元素个数 = (start - end - 1) / abs(step) + 1
                return Math.max(0, (start - end - 1) / absStep + 1);
            }
        }
    }
}