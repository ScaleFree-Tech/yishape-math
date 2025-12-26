package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class DebugEmptySliceTest3 {

    @Test
    public void debugEmptySlice() {
        // 创建测试矩阵
        float[][] testData = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {5.0f, 6.0f, 7.0f, 8.0f},
            {9.0f, 10.0f, 11.0f, 12.0f},
            {13.0f, 14.0f, 15.0f, 16.0f}
        };
        
        IMatrix<Float> matrix = IFloatMatrix.of(testData);
        
        System.out.println("Original matrix shape: " + matrix.rows() + "x" + matrix.cols());
        
        // 测试空行切片
        IMatrix<Float> result = matrix.slice("2:2", "1:3");
        
        System.out.println("Result matrix shape: " + result.rows() + "x" + result.cols());
        
        // 检查结果
        assertEquals(0, result.rows(), "Expected 0 rows");
        assertEquals(2, result.cols(), "Expected 2 columns");
    }
}