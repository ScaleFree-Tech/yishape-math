package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 测试空切片行为
 */
public class EmptySliceTest {

    @Test
    public void testEmptySliceBehavior() {
        double[][] testData = {
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0},
            {9.0, 10.0, 11.0, 12.0},
            {13.0, 14.0, 15.0, 16.0}
        };
        
        IMatrix<Double> matrix = IDoubleMatrix.of(testData);
        
        // 测试空行切片
        IMatrix<Double> result = matrix.slice("2:2", "1:3");
        System.out.println("Empty row slice - rows: " + result.rows() + ", cols: " + result.cols());
        
        // 测试空列切片
        result = matrix.slice("1:3", "2:2");
        System.out.println("Empty column slice - rows: " + result.rows() + ", cols: " + result.cols());
    }
}