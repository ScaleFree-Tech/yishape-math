package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 更详细的切片调试测试
 */
public class SliceDebugTest2 {

    @Test
    public void debugSliceSizeCalculation() {
        double[][] testData = {
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0},
            {9.0, 10.0, 11.0, 12.0},
            {13.0, 14.0, 15.0, 16.0}
        };
        
        IMatrix<Double> matrix = IDoubleMatrix.of(testData);
        
        // 测试不同的切片表达式
        String[] expressions = {"::-1", "3::-1", "2::-1", "1::-1", "0::-1"};
        
        for (String expr : expressions) {
            try {
                IMatrix<Double> result = matrix.slice(expr, ":");
                System.out.println("Expression: " + expr + " -> rows: " + result.rows() + ", cols: " + result.cols());
            } catch (Exception e) {
                System.out.println("Expression: " + expr + " -> Error: " + e.getMessage());
            }
        }
    }
    
    @Test
    public void debugSimpleNegativeStep() {
        double[][] testData = {
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0}
        };
        
        IMatrix<Double> matrix = IDoubleMatrix.of(testData);
        
        // 测试简单的负数步长
        try {
            IMatrix<Double> result = matrix.slice("::-1", ":");
            System.out.println("Simple negative step - rows: " + result.rows() + ", cols: " + result.cols());
            
            if (result.rows() > 0 && result.cols() > 0) {
                System.out.println("First row first element: " + result.get(0, 0));
                System.out.println("Second row first element: " + result.get(1, 0));
            }
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}