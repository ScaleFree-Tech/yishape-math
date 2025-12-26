package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 调试切片方法的测试类
 */
public class SliceDebugTest {

    @Test
    public void debugNegativeStepSlicing() {
        double[][] testData = {
            {1.0, 2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0, 8.0},
            {9.0, 10.0, 11.0, 12.0},
            {13.0, 14.0, 15.0, 16.0}
        };
        
        IMatrix<Double> matrix = IDoubleMatrix.of(testData);
        
        // 测试负数步长切片: matrix[::-1, ::-1]
        // 这应该返回一个4x4的矩阵，元素按逆序排列
        IMatrix<Double> result = matrix.slice("::-1", "::-1");
        
        System.out.println("Result rows: " + result.rows());
        System.out.println("Result cols: " + result.cols());
        
        if (result.rows() > 0 && result.cols() > 0) {
            System.out.println("First element: " + result.get(0, 0));
            System.out.println("Last element: " + result.get(result.rows()-1, result.cols()-1));
        }
    }
    
    @Test
    public void debugSliceExpressionParser() {
        // 测试SliceExpressionParser对负数步长的处理
        try {
            SliceExpressionParser.SliceResult result = SliceExpressionParser.parse("::-1", 4);
            System.out.println("Start: " + result.start);
            System.out.println("End: " + result.end);
            System.out.println("Step: " + result.step);
            System.out.println("ActualStart: " + result.actualStart);
            System.out.println("ActualEnd: " + result.actualEnd);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}