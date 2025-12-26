package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class DebugEmptySliceTest4 {

    @Test
    public void debugSliceParsing() {
        // 创建测试矩阵
        float[][] testData = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {5.0f, 6.0f, 7.0f, 8.0f},
            {9.0f, 10.0f, 11.0f, 12.0f},
            {13.0f, 14.0f, 15.0f, 16.0f}
        };
        
        IMatrix<Float> matrix = IFloatMatrix.of(testData);
        
        System.out.println("Original matrix shape: " + matrix.rows() + "x" + matrix.cols());
        
        // 测试列切片解析
        SliceExpressionParser.SliceResult colResult = SliceExpressionParser.parse("1:3", matrix.cols());
        System.out.println("Column slice result:");
        System.out.println("  start: " + colResult.start);
        System.out.println("  end: " + colResult.end);
        System.out.println("  step: " + colResult.step);
        System.out.println("  actualStart: " + colResult.actualStart);
        System.out.println("  actualEnd: " + colResult.actualEnd);
        
        // 计算列数
        int resultCols = calculateSliceSize(colResult.actualStart, colResult.actualEnd, colResult.step);
        System.out.println("Calculated resultCols: " + resultCols);
        
        // 测试行切片解析
        SliceExpressionParser.SliceResult rowResult = SliceExpressionParser.parse("2:2", matrix.rows());
        System.out.println("Row slice result:");
        System.out.println("  start: " + rowResult.start);
        System.out.println("  end: " + rowResult.end);
        System.out.println("  step: " + rowResult.step);
        System.out.println("  actualStart: " + rowResult.actualStart);
        System.out.println("  actualEnd: " + rowResult.actualEnd);
        
        // 计算行数
        int resultRows = calculateSliceSize(rowResult.actualStart, rowResult.actualEnd, rowResult.step);
        System.out.println("Calculated resultRows: " + resultRows);
    }
    
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