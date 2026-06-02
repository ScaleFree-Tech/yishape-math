package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * RereDoubleMatrix 切片方法测试类
 * 测试 slice 方法是否支持所有 NumPy 风格的切片操作，包括负数索引
 */
public class RereDoubleMatrixSliceTest {

    private final double[][] testData = {
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0, 16.0}
    };

    private IMatrix<Double> createTestMatrix() {
        return IDoubleMatrix.of(testData);
    }

    @Test
    public void testBasicSlicing() {
        IMatrix<Double> matrix = createTestMatrix();
        
        // 测试基本切片: matrix[1:3, 1:3]
        IMatrix<Double> result = matrix.slice("1:3", "1:3");
        assertEquals(2, result.rows());
        assertEquals(2, result.cols());
        assertEquals(6.0, result.get(0, 0));
        assertEquals(7.0, result.get(0, 1));
        assertEquals(10.0, result.get(1, 0));
        assertEquals(11.0, result.get(1, 1));
    }

    @Test
    public void testNegativeIndexing() {
        IMatrix<Double> matrix = createTestMatrix();
        
        // 测试负数索引: matrix[-2:-1, -3:-1]
        // -2 对应索引 2, -1 对应索引 3, -3 对应索引 1
        IMatrix<Double> result = matrix.slice("-2:-1", "-3:-1");
        assertEquals(1, result.rows());
        assertEquals(2, result.cols());
        assertEquals(10.0, result.get(0, 0));
        assertEquals(11.0, result.get(0, 1));
    }

    @Test
    public void testNegativeStartIndex() {
        IMatrix<Double> matrix = createTestMatrix();
        
        // 测试负数起始索引: matrix[-2:, 1:]
        // -2 对应索引 2, 所以取第2行到末尾
        // 第2行: [9.0, 10.0, 11.0, 12.0], 第3行: [13.0, 14.0, 15.0, 16.0]
        // 从第1列开始: [10.0, 11.0, 12.0] 和 [14.0, 15.0, 16.0]
        IMatrix<Double> result = matrix.slice("-2:", "1:");
        assertEquals(2, result.rows());
        assertEquals(3, result.cols());
        assertEquals(10.0, result.get(0, 0));  // matrix[2, 1]
        assertEquals(11.0, result.get(0, 1));  // matrix[2, 2]
        assertEquals(14.0, result.get(1, 0));  // matrix[3, 1]
        assertEquals(15.0, result.get(1, 1));  // matrix[3, 2]
    }

    @Test
    public void testNegativeEndIndex() {
        IMatrix<Double> matrix = createTestMatrix();
        
        // 测试负数结束索引: matrix[:3, :-1]
        // :-1 表示到倒数第二个元素（不包含最后一个）
        IMatrix<Double> result = matrix.slice(":3", ":-1");
        assertEquals(3, result.rows());
        assertEquals(3, result.cols());
        assertEquals(1.0, result.get(0, 0));
        assertEquals(3.0, result.get(0, 2));
        assertEquals(9.0, result.get(2, 0));
        assertEquals(11.0, result.get(2, 2));
    }

    @Test
    public void testStepSlicing() {
        IMatrix<Double> matrix = createTestMatrix();
        
        // 测试步长切片: matrix[::2, ::2]
        IMatrix<Double> result = matrix.slice("::2", "::2");
        assertEquals(2, result.rows());
        assertEquals(2, result.cols());
        assertEquals(1.0, result.get(0, 0));
        assertEquals(3.0, result.get(0, 1));
        assertEquals(9.0, result.get(1, 0));
        assertEquals(11.0, result.get(1, 1));
    }

    @Test
    public void testNegativeStepSlicing() {
        IMatrix<Double> matrix = createTestMatrix();
        
        // 测试负数步长切片: matrix[::-1, ::-1]
        IMatrix<Double> result = matrix.slice("::-1", "::-1");
        assertEquals(4, result.rows());
        assertEquals(4, result.cols());
        assertEquals(16.0, result.get(0, 0));
        assertEquals(13.0, result.get(0, 3));
        assertEquals(4.0, result.get(3, 0));
        assertEquals(1.0, result.get(3, 3));
    }

    @Test
    public void testMixedNegativeAndPositive() {
        IMatrix<Double> matrix = createTestMatrix();
        
        // 测试混合正负索引: matrix[1:-1, 0:3]
        IMatrix<Double> result = matrix.slice("1:-1", "0:3");
        assertEquals(2, result.rows());
        assertEquals(3, result.cols());
        assertEquals(5.0, result.get(0, 0));
        assertEquals(7.0, result.get(0, 2));
        assertEquals(9.0, result.get(1, 0));
        assertEquals(11.0, result.get(1, 2));
    }

    @Test
    public void testSingleDimensionSlicing() {
        IMatrix<Double> matrix = createTestMatrix();
        
        // 测试单维度切片
        IMatrix<Double> rowResult = matrix.sliceRows("1:3");
        assertEquals(2, rowResult.rows());
        assertEquals(4, rowResult.cols());
        
        IMatrix<Double> colResult = matrix.sliceColumns("1:3");
        assertEquals(4, colResult.rows());
        assertEquals(2, colResult.cols());
    }

    @Test
    public void testEdgeCases() {
        IMatrix<Double> matrix = createTestMatrix();
        
        // 测试边界情况
        IMatrix<Double> result1 = matrix.slice("0:1", "0:1");  // 单个元素
        assertEquals(1, result1.rows());
        assertEquals(1, result1.cols());
        assertEquals(1.0, result1.get(0, 0));
        
        IMatrix<Double> result2 = matrix.slice("3:4", "3:4");  // 最后一个元素
        assertEquals(1, result2.rows());
        assertEquals(1, result2.cols());
        assertEquals(16.0, result2.get(0, 0));
    }

    @Test
    public void testInvalidExpressions() {
        IMatrix<Double> matrix = createTestMatrix();
        
        // 测试无效表达式
        assertThrows(IllegalArgumentException.class, () -> {
            matrix.slice("invalid", "0:2");
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            matrix.slice("0:2", "invalid");
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            matrix.slice("0:2:0", "0:2");  // 步长为0
        });
    }

    @Test
    public void testEmptySlices() {
        IMatrix<Double> matrix = createTestMatrix();

        // 测试空切片 — 空行但非空列 → shape (0, 2)
        IMatrix<Double> result = matrix.slice("2:2", "1:3");
        assertEquals(0, result.rows());
        assertEquals(2, result.cols());

        // 空列但非空行 → shape (2, 0)
        result = matrix.slice("1:3", "2:2");
        assertEquals(2, result.rows());
        assertEquals(0, result.cols());

        // 测试完全空的切片
        result = matrix.slice("2:2", "2:2");  // 空行和空列
        assertEquals(0, result.rows());
        assertEquals(0, result.cols());
    }

    @Test
    public void testOutOfBoundsNegativeIndices() {
        IMatrix<Double> matrix = createTestMatrix();
        
        // 测试超出边界的负数索引
        // -5 超出了 4x4 矩阵的范围
        IMatrix<Double> result = matrix.slice("-5:", "0:2");
        assertEquals(4, result.rows());  // 应该被截断到有效范围
        assertEquals(2, result.cols());
    }

    @Test
    public void testComplexSlicing() {
        IMatrix<Double> matrix = createTestMatrix();
        
        // 测试复杂的切片组合: matrix[::-2, 1::-2]
        // 行: 从末尾开始，步长-2: [3, 1] (即第3行和第1行)
        // 列: 从索引1开始到开头，步长-2: [1, -1] 但-1无效，所以只有[1]
        // 实际上应该是: 行[3,1], 列[1]
        IMatrix<Double> result = matrix.slice("::-2", "1::-2");
        assertEquals(2, result.rows());
        assertEquals(1, result.cols());
        
        // 验证结果
        assertEquals(14.0, result.get(0, 0));  // matrix[3, 1]
        assertEquals(6.0, result.get(1, 0));   // matrix[1, 1]
    }

    @Test
    public void testNumPyCompatibility() {
        // 创建与 NumPy 测试兼容的矩阵
        IMatrix<Double> matrix = createTestMatrix();
        
        // 模拟 NumPy 的常见切片操作
        // matrix[1:3, 1:3] - 中心 2x2 子矩阵
        IMatrix<Double> center = matrix.slice("1:3", "1:3");
        assertEquals(6.0, center.get(0, 0));
        assertEquals(7.0, center.get(0, 1));
        assertEquals(10.0, center.get(1, 0));
        assertEquals(11.0, center.get(1, 1));
        
        // matrix[-2:, -2:] - 右下角 2x2 子矩阵
        IMatrix<Double> bottomRight = matrix.slice("-2:", "-2:");
        assertEquals(11.0, bottomRight.get(0, 0));
        assertEquals(12.0, bottomRight.get(0, 1));
        assertEquals(15.0, bottomRight.get(1, 0));
        assertEquals(16.0, bottomRight.get(1, 1));
        
        // matrix[::2, ::2] - 每隔一个元素
        IMatrix<Double> everyOther = matrix.slice("::2", "::2");
        assertEquals(1.0, everyOther.get(0, 0));
        assertEquals(3.0, everyOther.get(0, 1));
        assertEquals(9.0, everyOther.get(1, 0));
        assertEquals(11.0, everyOther.get(1, 1));
    }
}