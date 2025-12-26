package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * RereFloatMatrix 切片方法测试类
 * 测试 slice 方法是否支持所有 NumPy 风格的切片操作
 */
public class RereFloatMatrixSliceTest {

    private final float[][] testData = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
        {9.0f, 10.0f, 11.0f, 12.0f},
        {13.0f, 14.0f, 15.0f, 16.0f}
    };

    private IMatrix<Float> createTestMatrix() {
        return IFloatMatrix.of(testData);
    }

    @Test
    public void testBasicSlicing() {
        IMatrix<Float> matrix = createTestMatrix();

        // 测试基本切片: matrix[1:3, 1:3]
        IMatrix<Float> result = matrix.slice("1:3", "1:3");
        assertEquals(2, result.rows());
        assertEquals(2, result.cols());
        assertEquals(6.0f, result.get(0, 0));
        assertEquals(7.0f, result.get(0, 1));
        assertEquals(10.0f, result.get(1, 0));
        assertEquals(11.0f, result.get(1, 1));
    }

    @Test
    public void testNegativeIndexing() {
        IMatrix<Float> matrix = createTestMatrix();

        // 测试负数索引: matrix[-2:-1, -3:-1]
        IMatrix<Float> result = matrix.slice("-2:-1", "-3:-1");
        assertEquals(1, result.rows());
        assertEquals(2, result.cols());
        assertEquals(10.0f, result.get(0, 0));
        assertEquals(11.0f, result.get(0, 1));
    }

    @Test
    public void testNegativeStepSlicing() {
        IMatrix<Float> matrix = createTestMatrix();

        // 测试负数步长切片: matrix[::-1, ::-1]
        IMatrix<Float> result = matrix.slice("::-1", "::-1");
        assertEquals(4, result.rows());
        assertEquals(4, result.cols());
        assertEquals(16.0f, result.get(0, 0));
        assertEquals(13.0f, result.get(0, 3));
        assertEquals(4.0f, result.get(3, 0));
        assertEquals(1.0f, result.get(3, 3));
    }

    @Test
    public void testEmptySlices() {
        IMatrix<Float> matrix = createTestMatrix();

        // 测试空切片
        IMatrix<Float> result = matrix.slice("2:2", "1:3");
        assertEquals(0, result.rows());
        assertEquals(2, result.cols());
    }
}