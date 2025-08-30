package com.reremouse.lab.math.test;

import com.reremouse.lab.math.dimreduce.RereSVD;
import com.reremouse.lab.math.IMatrix;

/**
 * SVD降维算法测试类 / SVD Dimensionality Reduction Test Class
 * 
 * @author lteb2
 */
public class RereSVDTest {
    
    public static void main(String[] args) {
        testSVDDimensionReduction();
    }
    
    /**
     * 测试SVD降维功能 / Test SVD dimensionality reduction functionality
     */
    public static void testSVDDimensionReduction() {
        System.out.println("=== SVD降维算法测试 / SVD Dimensionality Reduction Test ===");
        
        // 创建测试数据：4x3矩阵（4个样本，3个特征）
        // Create test data: 4x3 matrix (4 samples, 3 features)
        float[][] testData = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f}, 
            {7.0f, 8.0f, 9.0f},
            {2.0f, 3.0f, 4.0f}
        };
        
        IMatrix originalMatrix = IMatrix.of(testData);
        RereSVD svd = new RereSVD();
        
        System.out.println("原始数据矩阵 / Original data matrix:");
        printMatrix(originalMatrix);
        
        try {
            // 测试降维到2维 / Test reduction to 2 dimensions
            System.out.println("\n降维到2维 / Reduce to 2 dimensions:");
            IMatrix reduced2D = svd.dimensionReduction(originalMatrix, 2);
            printMatrix(reduced2D);
            System.out.println("降维后矩阵形状: " + reduced2D.getRowNum() + "x" + reduced2D.getColNum());
            
            // 测试降维到1维 / Test reduction to 1 dimension
            System.out.println("\n降维到1维 / Reduce to 1 dimension:");
            IMatrix reduced1D = svd.dimensionReduction(originalMatrix, 1);
            printMatrix(reduced1D);
            System.out.println("降维后矩阵形状: " + reduced1D.getRowNum() + "x" + reduced1D.getColNum());
            
            // 测试边界情况：降维到原始维度 / Test edge case: reduce to original dimension
            System.out.println("\n保持原始维度 / Keep original dimensions:");
            IMatrix sameSize = svd.dimensionReduction(originalMatrix, 3);
            printMatrix(sameSize);
            
            System.out.println("\n✓ SVD降维算法测试成功 / SVD dimensionality reduction test passed!");
            
        } catch (Exception e) {
            System.err.println("× SVD降维算法测试失败 / SVD dimensionality reduction test failed: " + e.getMessage());
            e.printStackTrace();
        }
        
        // 测试异常情况 / Test exception cases
        testExceptionCases(svd, originalMatrix);
    }
    
    /**
     * 测试异常情况 / Test exception cases
     */
    private static void testExceptionCases(RereSVD svd, IMatrix matrix) {
        System.out.println("\n=== 异常情况测试 / Exception Cases Test ===");
        
        try {
            // 测试无效维度 / Test invalid dimension
            svd.dimensionReduction(matrix, 0);
            System.err.println("× 应该抛出异常：维度为0 / Should throw exception: dimension is 0");
        } catch (IllegalArgumentException e) {
            System.out.println("✓ 正确处理维度为0的异常 / Correctly handled dimension 0 exception");
        }
        
        try {
            // 测试维度过大 / Test dimension too large
            svd.dimensionReduction(matrix, 5);
            System.err.println("× 应该抛出异常：维度过大 / Should throw exception: dimension too large");
        } catch (IllegalArgumentException e) {
            System.out.println("✓ 正确处理维度过大的异常 / Correctly handled dimension too large exception");
        }
        
        try {
            // 测试空矩阵 / Test null matrix
            svd.dimensionReduction(null, 2);
            System.err.println("× 应该抛出异常：空矩阵 / Should throw exception: null matrix");
        } catch (IllegalArgumentException e) {
            System.out.println("✓ 正确处理空矩阵异常 / Correctly handled null matrix exception");
        }
    }
    
    /**
     * 打印矩阵 / Print matrix
     */
    private static void printMatrix(IMatrix matrix) {
        float[][] data = matrix.getData();
        for (int i = 0; i < data.length; i++) {
            System.out.print("[");
            for (int j = 0; j < data[i].length; j++) {
                System.out.printf("%8.4f", data[i][j]);
                if (j < data[i].length - 1) {
                    System.out.print(", ");
                }
            }
            System.out.println("]");
        }
    }
} 