package com.reremouse.lab.math.test;


import com.reremouse.lab.math.dimreduce.RereSVD;
import com.reremouse.lab.math.IMatrix;

/**
 * 简单的SVD降维测试 / Simple SVD Dimensionality Reduction Test
 */
public class TestSVD {
    
    public static void main(String[] args) {
        System.out.println("=== SVD降维算法测试 / SVD Dimensionality Reduction Test ===");
        
        // 创建测试数据：4x3矩阵（4个样本，3个特征）
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
            // 测试降维到2维
            System.out.println("\n降维到2维 / Reduce to 2 dimensions:");
            IMatrix reduced2D = svd.dimensionReduction(originalMatrix, 2);
            printMatrix(reduced2D);
            System.out.println("降维后矩阵形状: " + reduced2D.getRowNum() + "x" + reduced2D.getColNum());
            
            // 测试降维到1维
            System.out.println("\n降维到1维 / Reduce to 1 dimension:");
            IMatrix reduced1D = svd.dimensionReduction(originalMatrix, 1);
            printMatrix(reduced1D);
            System.out.println("降维后矩阵形状: " + reduced1D.getRowNum() + "x" + reduced1D.getColNum());
            
            System.out.println("\n✓ SVD降维算法测试成功！/ SVD dimensionality reduction test passed!");
            
        } catch (Exception e) {
            System.err.println("× SVD降维算法测试失败 / SVD dimensionality reduction test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
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