package com.reremouse.lab.math.test;

import com.reremouse.lab.math.dimreduce.RereUMAP;
import com.reremouse.lab.math.IMatrix;

/**
 * UMAP算法测试类
 * Test class for UMAP algorithm
 * 
 * @author lteb2
 */
public class UMAPTest {
    
    public static void main(String[] args) {
        System.out.println("=== UMAP 降维算法测试 ===");
        
        // 创建测试数据 - 3D 数据降到 2D
        float[][] testData = {
            {1.0f, 2.0f, 3.0f},
            {2.0f, 3.0f, 4.0f},
            {3.0f, 4.0f, 5.0f},
            {4.0f, 5.0f, 6.0f},
            {1.5f, 2.5f, 3.5f},
            {2.5f, 3.5f, 4.5f},
            {3.5f, 4.5f, 5.5f},
            {0.5f, 1.5f, 2.5f},
            {5.0f, 6.0f, 7.0f},
            {6.0f, 7.0f, 8.0f}
        };
        
        IMatrix originalData = IMatrix.of(testData);
        System.out.println("原始数据维度: " + java.util.Arrays.toString(originalData.shape()));
        
        // 创建UMAP实例并进行降维
        RereUMAP umap = new RereUMAP();
        IMatrix reducedData = umap.dimensionReduction(originalData, 2);
        
        System.out.println("降维后数据维度: " + java.util.Arrays.toString(reducedData.shape()));
        
        // 输出降维结果
        System.out.println("\n降维结果:");
        float[][] result = reducedData.getData();
        for (int i = 0; i < result.length; i++) {
            System.out.printf("点 %d: [%.4f, %.4f]%n", 
                            i + 1, result[i][0], result[i][1]);
        }
        
        // 测试新添加的数学方法
        System.out.println("\n=== 测试新添加的数学方法 ===");
        
        // 测试向量距离计算
        var v1 = originalData.getRow(0);
        var v2 = originalData.getRow(1);
        
        System.out.printf("向量1与向量2的欧几里得距离: %.4f%n", v1.euclideanDistance(v2));
        System.out.printf("向量1与向量2的曼哈顿距离: %.4f%n", v1.manhattanDistance(v2));
        System.out.printf("向量1与向量2的余弦相似度: %.4f%n", v1.cosineSimilarity(v2));
        
        // 测试矩阵方法
        System.out.printf("原始数据矩阵的Frobenius范数: %.4f%n", originalData.frobeniusNorm());
        System.out.printf("降维数据矩阵的Frobenius范数: %.4f%n", reducedData.frobeniusNorm());
        
        // 测试矩阵归一化
        IMatrix normalizedRows = originalData.normalizeRows();
        System.out.printf("按行归一化后的矩阵Frobenius范数: %.4f%n", normalizedRows.frobeniusNorm());
        
        System.out.println("\n=== UMAP测试完成 ===");
    }
} 