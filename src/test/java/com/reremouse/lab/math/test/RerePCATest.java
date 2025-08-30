package com.reremouse.lab.math.test;

import com.reremouse.lab.math.dimreduce.RerePCA;
import com.reremouse.lab.math.dimreduce.RerePCA;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;

/**
 * PCA算法测试类
 */
public class RerePCATest {
    
    public static void main(String[] args) {
        testPCA();
    }
    
    public static void testPCA() {
        System.out.println("=== PCA算法测试 ===");
        
        // 创建测试数据 (4x3矩阵: 4个样本, 3个特征)
        float[][] testData = {
            {2.5f, 2.4f, 1.0f},  // 样本1
            {0.5f, 0.7f, 0.2f},  // 样本2
            {2.2f, 2.9f, 1.1f},  // 样本3
            {1.9f, 2.2f, 0.9f}   // 样本4
        };
        
        IMatrix originalData = IMatrix.of(testData);
        System.out.println("原始数据:");
        printMatrix(originalData);
        
        // 测试数据中心化
        IMatrix centeredData = originalData.center();
        System.out.println("\n中心化后的数据:");
        printMatrix(centeredData);
        
        // 测试协方差矩阵计算
        IMatrix covMatrix = centeredData.covarianceFromCentered();
        System.out.println("\n协方差矩阵:");
        printMatrix(covMatrix);
        
        // 测试特征分解
        try {
            System.out.println("\n开始特征分解...");
            Tuple2<IVector, IMatrix> eigenResult = covMatrix.eigen();
            IVector eigenValues = eigenResult._1;
            IMatrix eigenVectors = eigenResult._2;
            
            System.out.println("\n特征值:");
            printVector(eigenValues);
            
            System.out.println("\n特征向量:");
            printMatrix(eigenVectors);
            
        } catch (Exception e) {
            System.err.println("特征分解失败: " + e.getMessage());
            e.printStackTrace();
        }
        
        // 创建PCA对象并进行降维
        RerePCA pca = new RerePCA();
        
        try {
            System.out.println("\n开始PCA降维...");
            
            // 降维到2维
            System.out.println("尝试降维到2维...");
            IMatrix reducedData = pca.dimensionReduction(originalData, 2);
            System.out.println("降维到2维成功!");
            System.out.println("降维到2维后的数据:");
            printMatrix(reducedData);
            
            // 降维到1维
            System.out.println("\n尝试降维到1维...");
            IMatrix reducedData1D = pca.dimensionReduction(originalData, 1);
            System.out.println("降维到1维成功!");
            System.out.println("降维到1维后的数据:");
            printMatrix(reducedData1D);
            
            System.out.println("\n=== PCA测试完成 ===");
            
        } catch (Exception e) {
            System.err.println("PCA测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void printMatrix(IMatrix matrix) {
        try {
            float[][] data = matrix.getData();
            System.out.println("矩阵维度: " + data.length + "x" + data[0].length);
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    System.out.printf("%8.4f ", data[i][j]);
                }
                System.out.println();
            }
        } catch (Exception e) {
            System.err.println("打印矩阵时出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void printVector(IVector vector) {
        float[] data = vector.getData();
        for (int i = 0; i < data.length; i++) {
            System.out.printf("%8.4f ", data[i]);
        }
        System.out.println();
    }
} 