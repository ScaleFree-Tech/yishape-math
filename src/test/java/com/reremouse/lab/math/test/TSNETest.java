package com.reremouse.lab.math.test;

import com.reremouse.lab.math.dimreduce.RereTSNE;
import com.reremouse.lab.math.IMatrix;

/**
 * t-SNE算法测试类
 */
public class TSNETest {
    
    public static void main(String[] args) {
        // 创建一个简单的测试数据集
        // 4个样本，每个样本3维
        float[][] testData = {
            {1.0f, 2.0f, 3.0f},   // 样本1
            {4.0f, 5.0f, 6.0f},   // 样本2
            {7.0f, 8.0f, 9.0f},   // 样本3
            {2.0f, 3.0f, 4.0f}    // 样本4
        };
        
        IMatrix originalData = IMatrix.of(testData);
        
        System.out.println("原始数据维度: " + originalData.getRowNum() + "x" + originalData.getColNum());
        System.out.println("原始数据:");
        printMatrix(originalData);
        
        // 创建t-SNE实例并进行降维
        RereTSNE tsne = new RereTSNE();
        IMatrix reducedData = tsne.dimensionReduction(originalData, 2); // 降至2维
        
        System.out.println("\n降维后数据维度: " + reducedData.getRowNum() + "x" + reducedData.getColNum());
        System.out.println("降维后数据:");
        printMatrix(reducedData);
    }
    
    private static void printMatrix(IMatrix matrix) {
        float[][] data = matrix.getData();
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                System.out.printf("%.4f\t", data[i][j]);
            }
            System.out.println();
        }
    }
} 