package com.yishape.lab.math.linalg;

public class TestMatrixNegativeZero {
    public static void main(String[] args) {
        // 创建包含负零的矩阵
        double[][] data = {
            {-0.0, 0.0, 1.0},
            {-1.0, 0.000001, -0.000001},
            {2.5, -2.5, 0.0}
        };
        IDoubleMatrix matrix = new RereDoubleMatrix(data);
        
        System.out.println("矩阵内容:");
        System.out.println(matrix.toString());
    }
}