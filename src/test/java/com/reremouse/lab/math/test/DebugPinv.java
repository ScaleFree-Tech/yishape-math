package com.reremouse.lab.math.test;

import com.reremouse.lab.math.RereMatrix;
import com.reremouse.lab.util.Tuple3;
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;

public class DebugPinv {
    public static void main(String[] args) {
        // 测试简单的2x2奇异矩阵
        float[][] singularData = {{1, 2}, {2, 4}};
        IMatrix singular = new RereMatrix(singularData);
        
        System.out.println("测试矩阵 (2x2):");
        printMatrix(singular);
        
        try {
            // 先测试SVD
            System.out.println("\n进行SVD分解...");
            Tuple3<IMatrix, IVector, IMatrix> svdResult = singular.svd();
            IMatrix U = svdResult._1;
            IVector S = svdResult._2;
            IMatrix VT = svdResult._3;
            
            System.out.println("U矩阵 (" + U.getRowNum() + "x" + U.getColNum() + "):");
            printMatrix(U);
            
            System.out.println("奇异值向量 (长度=" + S.length() + "):");
            printVector(S);
            
            System.out.println("VT矩阵 (" + VT.getRowNum() + "x" + VT.getColNum() + "):");
            printMatrix(VT);
            
            // 现在测试伪逆
            System.out.println("\n计算伪逆...");
            IMatrix pinv = singular.pinv();
            System.out.println("伪逆矩阵:");
            printMatrix(pinv);
            
        } catch (Exception e) {
            System.out.println("错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public static void printMatrix(IMatrix matrix) {
        float[][] data = matrix.getData();
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                System.out.printf("%8.4f ", data[i][j]);
            }
            System.out.println();
        }
    }
    
    public static void printVector(IVector vector) {
        float[] data = vector.getData();
        for (int i = 0; i < data.length; i++) {
            System.out.printf("%8.4f ", data[i]);
        }
        System.out.println();
    }
} 