package com.reremouse.lab.math.test;

import com.reremouse.lab.math.RereMatrix;
import com.reremouse.lab.util.Tuple3;
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;

public class DebugPinv3x2 {
    public static void main(String[] args) {
        // 测试3x2矩阵
        float[][] data3x2 = {{1, 2}, {3, 4}, {5, 6}};
        IMatrix A = new RereMatrix(data3x2);
        
        System.out.println("测试矩阵 A (3x2):");
        printMatrix(A);
        
        try {
            // 先测试SVD
            System.out.println("\n进行SVD分解...");
            Tuple3<IMatrix, IVector, IMatrix> svdResult = A.svd();
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
            IMatrix pinvA = A.pinv();
            System.out.println("伪逆矩阵 A+ (" + pinvA.getRowNum() + "x" + pinvA.getColNum() + "):");
            printMatrix(pinvA);
            
            // 验证 A * A+ * A = A
            System.out.println("\n验证 A * A+ * A = A:");
            System.out.println("A的维度: " + A.getRowNum() + "x" + A.getColNum());
            System.out.println("A+的维度: " + pinvA.getRowNum() + "x" + pinvA.getColNum());
            
            IMatrix temp = ((RereMatrix)A).mmul(pinvA);
            System.out.println("A * A+的维度: " + temp.getRowNum() + "x" + temp.getColNum());
            printMatrix(temp);
            
            IMatrix result = ((RereMatrix)temp).mmul(A);
            System.out.println("(A * A+) * A的维度: " + result.getRowNum() + "x" + result.getColNum());
            printMatrix(result);
            
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