package com.reremouse.lab.math.compute;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.RereMatrix;
import com.reremouse.lab.util.Tuple3;

/**
 * 调试测试，查看SVD的实际奇异值
 */
public class CPUComputeUtilsDebugTest {
    
    public static void main(String[] args) {
        // 创建一个简单的2x3矩阵
        float[][] data = {{1, 2, 3}, {4, 5, 6}};
        IMatrix matrix = new RereMatrix(data);
        
        System.out.println("原始矩阵:");
        for (int i = 0; i < matrix.getRows(); i++) {
            for (int j = 0; j < matrix.getColumns(); j++) {
                System.out.print(matrix.get(i, j) + " ");
            }
            System.out.println();
        }
        
        Tuple3<IMatrix, com.reremouse.lab.math.linalg.IVector, IMatrix> result = CPUComputeUtils.svd(matrix);
        IMatrix U = result._1;
        com.reremouse.lab.math.linalg.IVector singularValues = result._2;
        IMatrix VT = result._3;
        
        System.out.println("\nU维度: " + U.getRows() + "x" + U.getColumns());
        System.out.println("奇异值数量: " + singularValues.length());
        System.out.println("V^T维度: " + VT.getRows() + "x" + VT.getColumns());
        
        // 打印奇异值
        System.out.print("奇异值: ");
        for (int i = 0; i < singularValues.length(); i++) {
            System.out.print(singularValues.get(i) + " ");
        }
        System.out.println();
        
        // 检查每个奇异值
        for (int i = 0; i < singularValues.length(); i++) {
            System.out.println("奇异值[" + i + "] = " + singularValues.get(i) + " (是否>0: " + (singularValues.get(i) > 0) + ")");
        }
    }
}
