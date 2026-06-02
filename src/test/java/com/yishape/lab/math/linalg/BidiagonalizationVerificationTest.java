package com.yishape.lab.math.linalg;

import com.yishape.lab.math.linalg.decomposition.impl.RereBidiagonalDecomposition;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;

public class BidiagonalizationVerificationTest {

    @Test
    public void testBidiagonalizationAccuracy() {
        System.out.println("=== 双对角化验证测试 ===");
        
        // 使用与SVD调试相同的4x3测试矩阵
        double[][] testData = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {10.0, 11.0, 12.0}
        };
        
        IDoubleMatrix A = IDoubleMatrix.of(testData);
        System.out.println("原始矩阵 A (4x3):");
        printMatrix(A);
        
        // 执行双对角化
        RereBidiagonalDecomposition bidiag = new RereBidiagonalDecomposition();
        Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> result = bidiag.decompose(A);
        
        IDoubleMatrix U = (IDoubleMatrix) result.getFirst();
        IDoubleMatrix B = (IDoubleMatrix) result.getSecond();
        IDoubleMatrix V = (IDoubleMatrix) result.getThird();
        
        System.out.println("\n双对角化结果:");
        System.out.println("U 矩阵 (" + U.getRowNum() + "x" + U.getColNum() + "):");
        printMatrix(U);
        
        System.out.println("\nB 矩阵 (双对角矩阵) (" + B.getRowNum() + "x" + B.getColNum() + "):");
        printMatrix(B);
        
        System.out.println("\nV 矩阵 (" + V.getRowNum() + "x" + V.getColNum() + "):");
        printMatrix(V);
        
        // 验证重构：A = U * B * V^T
        IDoubleMatrix VT = (IDoubleMatrix) V.transpose();
        IDoubleMatrix UB = (IDoubleMatrix) U.mmul(B);
        IDoubleMatrix reconstructed = (IDoubleMatrix) UB.mmul(VT);
        
        System.out.println("\n重构矩阵 A' = U * B * V^T:");
        printMatrix(reconstructed);
        
        // 计算重构误差
        double maxError = 0.0;
        for (int i = 0; i < A.getRowNum(); i++) {
            for (int j = 0; j < A.getColNum(); j++) {
                double error = Math.abs(A.get(i, j) - reconstructed.get(i, j));
                maxError = Math.max(maxError, error);
            }
        }
        
        System.out.println("\n重构误差 (最大绝对误差): " + maxError);
        
        // 验证U和V的正交性
        System.out.println("\n正交性验证:");
        
        // 对于U矩阵，如果是m x minDim，则U^T * U应该是单位矩阵
        IDoubleMatrix UT = (IDoubleMatrix) U.transpose();
        IDoubleMatrix UTU = (IDoubleMatrix) UT.mmul(U);
        System.out.println("U^T * U (应该接近单位矩阵):");
        printMatrix(UTU);
        
        // 对于V矩阵，V^T * V应该是单位矩阵
        IDoubleMatrix VTV = (IDoubleMatrix) VT.mmul(V);
        System.out.println("\nV^T * V (应该是单位矩阵):");
        printMatrix(VTV);
        
        // 验证B是否为双对角矩阵
        System.out.println("\n双对角矩阵验证:");
        boolean isBidiagonal = true;
        for (int i = 0; i < B.getRowNum(); i++) {
            for (int j = 0; j < B.getColNum(); j++) {
                if (j != i && j != i + 1) {
                    if (Math.abs(B.get(i, j)) > 1e-10) {
                        isBidiagonal = false;
                        System.out.println("非零元素在位置 (" + i + "," + j + "): " + B.get(i, j));
                    }
                }
            }
        }
        System.out.println("是否为双对角矩阵: " + isBidiagonal);
    }
    
    private void printMatrix(IDoubleMatrix matrix) {
        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                System.out.printf("%8.6f ", matrix.get(i, j));
            }
            System.out.println();
        }
    }
}