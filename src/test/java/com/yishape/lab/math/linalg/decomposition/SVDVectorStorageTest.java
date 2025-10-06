package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecomposition;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;

public class SVDVectorStorageTest {

    @Test
    public void testSVDVectorStorageFormat() {
        System.out.println("=== SVD向量存储方式验证测试 ===");
        
        // 使用一个简单的方阵
        double[][] data = {
            {3.0, 2.0},
            {2.0, 3.0}
        };
        
        IDoubleMatrix A = IDoubleMatrix.of(data);
        System.out.println("原始矩阵 A (2x2):");
        printMatrix(A);
        
        // 执行SVD分解
        RereSVDDecomposition svd = new RereSVDDecomposition();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);
        
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        
        System.out.println("\nSVD分解结果:");
        System.out.println("U矩阵 (" + U.getRowNum() + "x" + U.getColNum() + "):");
        printMatrix((IDoubleMatrix) U);
        
        System.out.println("\n奇异值:");
        for (int i = 0; i < S.length(); i++) {
            System.out.printf("s[%d] = %.6f\n", i, S.get(i));
        }
        
        System.out.println("\nV^T矩阵 (" + VT.getRowNum() + "x" + VT.getColNum() + "):");
        printMatrix((IDoubleMatrix) VT);
        
        // 验证重构：A = U * S * V^T
        System.out.println("\n=== 重构验证 ===");
        
        // 方法1：标准重构 A = U * S * V^T
        IDoubleMatrix S_matrix = (IDoubleMatrix) Linalg.zeros(U.getColNum(), VT.getRowNum());
        for (int i = 0; i < Math.min(S_matrix.getRowNum(), S_matrix.getColNum()); i++) {
            S_matrix.set(i, i, S.get(i));
        }
        
        System.out.println("奇异值矩阵 S (" + S_matrix.getRowNum() + "x" + S_matrix.getColNum() + "):");
        printMatrix(S_matrix);
        
        IDoubleMatrix reconstructed1 = (IDoubleMatrix) U.mmul(S_matrix).mmul(VT);
        System.out.println("\n重构矩阵1 (U * S * V^T):");
        printMatrix(reconstructed1);
        
        double error1 = calculateReconstructionError(A, reconstructed1);
        System.out.println("重构误差1: " + error1);
        
        // 方法2：尝试不同的组合方式
        // 如果V^T实际上是V，那么应该使用 A = U * S * V
        IDoubleMatrix reconstructed2 = (IDoubleMatrix) U.mmul(S_matrix).mmul(VT.transpose());
        System.out.println("\n重构矩阵2 (U * S * V^T^T = U * S * V):");
        printMatrix(reconstructed2);
        
        double error2 = calculateReconstructionError(A, reconstructed2);
        System.out.println("重构误差2: " + error2);
        
        // 验证U和V的正交性
        System.out.println("\n=== 正交性验证 ===");
        
        // 检查U^T * U
        IMatrix<Double> UTU = U.transpose().mmul(U);
        System.out.println("U^T * U:");
        printMatrix((IDoubleMatrix) UTU);
        
        // 检查V^T * V^T^T = V^T * V
        IMatrix<Double> VTTV = VT.mmul(VT.transpose());
        System.out.println("\nV^T * V^T^T = V^T * V:");
        printMatrix((IDoubleMatrix) VTTV);
        
        // 检查V^T^T * V^T = V * V^T
        IMatrix<Double> VVTT = VT.transpose().mmul(VT);
        System.out.println("\nV^T^T * V^T = V * V^T:");
        printMatrix((IDoubleMatrix) VVTT);
        
        // 分析哪种重构方式更准确
        System.out.println("\n=== 结论 ===");
        if (error1 < error2) {
            System.out.println("标准重构 A = U * S * V^T 更准确");
            System.out.println("这表明返回的确实是V^T矩阵");
        } else {
            System.out.println("修正重构 A = U * S * V 更准确");
            System.out.println("这表明返回的可能是V矩阵而不是V^T");
        }
        
        System.out.println("误差比较: 标准=" + error1 + ", 修正=" + error2);
    }
    
    private void printMatrix(IDoubleMatrix matrix) {
        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                System.out.printf("%8.6f ", matrix.get(i, j));
            }
            System.out.println();
        }
    }
    
    private double calculateReconstructionError(IDoubleMatrix original, IDoubleMatrix reconstructed) {
        double error = 0.0;
        for (int i = 0; i < original.getRowNum(); i++) {
            for (int j = 0; j < original.getColNum(); j++) {
                double diff = original.get(i, j) - reconstructed.get(i, j);
                error += diff * diff;
            }
        }
        return Math.sqrt(error);
    }
}