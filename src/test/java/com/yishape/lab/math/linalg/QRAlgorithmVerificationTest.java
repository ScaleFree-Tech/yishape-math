package com.yishape.lab.math.linalg;

import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereBidiagonalDecomposition;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;

public class QRAlgorithmVerificationTest {

    @Test
    public void testQRAlgorithmOnBidiagonalMatrix() {
        System.out.println("=== QR算法验证测试 ===");
        
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
        
        // 步骤1：执行双对角化
        RereBidiagonalDecomposition bidiag = new RereBidiagonalDecomposition();
        Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bidiagResult = bidiag.decompose(A);
        
        IDoubleMatrix U_bidiag = (IDoubleMatrix) bidiagResult.getFirst();
        IDoubleMatrix B = (IDoubleMatrix) bidiagResult.getSecond();
        IDoubleMatrix V_bidiag = (IDoubleMatrix) bidiagResult.getThird();
        
        System.out.println("\n双对角化结果:");
        System.out.println("B 矩阵 (双对角矩阵):");
        printMatrix(B);
        
        // 步骤2：执行完整的SVD
        RereSVDDecomposition svd = new RereSVDDecomposition();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdResult = svd.decompose(A);
        
        IDoubleMatrix U_svd = (IDoubleMatrix) svdResult.getFirst();
        IVector<Double> singularValues = svdResult.getSecond();
        IDoubleMatrix VT_svd = (IDoubleMatrix) svdResult.getThird();
        
        System.out.println("\nSVD结果:");
        System.out.println("奇异值:");
        for (int i = 0; i < singularValues.length(); i++) {
            System.out.printf("%.6f ", singularValues.get(i));
        }
        System.out.println();
        
        System.out.println("\nU 矩阵 (SVD):");
        printMatrix(U_svd);
        
        System.out.println("\nVT 矩阵 (SVD):");
        printMatrix(VT_svd);
        
        // 验证SVD重构
        IDoubleMatrix S = IDoubleMatrix.zeros(U_svd.getColNum(), VT_svd.getRowNum());
        for (int i = 0; i < Math.min(singularValues.length(), Math.min(S.getRowNum(), S.getColNum())); i++) {
            S.set(i, i, singularValues.get(i));
        }
        
        IDoubleMatrix US = (IDoubleMatrix) U_svd.mmul(S);
        IDoubleMatrix reconstructed = (IDoubleMatrix) US.mmul(VT_svd);
        
        System.out.println("\nSVD重构矩阵 A' = U * S * V^T:");
        printMatrix(reconstructed);
        
        // 计算重构误差
        double maxError = 0.0;
        for (int i = 0; i < A.getRowNum(); i++) {
            for (int j = 0; j < A.getColNum(); j++) {
                double error = Math.abs(A.get(i, j) - reconstructed.get(i, j));
                maxError = Math.max(maxError, error);
            }
        }
        
        System.out.println("\nSVD重构误差 (最大绝对误差): " + maxError);
        
        // 比较双对角化的B矩阵对角线元素与SVD奇异值
        System.out.println("\n对角线元素比较:");
        System.out.println("双对角化B矩阵对角线元素:");
        for (int i = 0; i < Math.min(B.getRowNum(), B.getColNum()); i++) {
            System.out.printf("%.6f ", Math.abs(B.get(i, i)));
        }
        System.out.println();
        
        System.out.println("SVD奇异值:");
        for (int i = 0; i < singularValues.length(); i++) {
            System.out.printf("%.6f ", singularValues.get(i));
        }
        System.out.println();
        
        // 检查QR算法是否正确收敛
        System.out.println("\n分析:");
        System.out.println("如果QR算法正确，SVD奇异值应该是双对角化B矩阵对角线元素的绝对值（排序后）");
        
        // 提取B矩阵的对角线元素并排序
        double[] bDiagonal = new double[Math.min(B.getRowNum(), B.getColNum())];
        for (int i = 0; i < bDiagonal.length; i++) {
            bDiagonal[i] = Math.abs(B.get(i, i));
        }
        java.util.Arrays.sort(bDiagonal);
        
        // 反转数组以获得降序
        for (int i = 0; i < bDiagonal.length / 2; i++) {
            double temp = bDiagonal[i];
            bDiagonal[i] = bDiagonal[bDiagonal.length - 1 - i];
            bDiagonal[bDiagonal.length - 1 - i] = temp;
        }
        
        System.out.println("排序后的B对角线元素:");
        for (double val : bDiagonal) {
            System.out.printf("%.6f ", val);
        }
        System.out.println();
        
        boolean qrCorrect = true;
        for (int i = 0; i < Math.min(bDiagonal.length, singularValues.length()); i++) {
            double diff = Math.abs(bDiagonal[i] - singularValues.get(i));
            if (diff > 1e-6) {
                qrCorrect = false;
                System.out.println("差异在位置 " + i + ": " + diff);
            }
        }
        
        System.out.println("QR算法是否正确: " + qrCorrect);
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