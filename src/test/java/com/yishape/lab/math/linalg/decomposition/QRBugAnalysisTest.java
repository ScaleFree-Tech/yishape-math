package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.decomposition.impl.RereBidiagonalDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecomposition;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;

public class QRBugAnalysisTest {
    @Test
    public void testQRAlgorithmBugAnalysis() {
        System.out.println("QR算法错误分析测试");
        System.out.println("==================");
        
        // 使用相同的4x3测试矩阵
        double[][] data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {10.0, 11.0, 12.0}
        };
        
        IMatrix A = Linalg.matrix(data);
        System.out.println("原始矩阵 A:");
        printMatrix((IDoubleMatrix) A);
        
        // 分析原始矩阵的特性
        System.out.println("\n原始矩阵分析:");
        System.out.println("矩阵维度: " + A.getRowNum() + "x" + A.getColNum());
        
        // 计算矩阵的Frobenius范数
        double frobeniusNorm = 0.0;
        for (int i = 0; i < A.getRowNum(); i++) {
            for (int j = 0; j < A.getColNum(); j++) {
                double val = ((IDoubleMatrix) A).get(i, j);
                frobeniusNorm += val * val;
            }
        }
        frobeniusNorm = Math.sqrt(frobeniusNorm);
        System.out.println("Frobenius范数: " + frobeniusNorm);
        
        // 检查矩阵的行是否线性相关
        System.out.println("\n检查行向量:");
        for (int i = 0; i < A.getRowNum(); i++) {
            System.out.printf("行%d: [%.1f, %.1f, %.1f]\n", i+1, ((IDoubleMatrix) A).get(i, 0), ((IDoubleMatrix) A).get(i, 1), ((IDoubleMatrix) A).get(i, 2));
        }
        
        // 检查行向量之间的关系
        System.out.println("\n行向量关系分析:");
        System.out.println("行2 - 行1 = [3, 3, 3] = 3 * [1, 1, 1]");
        System.out.println("行3 - 行2 = [3, 3, 3] = 3 * [1, 1, 1]");
        System.out.println("行4 - 行3 = [3, 3, 3] = 3 * [1, 1, 1]");
        System.out.println("所有行向量都在同一个2维子空间中，矩阵的秩为2");
        System.out.println("第3个奇异值为0证实了这一点");
        
        // 执行双对角化
        RereBidiagonalDecomposition bidiag = new RereBidiagonalDecomposition();
        Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bidiagResult = bidiag.decompose(A);
        
        IDoubleMatrix U_bidiag = (IDoubleMatrix) bidiagResult.getFirst();
        IDoubleMatrix B = (IDoubleMatrix) bidiagResult.getSecond();
        IDoubleMatrix V_bidiag = (IDoubleMatrix) bidiagResult.getThird();
        
        System.out.println("\n双对角矩阵 B:");
        printMatrix(B);
        
        // 提取对角线和超对角线元素
        System.out.println("\n双对角矩阵的对角线元素:");
        for (int i = 0; i < Math.min(B.getRowNum(), B.getColNum()); i++) {
            System.out.printf("alpha[%d] = %.6f\n", i, B.get(i, i));
        }
        
        System.out.println("\n双对角矩阵的超对角线元素:");
        for (int i = 0; i < Math.min(B.getRowNum(), B.getColNum() - 1); i++) {
            System.out.printf("beta[%d] = %.6f\n", i, B.get(i, i + 1));
        }
        
        // 执行完整的SVD
        RereSVDDecomposition svd = new RereSVDDecomposition();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdResult = svd.decompose(A);
        
        IDoubleMatrix U_svd = (IDoubleMatrix) svdResult.getFirst();
        IVector<Double> singularValues = svdResult.getSecond();
        IDoubleMatrix VT_svd = (IDoubleMatrix) svdResult.getThird();
        
        // 调试：检查U和V矩阵的正交性
        System.out.println("\n调试信息：");
        IMatrix<Double> U_check = U_svd.transpose().mmul(U_svd);
        System.out.println("U^T * U (对于矩形矩阵，应该是单位矩阵):");
        printMatrix((IDoubleMatrix) U_check);
        
        IMatrix<Double> V_check = VT_svd.transpose().mmul(VT_svd);
        System.out.println("\nV * V^T (应该是单位矩阵):");
        printMatrix((IDoubleMatrix) V_check);
        
        // 奇异值个数应该是min(m,n)
        int actualSingularValueCount = Math.min(A.getRowNum(), A.getColNum());
        
        System.out.println("\nSVD奇异值:");
        for (int i = 0; i < actualSingularValueCount; i++) {
            System.out.printf("s[%d] = %.6f\n", i, singularValues.get(i));
        }
        
        // 重构矩阵验证
        System.out.println("\n重构过程分析:");
        System.out.println("U矩阵维度: " + U_svd.getRowNum() + "x" + U_svd.getColNum());
        System.out.println("奇异值个数: " + actualSingularValueCount);
        System.out.println("V^T矩阵维度: " + VT_svd.getRowNum() + "x" + VT_svd.getColNum());
        
        // 验证V^T的存储方式
        System.out.println("\n=== V^T存储方式验证 ===");
        System.out.println("如果返回的是V^T，那么重构应该是: A = U * S * V^T");
        System.out.println("如果返回的是V，那么重构应该是: A = U * S * V^T^T = U * S * V");
        
        // S矩阵应该是方形对角矩阵，维度为min(m,n) x min(m,n)
        int minDim = Math.min(A.getRowNum(), A.getColNum());
        IDoubleMatrix S = (IDoubleMatrix) Linalg.zeros(minDim, minDim);
        for (int i = 0; i < minDim; i++) {
            S.set(i, i, singularValues.get(i));
        }
        
        System.out.println("\n奇异值矩阵 S:");
        printMatrix(S);
        
        IDoubleMatrix temp = (IDoubleMatrix) U_svd.mmul(S);
        System.out.println("\nU * S 的结果:");
        printMatrix(temp);
        
        IDoubleMatrix reconstructed1 = (IDoubleMatrix) temp.mmul(VT_svd);
        System.out.println("\n重构矩阵1 A' = U * S * V^T:");
        printMatrix(reconstructed1);
        
        // 尝试另一种重构方式：A = U * S * V^T^T = U * S * V
        IDoubleMatrix reconstructed2 = (IDoubleMatrix) temp.mmul(VT_svd.transpose());
        System.out.println("\n重构矩阵2 A' = U * S * V^T^T = U * S * V:");
        printMatrix(reconstructed2);
        
        // 计算两种重构方式的误差
        double error1 = 0.0, error2 = 0.0;
        for (int i = 0; i < A.getRowNum(); i++) {
            for (int j = 0; j < A.getColNum(); j++) {
                double original = ((IDoubleMatrix) A).get(i, j);
                double diff1 = original - reconstructed1.get(i, j);
                double diff2 = original - reconstructed2.get(i, j);
                error1 += diff1 * diff1;
                error2 += diff2 * diff2;
            }
        }
        error1 = Math.sqrt(error1);
        error2 = Math.sqrt(error2);
        
        System.out.println("\n重构误差比较:");
        System.out.println("方式1 (A = U * S * V^T): " + error1);
        System.out.println("方式2 (A = U * S * V): " + error2);
        
        if (error1 < error2) {
            System.out.println("结论: 返回的确实是V^T矩阵");
        } else {
            System.out.println("结论: 返回的可能是V矩阵而不是V^T");
        }
        
        // 详细分析奇异值精度
        System.out.println("\n奇异值精度分析:");
        for (int i = 0; i < actualSingularValueCount; i++) {
            System.out.printf("σ[%d] = %.15f\n", i, singularValues.get(i));
        }
        
        // 检查第三个奇异值是否真的为0
        double thirdSingularValue = singularValues.get(2);
        System.out.printf("第三个奇异值: %.15e\n", thirdSingularValue);
        System.out.println("第三个奇异值是否接近0: " + (Math.abs(thirdSingularValue) < 1e-10));
        
        // 尝试只使用前两个奇异值重构
        System.out.println("\n使用前两个奇异值重构:");
        IDoubleMatrix S_rank2 = (IDoubleMatrix) Linalg.zeros(minDim, minDim);
        S_rank2.set(0, 0, singularValues.get(0));
        S_rank2.set(1, 1, singularValues.get(1));
        // 强制第三个奇异值为0
        
        IDoubleMatrix A_rank2 = (IDoubleMatrix) U_svd.mmul(S_rank2).mmul(VT_svd);
        double errorRank2 = 0.0;
        for (int i = 0; i < A.getRowNum(); i++) {
            for (int j = 0; j < A.getColNum(); j++) {
                double diff = ((IDoubleMatrix) A).get(i, j) - A_rank2.get(i, j);
                errorRank2 += diff * diff;
            }
        }
        errorRank2 = Math.sqrt(errorRank2);
        System.out.println("使用前两个奇异值的重构误差: " + errorRank2);
        
        // 检查矩阵乘法的中间结果
        System.out.println("\n中间结果检查:");
        System.out.println("S矩阵维度: " + S.getRowNum() + "x" + S.getColNum());
        System.out.println("U矩阵维度: " + U_svd.getRowNum() + "x" + U_svd.getColNum());
        IDoubleMatrix US = (IDoubleMatrix) U_svd.mmul(S);
        System.out.println("U * S 的维度: " + US.getRowNum() + "x" + US.getColNum());
        System.out.println("V^T 的维度: " + VT_svd.getRowNum() + "x" + VT_svd.getColNum());
        
        // 使用误差更小的重构方式进行后续分析
        IDoubleMatrix reconstructed = (error1 < error2) ? reconstructed1 : reconstructed2;
        double finalError = Math.min(error1, error2);
        
        // 计算每个元素的误差
        System.out.println("\n逐元素误差分析（使用最佳重构方式）:");
        for (int i = 0; i < A.getRowNum(); i++) {
            for (int j = 0; j < A.getColNum(); j++) {
                double original = ((IDoubleMatrix) A).get(i, j);
                double reconstructed_val = reconstructed.get(i, j);
                double error = Math.abs(original - reconstructed_val);
                System.out.printf("A[%d][%d]: 原始=%.3f, 重构=%.3f, 误差=%.6f\n", 
                    i, j, original, reconstructed_val, error);
            }
        }
        
        // 计算重构误差
        double error = 0.0;
        for (int i = 0; i < A.getRowNum(); i++) {
            for (int j = 0; j < A.getColNum(); j++) {
                double diff = ((IDoubleMatrix) A).get(i, j) - reconstructed.get(i, j);
                error += diff * diff;
            }
        }
        error = Math.sqrt(error);
        System.out.println("\n最终重构误差: " + finalError);
        
        // 分析问题：比较双对角矩阵的对角元素与奇异值
        System.out.println("\n问题分析:");
        System.out.println("双对角矩阵的对角元素应该通过QR算法收敛到奇异值");
        System.out.println("但是我们看到了显著的差异，这表明QR算法有错误");
        
        // 检查QR算法的初始化问题
        System.out.println("\n可能的错误原因:");
        System.out.println("1. Wilkinson位移计算错误");
        System.out.println("2. Givens旋转应用错误");
        System.out.println("3. 迭代收敛条件错误");
        System.out.println("4. 矩阵更新顺序错误");
    }
    
    private static void printMatrix(IDoubleMatrix matrix) {
        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                System.out.printf("%8.3f ", matrix.get(i, j));
            }
            System.out.println();
        }
    }
}