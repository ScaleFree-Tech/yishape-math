package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereBidiagonalDecomposition;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;
import java.io.FileWriter;
import java.io.IOException;

public class DebugBidiagonal {
    
    @Test
    public void debugBidiagonalization() {
        try (FileWriter writer = new FileWriter("bidiagonal_debug.txt")) {
            // 使用与SVD测试相同的矩阵
            double[][] rectangularData = {
                { 1.0, 2.0, 3.0},
                { 2.0, 3.0, 4.0},
                { 3.0, 4.0, 5.0},
                { 4.0, 5.0, 6.0}
            };
            IMatrix<Double> A = Linalg.matrix(rectangularData);
            
            writer.write("Original matrix A:\n");
            printMatrix(A, writer);
            
            RereBidiagonalDecomposition bidiag = new RereBidiagonalDecomposition();
            Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> result = bidiag.decompose(A);
            
            IMatrix<Double> U = result.getFirst();
            IMatrix<Double> B = result.getSecond();
            IMatrix<Double> V = result.getThird();
            
            writer.write("\nU matrix (" + U.getRowNum() + "x" + U.getColNum() + "):\n");
            printMatrix(U, writer);
            
            writer.write("\nBidiagonal matrix B (" + B.getRowNum() + "x" + B.getColNum() + "):\n");
            printMatrix(B, writer);
            
            writer.write("\nV matrix (" + V.getRowNum() + "x" + V.getColNum() + "):\n");
            printMatrix(V, writer);
            
            // 验证 A = U * B * V^T
            IMatrix<Double> VT = V.transpose();
            
            writer.write("\nMatrix dimensions:\n");
            writer.write("A: " + A.getRowNum() + "x" + A.getColNum() + "\n");
            writer.write("U: " + U.getRowNum() + "x" + U.getColNum() + "\n");
            writer.write("B: " + B.getRowNum() + "x" + B.getColNum() + "\n");
            writer.write("V: " + V.getRowNum() + "x" + V.getColNum() + "\n");
            writer.write("V^T: " + VT.getRowNum() + "x" + VT.getColNum() + "\n");
            
            try {
                writer.write("\nAttempting U * B...\n");
                writer.write("U dimensions: " + U.getRowNum() + "x" + U.getColNum() + "\n");
                writer.write("B dimensions: " + B.getRowNum() + "x" + B.getColNum() + "\n");
                writer.write("Expected result dimensions: " + U.getRowNum() + "x" + B.getColNum() + "\n");
                
                // 检查B矩阵的内容
                writer.write("\nB matrix content check:\n");
                for (int i = 0; i < B.getRowNum(); i++) {
                    for (int j = 0; j < B.getColNum(); j++) {
                        writer.write("B[" + i + "][" + j + "] = " + B.get(i, j) + "\n");
                    }
                }
                
                IMatrix<Double> UB = U.mmul(B);
                writer.write("U * B 计算成功\n");
                writer.write("UB dimensions: " + UB.getRowNum() + "x" + UB.getColNum() + "\n");
                
                // 继续计算 (U * B) * V^T
                writer.write("尝试计算 (U * B) * V^T:\n");
                writer.write("V^T dimensions: " + VT.getRowNum() + "x" + VT.getColNum() + "\n");
                
                IMatrix<Double> reconstructed = UB.mmul(VT);
                writer.write("重构矩阵计算成功\n");
                writer.write("Reconstructed A dimensions: " + reconstructed.getRowNum() + "x" + reconstructed.getColNum() + "\n");
                
                // 计算重构误差
                writer.write("\n重构误差分析:\n");
                double maxError = 0.0;
                for (int i = 0; i < A.getRowNum(); i++) {
                    for (int j = 0; j < A.getColNum(); j++) {
                        double error = Math.abs(A.get(i, j) - reconstructed.get(i, j));
                        maxError = Math.max(maxError, error);
                    }
                }
                writer.write("最大重构误差: " + maxError + "\n");
                
            } catch (Exception e) {
                writer.write("\nError during matrix multiplication: " + e.getMessage() + "\n");
                writer.write("Exception type: " + e.getClass().getName() + "\n");
                java.io.StringWriter sw = new java.io.StringWriter();
                java.io.PrintWriter pw = new java.io.PrintWriter(sw);
                e.printStackTrace(pw);
                writer.write("Stack trace:\n" + sw.toString() + "\n");
            }
            
            writer.write("\nDebug completed successfully.\n");
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    private static void printMatrix(IMatrix<Double> matrix, FileWriter writer) throws IOException {
        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                writer.write(String.format("%8.4f ", matrix.get(i, j)));
            }
            writer.write("\n");
        }
    }
}