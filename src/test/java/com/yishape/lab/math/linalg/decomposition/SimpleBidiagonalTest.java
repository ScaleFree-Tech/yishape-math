package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereBidiagonalDecomposition;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;

public class SimpleBidiagonalTest {

    @Test
    public void testBidiagonalization() {
        // 使用与SVD测试相同的矩阵
        double[][] rectangularData = {
            { 1.0, 2.0, 3.0},
            { 2.0, 3.0, 4.0},
            { 3.0, 4.0, 5.0},
            { 4.0, 5.0, 6.0}
        };
        IMatrix<Double> A = Linalg.matrix(rectangularData);
        
        System.out.println("Original matrix A:");
        printMatrix(A);
        
        RereBidiagonalDecomposition bidiag = new RereBidiagonalDecomposition();
        Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> result = bidiag.decompose(A);
        
        IMatrix<Double> U = result.getFirst();
        IMatrix<Double> B = result.getSecond();
        IMatrix<Double> V = result.getThird();
        
        System.out.println("\nU matrix (" + U.getRowNum() + "x" + U.getColNum() + "):");
        printMatrix(U);
        
        System.out.println("\nBidiagonal matrix B (" + B.getRowNum() + "x" + B.getColNum() + "):");
        printMatrix(B);
        
        System.out.println("\nV matrix (" + V.getRowNum() + "x" + V.getColNum() + "):");
        printMatrix(V);
        
        // 验证 A = U * B * V^T
        IMatrix<Double> VT = V.transpose();
        
        System.out.println("\nMatrix dimensions:");
        System.out.println("A: " + A.getRowNum() + "x" + A.getColNum());
        System.out.println("U: " + U.getRowNum() + "x" + U.getColNum());
        System.out.println("B: " + B.getRowNum() + "x" + B.getColNum());
        System.out.println("V: " + V.getRowNum() + "x" + V.getColNum());
        System.out.println("V^T: " + VT.getRowNum() + "x" + VT.getColNum());
        
        try {
            System.out.println("\nAttempting U * B...");
            IMatrix<Double> UB = U.mmul(B);
            System.out.println("UB dimensions: " + UB.rows() + "x" + UB.cols());
            
            System.out.println("Attempting (U * B) * V^T...");
            IMatrix<Double> reconstructed = UB.mmul(VT);
            System.out.println("Reconstruction successful!");
            
            System.out.println("\nReconstructed matrix (U * B * V^T):");
            printMatrix(reconstructed);
            
            // 检查重构误差
            double maxError = 0.0;
            for (int i = 0; i < A.getRowNum(); i++) {
                for (int j = 0; j < A.getColNum(); j++) {
                    double error = Math.abs(A.get(i, j) - reconstructed.get(i, j));
                    maxError = Math.max(maxError, error);
                }
            }
            
            System.out.println("\nMaximum reconstruction error: " + maxError);
            
        } catch (Exception e) {
            System.out.println("Matrix multiplication failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void printMatrix(IMatrix<Double> matrix) {
        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                System.out.printf("%10.6f ", matrix.get(i, j));
            }
            System.out.println();
        }
    }
}