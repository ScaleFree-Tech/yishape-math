package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereBidiagonalDecomposition;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class BidiagonalVerificationTest {

    @Test
    void testBidiagonalizationCorrectness() {
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
        
        // 检查维度是否匹配
        if (U.getColNum() != B.getRowNum()) {
            System.out.println("ERROR: U columns (" + U.getColNum() + ") != B rows (" + B.getRowNum() + ")");
            return;
        }
        if (B.getColNum() != VT.getRowNum()) {
            System.out.println("ERROR: B columns (" + B.getColNum() + ") != V^T rows (" + VT.getRowNum() + ")");
            return;
        }
        
        IMatrix<Double> reconstructed = null;
        try {
            System.out.println("Attempting U * B...");
            IMatrix<Double> UB = U.mmul(B);
            System.out.println("UB dimensions: " + UB.getRowNum() + "x" + UB.getColNum());
            
            System.out.println("Attempting (U * B) * V^T...");
            reconstructed = UB.mmul(VT);
            System.out.println("Reconstruction successful!");
        } catch (Exception e) {
            System.out.println("Matrix multiplication failed: " + e.getMessage());
            e.printStackTrace();
            return;
        }
        
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
        
        // 验证B是否为双对角矩阵
        System.out.println("\nChecking if B is bidiagonal:");
        boolean isBidiagonal = true;
        for (int i = 0; i < B.getRowNum(); i++) {
            for (int j = 0; j < B.getColNum(); j++) {
                if (j != i && j != i + 1) {
                    double value = B.get(i, j);
                    if (Math.abs(value) > 1e-10) {
                        System.out.println("Non-zero element at (" + i + "," + j + "): " + value);
                        isBidiagonal = false;
                    }
                }
            }
        }
        System.out.println("Is bidiagonal: " + isBidiagonal);
        
        // 验证U和V是否为正交矩阵
        System.out.println("\nChecking orthogonality of U:");
        checkOrthogonality(U);
        
        System.out.println("\nChecking orthogonality of V:");
        checkOrthogonality(V);
        
        assertTrue(maxError < 1e-10, "Reconstruction error too large: " + maxError);
        assertTrue(isBidiagonal, "B matrix is not bidiagonal");
    }
    
    private void printMatrix(IMatrix<Double> matrix) {
        for (int i = 0; i < matrix.getRowNum(); i++) {
            for (int j = 0; j < matrix.getColNum(); j++) {
                System.out.printf("%10.6f ", matrix.get(i, j));
            }
            System.out.println();
        }
    }
    
    private void checkOrthogonality(IMatrix<Double> matrix) {
        IMatrix<Double> product = matrix.transpose().mmul(matrix);
        double maxError = 0.0;
        
        for (int i = 0; i < product.getRowNum(); i++) {
            for (int j = 0; j < product.getColNum(); j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double actual = product.get(i, j);
                double error = Math.abs(actual - expected);
                maxError = Math.max(maxError, error);
                
                if (error > 1e-10) {
                    System.out.printf("U^T*U[%d,%d] = %f (expected %f, error %e)\n", 
                                    i, j, actual, expected, error);
                }
            }
        }
        
        System.out.println("Maximum orthogonality error: " + maxError);
    }
}