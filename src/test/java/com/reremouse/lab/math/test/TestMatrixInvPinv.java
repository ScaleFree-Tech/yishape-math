package com.reremouse.lab.math.test;

import com.reremouse.lab.math.RereMatrix;
import com.reremouse.lab.math.IMatrix;

/**
 * 测试矩阵逆和伪逆的实现
 */
public class TestMatrixInvPinv {
    
    public static void main(String[] args) {
        testMatrixInverse();
        testMatrixPseudoInverse();
        testSingularMatrix();
        testNonSquareMatrix();
    }
    
    /**
     * 测试矩阵逆
     */
    public static void testMatrixInverse() {
        System.out.println("=== 测试矩阵逆 ===");
        
        // 测试2x2矩阵
        float[][] data2x2 = {{2, 1}, {1, 1}};
        IMatrix A = new RereMatrix(data2x2);
        
        System.out.println("原矩阵 A:");
        printMatrix(A);
        
        try {
            IMatrix invA = A.inv();
            System.out.println("逆矩阵 A^(-1):");
            printMatrix(invA);
            
            // 验证 A * A^(-1) = I
            IMatrix I = ((RereMatrix)A).mmul(invA);
            System.out.println("验证 A * A^(-1):");
            printMatrix(I);
            
        } catch (Exception e) {
            System.out.println("错误: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    /**
     * 测试矩阵伪逆
     */
    public static void testMatrixPseudoInverse() {
        System.out.println("=== 测试矩阵伪逆 ===");
        
        // 测试3x2矩阵（非方阵）
        float[][] data3x2 = {{1, 2}, {3, 4}, {5, 6}};
        IMatrix A = new RereMatrix(data3x2);
        
        System.out.println("原矩阵 A (3x2):");
        printMatrix(A);
        
        try {
            IMatrix pinvA = A.pinv();
            System.out.println("伪逆矩阵 A⁺ (2x3):");
            printMatrix(pinvA);
            
            // 验证 A * A⁺ * A = A
            IMatrix temp = ((RereMatrix)A).mmul(pinvA);
            IMatrix result = ((RereMatrix)temp).mmul(A);
            System.out.println("验证 A * A⁺ * A:");
            printMatrix(result);
            
        } catch (Exception e) {
            System.out.println("错误: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    /**
     * 测试奇异矩阵
     */
    public static void testSingularMatrix() {
        System.out.println("=== 测试奇异矩阵 ===");
        
        // 奇异矩阵（行列式为0）
        float[][] singularData = {{1, 2}, {2, 4}};
        IMatrix singular = new RereMatrix(singularData);
        
        System.out.println("奇异矩阵:");
        printMatrix(singular);
        
        try {
            IMatrix inv = singular.inv();
            System.out.println("逆矩阵（不应该执行到这里）:");
            printMatrix(inv);
        } catch (ArithmeticException e) {
            System.out.println("预期的错误: " + e.getMessage());
        }
        
        // 但是可以计算伪逆
        try {
            IMatrix pinv = singular.pinv();
            System.out.println("伪逆矩阵:");
            printMatrix(pinv);
        } catch (Exception e) {
            System.out.println("伪逆计算错误: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    /**
     * 测试非方阵
     */
    public static void testNonSquareMatrix() {
        System.out.println("=== 测试非方阵 ===");
        
        float[][] nonSquareData = {{1, 2, 3}, {4, 5, 6}};
        IMatrix nonSquare = new RereMatrix(nonSquareData);
        
        System.out.println("非方阵 (2x3):");
        printMatrix(nonSquare);
        
        try {
            IMatrix inv = nonSquare.inv();
            System.out.println("逆矩阵（不应该执行到这里）:");
            printMatrix(inv);
        } catch (IllegalArgumentException e) {
            System.out.println("预期的错误: " + e.getMessage());
        }
        
        // 计算伪逆
        try {
            IMatrix pinv = nonSquare.pinv();
            System.out.println("伪逆矩阵 (3x2):");
            printMatrix(pinv);
        } catch (Exception e) {
            System.out.println("伪逆计算错误: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    /**
     * 打印矩阵
     */
    public static void printMatrix(IMatrix matrix) {
        float[][] data = matrix.getData();
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                System.out.printf("%8.4f ", data[i][j]);
            }
            System.out.println();
        }
    }
} 