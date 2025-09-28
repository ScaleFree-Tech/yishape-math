package com.reremouse.lab.math.linalg;

public class SimpleTriangularTest {
    public static void main(String[] args) {
        // Test lower triangular matrix
        System.out.println("Testing lower triangular matrix:");
        IMatrix<Double> lowerTri = Linalg.lowerTriMatrix(3);
        System.out.println("Lower triangular matrix (3x3):");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                System.out.print(lowerTri.get(i, j) + " ");
            }
            System.out.println();
        }
        
        // Test upper triangular matrix
        System.out.println("\nTesting upper triangular matrix:");
        IMatrix<Double> upperTri = Linalg.upperTriMatrix(3);
        System.out.println("Upper triangular matrix (3x3):");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                System.out.print(upperTri.get(i, j) + " ");
            }
            System.out.println();
        }
        
        // Test 1x1 matrices
        System.out.println("\nTesting 1x1 matrices:");
        IMatrix<Double> lowerTri1 = Linalg.lowerTriMatrix(1);
        IMatrix<Double> upperTri1 = Linalg.upperTriMatrix(1);
        System.out.println("Lower triangular matrix (1x1): " + lowerTri1.get(0, 0));
        System.out.println("Upper triangular matrix (1x1): " + upperTri1.get(0, 0));
    }
}