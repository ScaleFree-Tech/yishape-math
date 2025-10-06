package com.yishape.lab.math.linalg.decomposition;

public class EigenValueManualCalculation {
    public static void main(String[] args) {
        // Test matrix
        double[][] A = {
            { 4.0, 2.0, 1.0},
            { 2.0, 3.0, 1.0},
            { 1.0, 1.0, 2.0}
        };
        
        System.out.println("Matrix A:");
        printMatrix(A);
        
        // For a 3x3 matrix, the characteristic polynomial is:
        // det(A - λI) = 0
        // This expands to: -λ³ + tr(A)λ² + (sum of principal minors)λ - det(A) = 0
        
        // Calculate trace (sum of diagonal elements)
        double trace = A[0][0] + A[1][1] + A[2][2];
        System.out.println("Trace: " + trace);
        
        // Calculate determinant
        double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                   - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                   + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        System.out.println("Determinant: " + det);
        
        // Calculate sum of principal minors (2x2 determinants)
        double sumMinors = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) +
                          (A[0][0] * A[2][2] - A[0][2] * A[2][0]) +
                          (A[1][1] * A[2][2] - A[1][2] * A[2][1]);
        System.out.println("Sum of principal minors: " + sumMinors);
        
        // Characteristic polynomial: -λ³ + trace*λ² + sumMinors*λ - det = 0
        System.out.println("Characteristic polynomial: -λ³ + " + trace + "*λ² + " + sumMinors + "*λ - " + det + " = 0");
        
        // Let's also compute a few test values to see the behavior
        System.out.println("\nTesting some values in the polynomial f(λ) = -λ³ + " + trace + "*λ² + " + sumMinors + "*λ - " + det);
        for (double lambda = -1; lambda <= 7; lambda += 0.5) {
            double result = -lambda*lambda*lambda + trace*lambda*lambda + sumMinors*lambda - det;
            System.out.printf("f(%.1f) = %.2f\n", lambda, result);
        }
    }
    
    private static void printMatrix(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.printf("%.2f ", matrix[i][j]);
            }
            System.out.println();
        }
    }
}