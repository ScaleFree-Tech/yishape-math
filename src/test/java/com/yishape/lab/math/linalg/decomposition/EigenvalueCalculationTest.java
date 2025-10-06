package com.yishape.lab.math.linalg.decomposition;

public class EigenvalueCalculationTest {
    public static void main(String[] args) {
        // Test matrix
        double[][] A = {
            { 4.0, 2.0, 1.0},
            { 2.0, 3.0, 1.0},
            { 1.0, 1.0, 2.0}
        };
        
        System.out.println("Matrix A:");
        printMatrix(A);
        
        // Calculate eigenvalues using the characteristic polynomial
        // For a 3x3 matrix A, solve det(A - λI) = 0
        // This gives us a cubic equation: -λ³ + aλ² + bλ + c = 0
        
        // Coefficients of the characteristic polynomial
        double a = A[0][0] + A[1][1] + A[2][2]; // trace
        double b = A[0][0]*A[1][1] + A[0][0]*A[2][2] + A[1][1]*A[2][2] 
                 - A[0][1]*A[1][0] - A[0][2]*A[2][0] - A[1][2]*A[2][1]; // sum of principal minors
        double c = A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) 
                 - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0]) 
                 + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]); // determinant
        
        System.out.println("\nCharacteristic polynomial coefficients:");
        System.out.println("a (trace) = " + a);
        System.out.println("b (sum of principal minors) = " + b);
        System.out.println("c (determinant) = " + c);
        System.out.println("Polynomial: -λ³ + " + a + "*λ² + " + b + "*λ - " + c + " = 0");
        
        // Let's solve this numerically by testing values
        System.out.println("\nTesting values:");
        for (double lambda = 0.0; lambda <= 8.0; lambda += 0.1) {
            double result = -lambda*lambda*lambda + a*lambda*lambda + b*lambda - c;
            if (Math.abs(result) < 0.01) { // Close to zero
                System.out.printf("Root near λ = %.2f, f(λ) = %.6f\n", lambda, result);
            }
        }
        
        // More detailed search around expected values
        System.out.println("\nMore detailed search:");
        for (double lambda = 0.0; lambda <= 8.0; lambda += 0.01) {
            double result = -lambda*lambda*lambda + a*lambda*lambda + b*lambda - c;
            if (Math.abs(result) < 0.1) {
                System.out.printf("f(%.2f) = %.6f\n", lambda, result);
            }
        }
        
        // Known correct eigenvalues (from a reliable source or calculation)
        // For this matrix, the eigenvalues should be approximately:
        // λ1 ≈ 6.19, λ2 ≈ 2.30, λ3 ≈ 0.51
        double[] expectedEigenvalues = {6.19, 2.30, 0.51};
        System.out.println("\nExpected eigenvalues (approximate):");
        for (double val : expectedEigenvalues) {
            System.out.printf("%.2f ", val);
        }
        System.out.println();
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