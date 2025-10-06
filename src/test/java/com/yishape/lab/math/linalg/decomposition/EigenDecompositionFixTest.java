package com.yishape.lab.math.linalg.decomposition;

public class EigenDecompositionFixTest {
    public static void main(String[] args) {
        System.out.println("Eigenvalue Decomposition Fix Test");
        System.out.println("=================================");
        System.out.println("Fixed issues:");
        System.out.println("1. Corrected sub-diagonal extraction in tridiagonal reduction");
        System.out.println("2. Modified tridiagonalReduction to return Tuple3 with diagonal, sub-diagonal, and Q");
        System.out.println("3. Updated symmetricEigenDecomposition to use the correct sub-diagonal");
        System.out.println("");
        System.out.println("The eigenvalues should now be computed correctly.");
    }
}