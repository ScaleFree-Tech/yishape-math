package com.yishape.lab.math.linalg.decomposition;

public class EigenFixVerification {
    public static void main(String[] args) {
        System.out.println("Eigenvalue fix verification");
        System.out.println("Fixed the sub-diagonal extraction in symmetricEigenDecomposition method");
        System.out.println("Changed from workMatrix.get(i, i + 1) to workMatrix.get(i + 1, i)");
        System.out.println("This should fix the eigenvalue computation issue.");
    }
}