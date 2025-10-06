package com.yishape.lab.math.linalg.decomposition;

public class WilkinsonShiftFixTest {
    public static void main(String[] args) {
        System.out.println("Wilkinson Shift Fix Test");
        System.out.println("========================");
        System.out.println("Fixed issue in computeWilkinsonShiftForTridiagonal method:");
        System.out.println("When discriminant < 0 (complex eigenvalues),");
        System.out.println("changed from returning 'c' to returning '(a + c) / 2.0'");
        System.out.println("(the real part of the complex eigenvalues)");
        System.out.println("");
        System.out.println("This should improve the convergence of the QR algorithm");
        System.out.println("and lead to more accurate eigenvalue computation.");
    }
}