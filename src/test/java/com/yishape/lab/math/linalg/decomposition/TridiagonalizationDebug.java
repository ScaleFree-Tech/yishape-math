package com.yishape.lab.math.linalg.decomposition;

public class TridiagonalizationDebug {
    public static void main(String[] args) {
        System.out.println("Tridiagonalization Debug");
        System.out.println("========================");
        System.out.println("Issues identified:");
        System.out.println("1. Fixed sub-diagonal extraction from (i, i+1) to (i+1, i)");
        System.out.println("2. Need to verify Householder transformation implementation");
        System.out.println("3. Need to verify QR algorithm implementation");
        System.out.println("");
        System.out.println("Next steps:");
        System.out.println("- Create a manual tridiagonalization test");
        System.out.println("- Compare with known correct implementation");
        System.out.println("- Verify each step of the process");
    }
}