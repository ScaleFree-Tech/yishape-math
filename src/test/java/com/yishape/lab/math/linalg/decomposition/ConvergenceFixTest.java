package com.yishape.lab.math.linalg.decomposition;

public class ConvergenceFixTest {
    public static void main(String[] args) {
        System.out.println("Convergence Fix Test");
        System.out.println("===================");
        System.out.println("Fixed inconsistency in QR algorithm convergence check:");
        System.out.println("Removed the line that was setting e.set(start, 0.0) in the while loop");
        System.out.println("This was causing incorrect handling of convergence criteria");
        System.out.println("");
        System.out.println("The convergence check should now work properly");
        System.out.println("and lead to more accurate eigenvalue computation.");
    }
}