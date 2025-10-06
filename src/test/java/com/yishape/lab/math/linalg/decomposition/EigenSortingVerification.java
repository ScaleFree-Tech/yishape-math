package com.yishape.lab.math.linalg.decomposition;

public class EigenSortingVerification {
    public static void main(String[] args) {
        // Test the sorting logic with mixed positive and negative values
        double[] eigenvalues = {2.0, -1.0, 5.0, -3.0, 0.5};
        double[][] eigenvectors = new double[5][3]; // Just dummy data for testing
        
        // Initialize dummy eigenvectors
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 3; j++) {
                eigenvectors[i][j] = i * 10 + j;
            }
        }
        
        System.out.println("Before sorting:");
        System.out.print("Eigenvalues: ");
        for (double val : eigenvalues) {
            System.out.print(val + " ");
        }
        System.out.println();
        
        // Apply the bubble sort logic from the RereEigenDecomposition class
        bubbleSortEigenvaluesAndVectors(eigenvalues, eigenvectors);
        
        System.out.println("After sorting (should be descending):");
        System.out.print("Eigenvalues: ");
        for (double val : eigenvalues) {
            System.out.print(val + " ");
        }
        System.out.println();
        
        // Also test the Arrays.sort approach
        double[] eigenvalues2 = {2.0, -1.0, 5.0, -3.0, 0.5};
        System.out.println("\nUsing Arrays.sort approach:");
        System.out.print("Before: ");
        for (double val : eigenvalues2) {
            System.out.print(val + " ");
        }
        System.out.println();
        
        Integer[] indices = new Integer[eigenvalues2.length];
        for (int i = 0; i < eigenvalues2.length; i++) {
            indices[i] = i;
        }
        
        // Sort indices by eigenvalue (descending)
        java.util.Arrays.sort(indices, (i, j) -> Double.compare(eigenvalues2[j], eigenvalues2[i]));
        
        System.out.print("Sorted indices: ");
        for (int idx : indices) {
            System.out.print(idx + " ");
        }
        System.out.println();
        
        System.out.print("Eigenvalues (should be descending): ");
        for (int idx : indices) {
            System.out.print(eigenvalues2[idx] + " ");
        }
        System.out.println();
    }
    
    // Copy of the bubble sort implementation from RereEigenDecomposition
    private static void bubbleSortEigenvaluesAndVectors(double[] eigenvalues, double[][] eigenvectors) {
        int n = eigenvalues.length;

        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - 1 - i; j++) {
                // Compare adjacent eigenvalues - this should sort in descending order
                if (eigenvalues[j] < eigenvalues[j + 1]) {
                    // Swap eigenvalues
                    double tempVal = eigenvalues[j];
                    eigenvalues[j] = eigenvalues[j + 1];
                    eigenvalues[j + 1] = tempVal;

                    // Swap corresponding eigenvectors
                    double[] tempVec = eigenvectors[j];
                    eigenvectors[j] = eigenvectors[j + 1];
                    eigenvectors[j + 1] = tempVec;
                }
            }
        }
    }
}