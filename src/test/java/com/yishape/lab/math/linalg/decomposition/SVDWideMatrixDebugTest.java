package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2;
import com.yishape.lab.util.Tuple3;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Debug test for wide matrix SVD reconstruction error.
 * This test investigates the root cause of the 0.339 error in wide matrices.
 */
public class SVDWideMatrixDebugTest {

    @Test
    public void debugWideMatrixSVD() {
        // Same seed as Python: 789
        java.util.Random rand = new java.util.Random(789);
        int m = 5, n = 10;
        double[][] data = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = rand.nextGaussian();
            }
        }

        System.out.println("=== Wide Matrix SVD Debug ===");
        System.out.println("Matrix shape: " + m + " x " + n);

        // Print matrix
        System.out.println("\nOriginal Matrix A:");
        printMatrix(data);

        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();

        System.out.println("\n=== Decomposition Results ===");
        System.out.println("U shape: " + U.rows() + " x " + U.cols());
        System.out.println("S length: " + S.length());
        System.out.println("VT shape: " + VT.rows() + " x " + VT.cols());

        System.out.println("\nSingular values:");
        for (int i = 0; i < S.length(); i++) {
            System.out.printf("  S[%d] = %.15e%n", i, S.get(i));
        }

        // Test 1: Using only first k rows of VT (thin reconstruction)
        System.out.println("\n=== Test 1: Thin Reconstruction U * diag(S) * VT[0:k, :] ===");
        int k = S.length();  // = m = 5
        IMatrix<Double> D = Linalg.zeros(k, k);
        for (int i = 0; i < k; i++) D.set(i, i, S.get(i));

        IMatrix<Double> UD = U.mmul(D);  // 5x5 * 5x5 = 5x5

        // Extract first k rows of VT
        IMatrix<Double> VTk = Linalg.zeros(k, n);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                VTk.set(i, j, VT.get(i, j));
            }
        }

        IMatrix<Double> reconThin = UD.mmul(VTk);  // 5x5 * 5x10 = 5x10
        double thinError = maxError(A, reconThin);
        System.out.printf("Thin reconstruction error: %.15e%n", thinError);

        // Test 2: Using full VT (dimension mismatch for wide matrices)
        System.out.println("\n=== Test 2: Full VT reconstruction (dimension mismatch for wide matrices) ===");
        System.out.printf("Full VT shape: %d x %d, UD shape: %d x %d%n", VT.rows(), VT.cols(), UD.rows(), UD.cols());
        try {
            IMatrix<Double> reconFull = UD.mmul(VT);
            double fullError = maxError(A, reconFull);
            System.out.printf("Full reconstruction error: %.15e%n", fullError);
        } catch (IllegalArgumentException e) {
            System.out.println("Expected: dimension mismatch — " + e.getMessage());
        }

        // Test 3: Check U orthogonality
        System.out.println("\n=== Test 3: U Orthogonality ===");
        IMatrix<Double> UtU = U.transposeNew().mmul(U);
        double maxUError = 0;
        for (int i = 0; i < UtU.rows(); i++) {
            for (int j = 0; j < UtU.cols(); j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double err = Math.abs(UtU.get(i, j) - expected);
                maxUError = Math.max(maxUError, err);
            }
        }
        System.out.printf("U orthogonality error: %.15e%n", maxUError);

        // Test 4: Check V orthogonality
        System.out.println("\n=== Test 4: V Orthogonality ===");
        IMatrix<Double> V = VT.transposeNew();
        IMatrix<Double> VtV = V.transposeNew().mmul(V);
        double maxVError = 0;
        for (int i = 0; i < VtV.rows(); i++) {
            for (int j = 0; j < VtV.cols(); j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double err = Math.abs(VtV.get(i, j) - expected);
                maxVError = Math.max(maxVError, err);
            }
        }
        System.out.printf("V orthogonality error: %.15e%n", maxVError);

        // Test 5: Check what VT looks like
        System.out.println("\n=== Test 5: VT Matrix Analysis ===");
        System.out.println("VT rows: " + VT.rows() + ", cols: " + VT.cols());

        // The issue: VT should be k×n for thin SVD, but ISVDDecomposition says VT is n×n
        // For wide matrices, the implementation might be doing something wrong

        // Let's check if only the first k rows of VT are meaningful
        System.out.println("\nFirst " + k + " rows of VT (should be orthonormal):");
        IMatrix<Double> VTkCheck = Linalg.zeros(k, n);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                VTkCheck.set(i, j, VT.get(i, j));
            }
        }
        IMatrix<Double> VTkVTk = VTkCheck.mmul(VTkCheck.transposeNew());
        double maxVTkError = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double err = Math.abs(VTkCheck.mmul(VTkCheck.transposeNew()).get(i, j) - expected);
                maxVTkError = Math.max(maxVTkError, err);
            }
        }
        System.out.printf("VT[k x n] orthogonality error: %.15e%n", maxVTkError);

        // Test 6: What should the reconstruction be?
        // For thin SVD: A = U_k * S_k * VT_k
        // where U_k is m×k, S_k is k×k diagonal, VT_k is k×n
        System.out.println("\n=== Test 6: Expected Thin SVD ===");
        System.out.println("Expected: A (m×n) = U (m×k) * S (k×k) * VT (k×n)");
        System.out.println("Where k = min(m,n) = " + k);

        // Check what the bidiagonalization actually produces
        System.out.println("\n=== Current Implementation ===");
        System.out.println("cachedV cols: " + V.cols() + ", expected n: " + n);
        System.out.println("cachedVT rows: " + VT.rows() + ", expected n: " + n);

        // The issue: cachedV should be n×k (thin), but the code returns n×n (full)
        // Then cachedVT = V.transposeNew() gives k×n, but the code expects n×n VT

        // Print reconstruction for debugging
        System.out.println("\n=== Reconstruction Analysis ===");
        System.out.println("UD shape (U*S): " + UD.rows() + " x " + UD.cols());
        System.out.println("VTk shape (first k rows of VT): " + VTk.rows() + " x " + VTk.cols());
        System.out.println("reconThin shape: " + reconThin.rows() + " x " + reconThin.cols());

        // Verify thin reconstruction is correct
        assertTrue(thinError < 1e-10, "Thin reconstruction should be accurate");

        // The issue is in how the test computes the reconstruction
        // The test should use the thin VT (first k rows), not full VT
    }

    private double maxError(IMatrix<Double> A, IMatrix<Double> B) {
        double maxErr = 0;
        for (int i = 0; i < A.rows(); i++) {
            for (int j = 0; j < A.cols(); j++) {
                maxErr = Math.max(maxErr, Math.abs(A.get(i, j) - B.get(i, j)));
            }
        }
        return maxErr;
    }

    private void printMatrix(double[][] data) {
        for (int i = 0; i < data.length; i++) {
            System.out.print("[");
            for (int j = 0; j < data[i].length; j++) {
                System.out.printf("%10.4f", data[i][j]);
                if (j < data[i].length - 1) System.out.print(", ");
            }
            System.out.println("]");
        }
    }
}
