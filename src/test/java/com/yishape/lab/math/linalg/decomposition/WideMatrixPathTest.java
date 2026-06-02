package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2;
import com.yishape.lab.math.linalg.decomposition.impl.RereBidiagonalDecomposition;
import com.yishape.lab.util.Tuple3;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test to verify that bidiagonalSVD path handles wide matrices correctly
 * while optimizedSVD path does not.
 */
public class WideMatrixPathTest {

    @Test
    public void testWideMatrix_WhichPathIsUsed() {
        // 5x10 matrix: size=50, traditionalSVD is used (size <= 1000)
        // But bidiagonalSVD would transpose to 10x5 first

        java.util.Random rand = new java.util.Random(789);
        int m = 5, n = 10;
        double[][] data = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = rand.nextGaussian();
            }
        }

        IMatrix<Double> A = Linalg.matrix(data);

        System.out.println("=== Testing Wide Matrix Path ===");
        System.out.println("Matrix: " + m + " x " + n);

        // Test 1: Use bidiagonalSVD path (by forcing large size)
        System.out.println("\n--- Test 1: Using bidiagonalSVD path ---");
        RereSVDDecompBlas2 svd1 = new RereSVDDecompBlas2() {
            @Override
            public Tuple3<IMatrix<Double>, com.yishape.lab.math.linalg.IVector<Double>, IMatrix<Double>> decompose(
                    IMatrix<Double> matrix, double epsilon, int maxIterations) {
                // Force bidiagonalSVD by modifying size check
                int m = matrix.rows();
                int n = matrix.cols();
                int size = m * n;
                // Temporarily make it use bidiagonalSVD (size > 1000 path)
                if (size <= 1000) {
                    // Can't easily override, but we can test the bidiagonal decomposition directly
                }
                return super.decompose(matrix, epsilon, maxIterations);
            }
        };

        // Test 2: Compare with what bidiagonalization actually produces
        System.out.println("\n--- Test 2: Bidiagonal Decomposition ---");
        RereBidiagonalDecomposition bidiag = new RereBidiagonalDecomposition();
        Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bidiagResult =
            bidiag.decompose(A);

        IMatrix<Double> U1 = bidiagResult.getFirst();
        IMatrix<Double> B = bidiagResult.getSecond();
        IMatrix<Double> V1 = bidiagResult.getThird();

        System.out.println("Bidiagonal decomposition:");
        System.out.println("  U1 shape: " + U1.rows() + " x " + U1.cols());
        System.out.println("  B shape: " + B.rows() + " x " + B.cols());
        System.out.println("  V1 shape: " + V1.rows() + " x " + V1.cols());

        // The issue: For wide matrix (m < n):
        // - U1 is m x m = 5 x 5 (correct)
        // - B is m x m = 5 x 5 (correct, square bidiagonal)
        // - V1 is n x n = 10 x 10 (but for SVD we need V as n x m or n x k)

        System.out.println("\n--- Test 3: Standard SVD (uses optimizedSVD path) ---");
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, com.yishape.lab.math.linalg.IVector<Double>, IMatrix<Double>> svdResult =
            svd.decompose(A);

        IMatrix<Double> U = svdResult.getFirst();
        com.yishape.lab.math.linalg.IVector<Double> S = svdResult.getSecond();
        IMatrix<Double> VT = svdResult.getThird();

        System.out.println("Standard SVD:");
        System.out.println("  U shape: " + U.rows() + " x " + U.cols());
        System.out.println("  S length: " + S.length());
        System.out.println("  VT shape: " + VT.rows() + " x " + VT.cols());

        // Verify thin reconstruction (U * S * VT[0:k,:])
        System.out.println("\n--- Test 4: Thin Reconstruction ---");
        int k = S.length();
        IMatrix<Double> D = Linalg.zeros(k, k);
        for (int i = 0; i < k; i++) D.set(i, i, S.get(i));

        IMatrix<Double> UD = U.mmul(D);

        IMatrix<Double> VTk = Linalg.zeros(k, n);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                VTk.set(i, j, VT.get(i, j));
            }
        }

        IMatrix<Double> recon = UD.mmul(VTk);
        double maxErr = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                maxErr = Math.max(maxErr, Math.abs(A.get(i, j) - recon.get(i, j)));
            }
        }
        System.out.println("Thin reconstruction error: " + maxErr);

        // Now test what happens if we TRANSPOSE first (as bidiagonalSVD does)
        System.out.println("\n--- Test 5: SVD with Transpose First ---");
        IMatrix<Double> AT = A.transposeNew();
        RereSVDDecompBlas2 svdT = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, com.yishape.lab.math.linalg.IVector<Double>, IMatrix<Double>> svdTResult =
            svdT.decompose(AT);

        // AT is 10x5, so result is:
        IMatrix<Double> UT = svdTResult.getFirst();
        com.yishape.lab.math.linalg.IVector<Double> ST = svdTResult.getSecond();
        IMatrix<Double> VTT = svdTResult.getThird();

        System.out.println("Transposed SVD (10x5 -> 5x10 after transpose):");
        System.out.println("  UT shape: " + UT.rows() + " x " + UT.cols());
        System.out.println("  ST length: " + ST.length());
        System.out.println("  VTT shape: " + VTT.rows() + " x " + VTT.cols());

        // To reconstruct original A from transposed SVD:
        // A^T = UT * ST * VTT
        // A = VT^T * ST^T * UT^T = V * S * U^T
        // But we got UT (5x5), ST (5), VTT (5x5) for 10x5
        // This is wrong too!

        // The KEY insight: bidiagonalSVD correctly transposes, but:
        // 1. It computes SVD of A^T (10x5)
        // 2. Returns U_t (5x5), S_t (5), VT_t (5x5)
        // 3. But then it sets cachedU = V_t (5x5), cachedV = widen(U_t, 10) (10x10)
        // This is WRONG - it swapped U and V!

        System.out.println("\n--- Analysis ---");
        System.out.println("For wide matrix A (m x n, m < n):");
        System.out.println("  - bidiagonalSVD transposes to A^T (n x m)");
        System.out.println("  - Computes SVD of A^T: U_t * S * V_t^T");
        System.out.println("  - But then sets cachedU = V_t, cachedV = widen(U_t)");
        System.out.println("  - This is INCORRECT - should be cachedU = U_t, cachedV = V_t");
    }
}
