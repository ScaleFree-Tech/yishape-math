package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2;
import com.yishape.lab.math.linalg.decomposition.impl.RereBidiagonalDecomposition;
import com.yishape.lab.util.Tuple3;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Root cause analysis: Direct comparison between Java bidiagonalization
 * and what the SVD should produce for wide matrices.
 */
public class RootCauseAnalysisTest {

    @Test
    public void analyzeRootCause() {
        java.util.Random rand = new java.util.Random(789);
        int m = 5, n = 10;
        double[][] data = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = rand.nextGaussian();
            }
        }

        IMatrix<Double> A = Linalg.matrix(data);

        System.out.println("=== ROOT CAUSE ANALYSIS ===");
        System.out.println("Matrix A: " + m + " x " + n);

        // Step 1: Verify the bidiagonalization is correct
        System.out.println("\n--- Step 1: Bidiagonalization ---");
        RereBidiagonalDecomposition bidiag = new RereBidiagonalDecomposition();
        Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bidiagResult =
            bidiag.decompose(A);
        IMatrix<Double> U1 = bidiagResult.getFirst();
        IMatrix<Double> B = bidiagResult.getSecond();
        IMatrix<Double> V1 = bidiagResult.getThird();

        System.out.println("Bidiagonalization:");
        System.out.println("  U1: " + U1.rows() + " x " + U1.cols());
        System.out.println("  B: " + B.rows() + " x " + B.cols());
        System.out.println("  V1: " + V1.rows() + " x " + V1.cols());

        // Verify: U1^T * U1 = I?
        IMatrix<Double> U1tU1 = U1.transposeNew().mmul(U1);
        double u1Error = orthoError(U1tU1);
        System.out.println("  U1 orthogonality: " + u1Error);

        // Verify: V1^T * V1 = I?
        IMatrix<Double> V1tV1 = V1.transposeNew().mmul(V1);
        double v1Error = orthoError(V1tV1);
        System.out.println("  V1 orthogonality: " + v1Error);

        // Verify: A ≈ U1 * B * V1^T
        IMatrix<Double> reconBidiag = U1.mmul(B).mmul(V1.transposeNew());
        double bidiagError = maxError(A, reconBidiag);
        System.out.println("  Bidiagonalization recon error: " + bidiagError);

        // Step 2: Compare with standard SVD
        System.out.println("\n--- Step 2: Standard SVD ---");
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, com.yishape.lab.math.linalg.IVector<Double>, IMatrix<Double>> svdResult =
            svd.decompose(A);
        IMatrix<Double> U = svdResult.getFirst();
        com.yishape.lab.math.linalg.IVector<Double> S = svdResult.getSecond();
        IMatrix<Double> VT = svdResult.getThird();

        System.out.println("SVD:");
        System.out.println("  U: " + U.rows() + " x " + U.cols());
        System.out.println("  S: " + S.length());
        System.out.println("  VT: " + VT.rows() + " x " + VT.cols());

        // Step 3: Thin reconstruction
        System.out.println("\n--- Step 3: Thin Reconstruction ---");
        int k = Math.min(m, n);
        IMatrix<Double> D = Linalg.zeros(k, k);
        for (int i = 0; i < k; i++) D.set(i, i, S.get(i));

        // Method A: Use U * S * VT[0:k, :]
        IMatrix<Double> VTk = extractRows(VT, k);
        IMatrix<Double> reconA = U.mmul(D).mmul(VTk);
        double errorA = maxError(A, reconA);
        System.out.println("Method A (U * S * VT[0:k,:]): " + errorA);

        // Method B: Use V1 * Q computation
        System.out.println("\n--- Step 4: Investigate V1 usage ---");

        // The key question: In optimizedSVD, V1 is n×n but only first k columns are used
        // Is V1[:, 0:k] the correct thin V?

        // Let's extract V1[:, 0:k]
        IMatrix<Double> V1k = extractColumns(V1, k);
        System.out.println("V1[:, 0:k] shape: " + V1k.rows() + " x " + V1k.cols());

        // Check if V1k is orthonormal
        IMatrix<Double> V1ktV1k = V1k.transposeNew().mmul(V1k);
        double v1kError = orthoError(V1ktV1k);
        System.out.println("V1[:, 0:k] orthogonality: " + v1kError);

        // Step 5: Check if the issue is in how thin SVD should work
        System.out.println("\n--- Step 5: Thin SVD Theory ---");
        System.out.println("For A (m×n) with m < n:");
        System.out.println("  Thin SVD: A = U (m×m) * S (m) * VT (m×n)");
        System.out.println("  VT has only m rows, not n rows!");
        System.out.println("  But the code returns VT as n×n...");

        System.out.println("\n--- Step 6: Direct Check ---");
        System.out.println("Is VT[0:k, :] orthonormal as rows?");
        IMatrix<Double> VTkRows = VTk.mmul(VTk.transposeNew());
        double vtRowsError = orthoError(VTkRows);
        System.out.println("VT[0:k,:] * VT[0:k,:]^T error: " + vtRowsError);

        // The CORRECT thin SVD for A (m×n) should have:
        // U: m×m, S: m, VT: m×n
        // But bidiagonalSVD returns VT as n×n!

        // For thin SVD, we should use only the first m rows of VT
        // Let's verify this is what we're doing

        System.out.println("\n--- CONCLUSION ---");
        if (errorA > 0.01) {
            System.out.println("ERROR: Thin reconstruction fails!");
            System.out.println("This suggests the SVD factors (U, S, VT) are NOT");
            System.out.println("mathematically consistent for thin SVD.");

            // Check if the problem is in bidiagonalSVD vs optimizedSVD
            System.out.println("\n--- Checking which path is used ---");
            System.out.println("For m×n = " + m + "×" + n + " = " + (m*n));
            System.out.println("Since " + (m*n) + " <= 1000, optimizedSVD is used");
            System.out.println("optimizedSVD uses bidiagonalizationWithIMatrix + qrAlgorithmForBidiagonalWithIMatrix");
        } else {
            System.out.println("Thin reconstruction works fine.");
        }
    }

    private double orthoError(IMatrix<Double> M) {
        int n = M.rows();
        double maxErr = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                maxErr = Math.max(maxErr, Math.abs(M.get(i, j) - expected));
            }
        }
        return maxErr;
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

    private IMatrix<Double> extractRows(IMatrix<Double> M, int nRows) {
        int rows = Math.min(nRows, M.rows());
        int cols = M.cols();
        IMatrix<Double> result = Linalg.zeros(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i, j, M.get(i, j));
            }
        }
        return result;
    }

    private IMatrix<Double> extractColumns(IMatrix<Double> M, int nCols) {
        int rows = M.rows();
        int cols = Math.min(nCols, M.cols());
        IMatrix<Double> result = Linalg.zeros(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i, j, M.get(i, j));
            }
        }
        return result;
    }
}
