package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2;
import com.yishape.lab.math.linalg.decomposition.impl.RereBidiagonalDecomposition;
import com.yishape.lab.util.Tuple3;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Verify that wide matrix (m &lt; n) SVD produces correct thin reconstruction.
 *
 * The fix: (1) optimizedSVD / divideAndConquerSVD now transpose wide matrices before
 * bidiagonalization, avoiding loss of the last superdiagonal element; (2) widenVToFullOrthogonal
 * preserves the right singular vectors by keeping the first k columns unchanged and
 * orthogonalizing only the null-space completion columns via MGS.
 */
public class TransposeBugTest {

    private static final double TOL = 1e-10;

    @Test
    public void verifyTransposeBug() {
        java.util.Random rand = new java.util.Random(789);
        int m = 5, n = 10;
        double[][] data = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = rand.nextGaussian();
            }
        }

        IMatrix<Double> A = Linalg.matrix(data);

        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, com.yishape.lab.math.linalg.IVector<Double>, IMatrix<Double>> result =
            svd.decompose(A);
        IMatrix<Double> U = result.getFirst();
        com.yishape.lab.math.linalg.IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();

        // Verify shapes
        assertEquals(m, U.rows());
        assertEquals(m, U.cols());
        assertEquals(m, S.length());
        assertEquals(n, VT.rows(), "VT should be n×n after widening");
        assertEquals(n, VT.cols(), "VT should be n×n after widening");

        // Thin reconstruction: A = U * S * VT[0:k, :]
        int k = Math.min(m, n);
        IMatrix<Double> D = Linalg.zeros(k, k);
        for (int i = 0; i < k; i++) D.set(i, i, S.get(i));
        IMatrix<Double> US = U.mmul(D);

        IMatrix<Double> VTk = Linalg.zeros(k, n);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                VTk.set(i, j, VT.get(i, j));
            }
        }
        IMatrix<Double> recon = US.mmul(VTk);

        double maxError = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                maxError = Math.max(maxError, Math.abs(A.get(i, j) - recon.get(i, j)));
            }
        }
        assertTrue(maxError < TOL, "Wide matrix thin reconstruction error: " + maxError
            + " (should be near machine precision after fix)");
    }
}
