package com.yishape.lab.math.linalg;

import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas3;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class BoundaryConditionTest {

    // ==================== NaN propagation ====================

    @Test
    public void add_nanMatrix_producesNanResult() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{1, Double.NaN}, {3, 4}});
        IDoubleMatrix b = IDoubleMatrix.of(new double[][]{{5, 6}, {7, 8}});
        IDoubleMatrix c = a.add(b);
        assertTrue(Double.isNaN(c.get(0, 1)));
    }

    @Test
    public void sub_nanMatrix_producesNanResult() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{1, 2}, {Double.NaN, 4}});
        IDoubleMatrix b = IDoubleMatrix.of(new double[][]{{5, 6}, {7, 8}});
        IDoubleMatrix c = a.sub(b);
        assertTrue(Double.isNaN(c.get(1, 0)));
    }

    @Test
    public void multiply_nanElement_propagates() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{Double.NaN, 2}, {3, 4}});
        IDoubleMatrix b = IDoubleMatrix.of(new double[][]{{1, 0}, {0, 1}});
        IDoubleMatrix c = a.multiply(b);
        // NaN * 1 + 2 * 0 = NaN at c(0,0)
        assertTrue(Double.isNaN(c.get(0, 0)));
    }

    @Test
    public void sum_nanElements_returnsNaN() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{1, 2}, {Double.NaN, 4}});
        assertTrue(Double.isNaN(a.sumValue()));
    }

    @Test
    public void mean_nanElements_returnsNaN() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{1, 2}, {3, Double.NaN}});
        assertTrue(Double.isNaN(a.meanValue()));
    }

    // ==================== Inf handling ====================

    @Test
    public void add_infElements_propagatesInf() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{1, Double.POSITIVE_INFINITY}, {3, 4}});
        IDoubleMatrix b = IDoubleMatrix.of(new double[][]{{5, 6}, {7, 8}});
        IDoubleMatrix c = a.add(b);
        assertTrue(Double.isInfinite(c.get(0, 1)));
    }

    @Test
    public void multiplyByScalar_inf_preservesInf() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{Double.POSITIVE_INFINITY, 2}, {3, 4}});
        IDoubleMatrix b = a.multiplyByScalar(2.0);
        assertTrue(Double.isInfinite(b.get(0, 0)));
    }

    // ==================== Zero matrix edge cases ====================

    @Test
    public void inv_zeroMatrix_producesInfOrThrows() {
        IDoubleMatrix a = IDoubleMatrix.zeros(2, 2);
        try {
            IDoubleMatrix ai = a.inv();
            // if it doesn't throw, the result should contain infinities
            assertTrue(Double.isInfinite(ai.get(0, 0)) || Double.isNaN(ai.get(0, 0)));
        } catch (Exception e) {
            // also acceptable
        }
    }

    @Test
    public void solve_zeroRhs_givesZeroSolution() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{2, 0}, {0, 3}});
        IDoubleMatrix b = IDoubleMatrix.zeros(2, 1);
        IDoubleMatrix x = a.solve(b);
        assertEquals(0, x.get(0, 0), 1e-10);
        assertEquals(0, x.get(1, 0), 1e-10);
    }

    @Test
    public void det_zeroMatrix_isZero() {
        IDoubleMatrix a = IDoubleMatrix.zeros(3, 3);
        assertEquals(0, a.det(), 1e-10);
    }

    @Test
    public void det_identity_isOne() {
        IDoubleMatrix a = IDoubleMatrix.eye(3);
        assertEquals(1, a.det(), 1e-10);
    }

    // ==================== Singular matrix edge cases ====================

    @Test
    public void solve_singularMatrix_throwsException() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{1, 2}, {2, 4}});
        IDoubleMatrix b = IDoubleMatrix.of(new double[][]{{1}, {1}});
        assertThrows(Exception.class, () -> a.solve(b));
    }

    @Test
    public void inv_singularMatrix_throwsException() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{1, 1}, {1, 1}});
        assertThrows(Exception.class, a::inv);
    }

    // ==================== Single element matrix ====================

    @Test
    public void det_1x1_returnsElementValue() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{42}});
        assertEquals(42, a.det(), 1e-10);
    }

    @Test
    public void inv_1x1_returnsReciprocal() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{4}});
        IDoubleMatrix ai = a.inv();
        assertEquals(0.25, ai.get(0, 0), 1e-10);
    }

    // ==================== Large value handling ====================

    @Test
    public void multiply_largeValues_noOverflow() {
        double big = 1e100;
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{big, 0}, {0, big}});
        IDoubleMatrix b = IDoubleMatrix.of(new double[][]{{2, 0}, {0, 2}});
        IDoubleMatrix c = a.multiply(b);
        assertTrue(Double.isFinite(c.get(0, 0)));
        assertEquals(2 * big, c.get(0, 0), 1e90);
    }

    @Test
    public void multiply_smallValues_noUnderflowToZero() {
        double tiny = 1e-100;
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{tiny, 0}, {0, tiny}});
        IDoubleMatrix b = IDoubleMatrix.of(new double[][]{{2, 0}, {0, 2}});
        IDoubleMatrix c = a.multiply(b);
        assertTrue(c.get(0, 0) > 0);
    }

    // ==================== Non-square matrices ====================

    @Test
    public void det_nonSquare_throwsException() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{1, 2, 3}, {4, 5, 6}});
        assertThrows(IllegalArgumentException.class, a::det);
    }

    @Test
    public void inv_nonSquare_throwsException() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{1, 2, 3}, {4, 5, 6}});
        assertThrows(IllegalArgumentException.class, a::inv);
    }

    // ==================== Float matrix NaN/Inf ====================

    @Test
    public void floatAdd_nanMatrix_producesNanResult() {
        IFloatMatrix a = IFloatMatrix.of(new float[][]{{1, Float.NaN}, {3, 4}});
        IFloatMatrix b = IFloatMatrix.of(new float[][]{{5, 6}, {7, 8}});
        IFloatMatrix c = a.add(b);
        assertTrue(Double.isNaN(c.get(0, 1)), "NaN should propagate");
    }

    @Test
    public void floatDet_zeroMatrix_isZero() {
        IFloatMatrix a = IFloatMatrix.zeros(3, 3);
        assertEquals(0.0, a.det(), 1e-6);
    }

    // ==================== Decomposition edge cases ====================

    @Test
    public void svd_zeroMatrix_singularValuesAreZero() {
        IDoubleMatrix a = IDoubleMatrix.zeros(3, 3);
        RereSVDDecompBlas3 svd = new RereSVDDecompBlas3();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(a);
        IVector<Double> sv = result._2;
        for (int i = 0; i < sv.length(); i++) {
            assertEquals(0, sv.get(i), 1e-10);
        }
    }

    @Test
    public void eigen_zeroMatrix_eigenvaluesAreZero() {
        IDoubleMatrix a = IDoubleMatrix.zeros(3, 3);
        RereEigenDecomposition eig = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eig.decompose(a);
        IVector<Double> eigenvals = result._1;
        // eigenvalues are stored as [re0, im0, re1, im1, ...]
        for (int i = 0; i < eigenvals.length(); i++) {
            assertEquals(0, eigenvals.get(i), 1e-10);
        }
    }

    @Test
    public void svd_diagonal_singularValuesMatchAbs() {
        IDoubleMatrix a = IDoubleMatrix.of(new double[][]{{3, 0}, {0, -4}});
        RereSVDDecompBlas3 svd = new RereSVDDecompBlas3();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(a);
        IVector<Double> sv = result._2;
        boolean has3 = false, has4 = false;
        for (int i = 0; i < sv.length(); i++) {
            if (Math.abs(sv.get(i) - 3) < 1e-10) has3 = true;
            if (Math.abs(sv.get(i) - 4) < 1e-10) has4 = true;
        }
        assertTrue(has3);
        assertTrue(has4);
    }
}
