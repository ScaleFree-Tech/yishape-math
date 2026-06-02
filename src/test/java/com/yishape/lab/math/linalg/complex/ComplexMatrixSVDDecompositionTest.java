package com.yishape.lab.math.linalg.complex;

import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ComplexMatrixSVDDecompositionTest {

    @Test
    public void svd_diagonal_returnsDiagonalSingularValues() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{3, 0}, {0, 4}}, new double[2][2]
        );
        Tuple3<IComplexMatrix, IComplexMatrix.IComplexVector, IComplexMatrix> svd = a.svd();
        IComplexMatrix.IComplexVector S = svd._2;
        assertTrue(Math.abs(S.get(0).real - 4.0) < 1e-6 || Math.abs(S.get(0).real - 3.0) < 1e-6);
        assertTrue(Math.abs(S.get(1).real - 4.0) < 1e-6 || Math.abs(S.get(1).real - 3.0) < 1e-6);
    }

    @Test
    public void svd_identity_singularValuesAreOnes() {
        IComplexMatrix eye = IComplexMatrix.fromRealImag(
            new double[][]{{1, 0}, {0, 1}}, new double[2][2]
        );
        Tuple3<IComplexMatrix, IComplexMatrix.IComplexVector, IComplexMatrix> svd = eye.svd();
        IComplexMatrix.IComplexVector S = svd._2;
        assertEquals(1, S.get(0).real, 1e-6);
        assertEquals(1, S.get(1).real, 1e-6);
    }

    @Test
    public void svd_smallMatrix_hasExpectedDimensions() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}, {5, 6}}, new double[3][2]
        );
        Tuple3<IComplexMatrix, IComplexMatrix.IComplexVector, IComplexMatrix> svd = a.svd();
        assertEquals(3, svd._1.rows());
        assertEquals(2, svd._1.cols());
        assertEquals(2, svd._2.length());
        assertEquals(2, svd._3.rows());
        assertEquals(2, svd._3.cols());
    }

    @Test
    public void svd_singularValues_nonzero() {
        IComplexMatrix a = IComplexMatrix.fromRealImag(
            new double[][]{{1, 2}, {3, 4}}, new double[2][2]
        );
        Tuple3<IComplexMatrix, IComplexMatrix.IComplexVector, IComplexMatrix> svd = a.svd();
        IComplexMatrix.IComplexVector S = svd._2;
        assertTrue(S.get(0).real > 0);
        assertTrue(S.get(1).real > 0);
    }
}
