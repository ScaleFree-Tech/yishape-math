package com.yishape.lab.math.linalg.sparse;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SparseDenseMultiplyTest {

    @Test
    public void multiplyDense_sparseTimesDenseMatrix_matchesDenseMultiply() {
        double[][] denseData = {{1, 2}, {3, 4}, {5, 6}};
        ISparseMatrix sparse = ISparseMatrix.fromDense(new double[][]{{1, 0, 0}, {0, 2, 0}});
        IMatrix<Double> dense = Linalg.matrix(denseData);

        IMatrix<Double> result = sparse.multiplyDense(dense);

        assertEquals(2, result.nrow());
        assertEquals(2, result.ncol());
        assertEquals(1.0, result.get(0, 0), 1e-10);
        assertEquals(2.0, result.get(0, 1), 1e-10);
        assertEquals(6.0, result.get(1, 0), 1e-10);
        assertEquals(8.0, result.get(1, 1), 1e-10);
    }

    @Test
    public void multiplyDense_dimensionMismatch_throwsException() {
        ISparseMatrix sparse = ISparseMatrix.eye(3);
        IMatrix<Double> wrong = Linalg.matrix(new double[2][3]);
        assertThrows(IllegalArgumentException.class, () -> sparse.multiplyDense(wrong));
    }

    @Test
    public void multiplyDense_identityTimesDense_equalsOriginal() {
        ISparseMatrix eye = ISparseMatrix.eye(3);
        double[][] d = {{1, 2}, {3, 4}, {5, 6}};
        IMatrix<Double> dense = Linalg.matrix(d);
        IMatrix<Double> result = eye.multiplyDense(dense);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(d[i][j], result.get(i, j), 1e-10);
            }
        }
    }

    @Test
    public void multiplyDenseFromLeft_denseTimesSparse_matchesReference() {
        double[][] denseData = {{1, 2}, {3, 4}};
        ISparseMatrix sparse = ISparseMatrix.fromDense(new double[][]{{1, 0}, {0, 2}});
        IMatrix<Double> dense = Linalg.matrix(denseData);

        IMatrix<Double> result = sparse.multiplyDenseFromLeft(dense);

        assertEquals(2, result.nrow());
        assertEquals(2, result.ncol());
        assertEquals(1.0, result.get(0, 0), 1e-10);
        assertEquals(4.0, result.get(0, 1), 1e-10);
        assertEquals(3.0, result.get(1, 0), 1e-10);
        assertEquals(8.0, result.get(1, 1), 1e-10);
    }

    @Test
    public void denseMultiplySparse_viaIDoubleMatrix_matchesDense() {
        IDoubleMatrix dense = IDoubleMatrix.of(new double[][]{{1, 2}, {3, 4}});
        ISparseMatrix sparse = ISparseMatrix.fromDense(new double[][]{{1, 0}, {0, 2}});

        IDoubleMatrix result = dense.multiply(sparse);

        assertEquals(1.0, result.get(0, 0), 1e-10);
        assertEquals(4.0, result.get(0, 1), 1e-10);
        assertEquals(3.0, result.get(1, 0), 1e-10);
        assertEquals(8.0, result.get(1, 1), 1e-10);
    }

    @Test
    public void multiply_sparseVector_identityTimesVector_isIdentity() {
        ISparseMatrix eye = ISparseMatrix.eye(4);
        IVector<Double> v = Linalg.vector(new double[]{1, 2, 3, 4});
        IVector<Double> r = eye.multiply(v);
        for (int i = 0; i < 4; i++) {
            assertEquals(v.get(i), r.get(i), 1e-10);
        }
    }
}
