package com.yishape.lab.math.linalg.sparse;

import com.yishape.lab.math.linalg.IMatrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SparseFormatConversionTest {

    @Test
    public void fromDense_smallMatrix_preservesNonzeros() {
        double[][] data = {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
        ISparseMatrix s = ISparseMatrix.fromDense(data);
        assertEquals(3, s.nnz());
        assertEquals(1.0, s.get(0, 0));
        assertEquals(2.0, s.get(1, 1));
        assertEquals(3.0, s.get(2, 2));
        assertEquals(0.0, s.get(0, 1));
    }

    @Test
    public void fromDense_withTolerance_excludesBelowThreshold() {
        double[][] data = {{1, 1e-12, 0.01}};
        ISparseMatrix s = ISparseMatrix.fromDense(data, 1e-5);
        assertEquals(2, s.nnz());
        assertEquals(1.0, s.get(0, 0));
        assertEquals(0.01, s.get(0, 2));
        assertEquals(0.0, s.get(0, 1));
    }

    @Test
    public void roundTrip_cooToCsrToCsc_preservesValues() {
        double[][] data = {{0, 2, 0}, {3, 0, 0}, {0, 0, 4}};
        ISparseMatrix original = ISparseMatrix.fromDense(data);
        ISparseMatrix csr = original.toFormat(SparseFormat.CSR);
        ISparseMatrix csc = csr.toFormat(SparseFormat.CSC);
        ISparseMatrix backToCoo = csc.toFormat(SparseFormat.COO);

        assertEquals(3, backToCoo.nnz());
        assertEquals(2.0, backToCoo.get(0, 1));
        assertEquals(3.0, backToCoo.get(1, 0));
        assertEquals(4.0, backToCoo.get(2, 2));
    }

    @Test
    public void toDense_roundTrip_preservesAllValues() {
        double[][] data = {{1, 0, 0}, {0, 5, 0}, {0, 0, 9}};
        ISparseMatrix s = ISparseMatrix.fromDense(data);
        IMatrix<Double> dense = s.toDense();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(data[i][j], dense.get(i, j), 1e-10);
            }
        }
    }

    @Test
    public void fromCSR_nnzMatrix_hasCorrectShape() {
        int[] rowPtr = {0, 2, 3, 4};
        int[] colInd = {0, 1, 0, 2};
        double[] values = {1, 2, 3, 4};
        ISparseMatrix s = ISparseMatrix.fromCSR(rowPtr, colInd, values, 3, 3);
        assertEquals(3, s.rows());
        assertEquals(3, s.cols());
        assertEquals(4, s.nnz());
    }

    @Test
    public void eye_identityMatrix_producesCorrectValues() {
        ISparseMatrix eye = ISparseMatrix.eye(4);
        assertEquals(4, eye.rows());
        assertEquals(4, eye.cols());
        assertEquals(4, eye.nnz());
        for (int i = 0; i < 4; i++) {
            assertEquals(1.0, eye.get(i, i));
            for (int j = 0; j < 4; j++) {
                if (i != j) assertEquals(0.0, eye.get(i, j));
            }
        }
    }

    @Test
    public void diag_diagonalMatrix_hasCorrectDiagonal() {
        ISparseMatrix d = ISparseMatrix.diag(new double[]{2, 3, 4});
        assertEquals(2.0, d.get(0, 0));
        assertEquals(3.0, d.get(1, 1));
        assertEquals(4.0, d.get(2, 2));
        assertEquals(0.0, d.get(0, 1));
    }
}
