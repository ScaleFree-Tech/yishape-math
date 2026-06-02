package com.yishape.lab.math.linalg.sparse;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SparseBoundaryTest {

    @Test
    public void fromDense_allZeros_hasZeroNnz() {
        double[][] data = {{0, 0}, {0, 0}};
        ISparseMatrix s = ISparseMatrix.fromDense(data);
        assertEquals(0, s.nnz());
        assertEquals(1.0, s.sparsity(), 1e-10);
    }

    @Test
    public void get_outOfBounds_throwsException() {
        ISparseMatrix s = ISparseMatrix.eye(3);
        assertThrows(IndexOutOfBoundsException.class, () -> s.get(3, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> s.get(0, 3));
    }

    @Test
    public void fromDense_allNonzero_hasFullNnz() {
        double[][] data = {{1, 2}, {3, 4}};
        ISparseMatrix s = ISparseMatrix.fromDense(data);
        assertEquals(4, s.nnz());
        assertEquals(0.0, s.sparsity(), 1e-10);
    }

    @Test
    public void singleElementMatrix_allOperationsWork() {
        double[][] data = {{42.0}};
        ISparseMatrix s = ISparseMatrix.fromDense(data);
        assertEquals(1, s.rows());
        assertEquals(1, s.cols());
        assertEquals(1, s.nnz());
        assertEquals(42.0, s.get(0, 0));

        ISparseMatrix scaled = s.scale(2.0);
        assertEquals(84.0, scaled.get(0, 0));

        ISparseMatrix t = s.transpose();
        assertEquals(42.0, t.get(0, 0));

        IMatrix<Double> d = s.toDense();
        assertEquals(42.0, d.get(0, 0), 1e-10);
    }

    @Test
    public void nonSquareMatrix_basicOps_preserveShape() {
        double[][] data = {{1, 0, 2}, {0, 3, 0}};
        ISparseMatrix s = ISparseMatrix.fromDense(data);
        assertEquals(2, s.rows());
        assertEquals(3, s.cols());
        ISparseMatrix t = s.transpose();
        assertEquals(3, t.rows());
        assertEquals(2, t.cols());
    }

    @Test
    public void copy_isIndependentOfOriginal() {
        ISparseMatrix a = ISparseMatrix.diag(new double[]{1, 2, 3});
        ISparseMatrix b = a.copy();
        assertEquals(3.0, b.get(2, 2));

        ISparseMatrix c = a.scale(5.0);
        assertEquals(15.0, c.get(2, 2));
        assertEquals(3.0, b.get(2, 2), "copy should be independent");
    }

    @Test
    public void diag_emptyArray_hasZeroNnz() {
        ISparseMatrix s = ISparseMatrix.diag(new double[]{});
        assertEquals(0, s.rows());
        assertEquals(0, s.cols());
        assertEquals(0, s.nnz());
    }
}
