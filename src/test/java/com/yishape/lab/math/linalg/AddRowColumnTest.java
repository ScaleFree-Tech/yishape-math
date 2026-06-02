package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AddRowColumnTest {

    @Test
    void doubleAddColumnAppendsOnRight() {
        var m = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
        var col = Linalg.vector(new double[]{10, 20});
        var result = m.addColumn(col);
        assertEquals(2, result.getRowNum());
        assertEquals(3, result.getColNum());
        assertArrayEquals(new double[]{1, 2, 10}, result.getRow(0).toDoubleArray(), 1e-12);
        assertArrayEquals(new double[]{3, 4, 20}, result.getRow(1).toDoubleArray(), 1e-12);
    }

    @Test
    void doubleAddColumnViaDoubleVectorOverload() {
        IDoubleVector col = Linalg.vector(new double[]{5, 6});
        var result = Linalg.matrix(new double[][]{{1, 2}, {3, 4}}).addColumn(col);
        assertEquals(5, result.get(0, 2), 1e-12);
        assertEquals(6, result.get(1, 2), 1e-12);
    }

    @Test
    void doubleAddRowAppendsOnBottom() {
        var m = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
        var row = Linalg.vector(new double[]{10, 20});
        var result = m.addRow(row);
        assertEquals(3, result.getRowNum());
        assertEquals(2, result.getColNum());
        assertArrayEquals(new double[]{10, 20}, result.getRow(2).toDoubleArray(), 1e-12);
    }

    @Test
    void floatAddColumnAndAddRow() {
        var m = Linalg.matrix(new float[][]{{1f, 2f}, {3f, 4f}});
        IFloatVector col = Linalg.vector(new float[]{5f, 6f});
        var withCol = m.addColumn(col);
        assertEquals(2, withCol.getRowNum());
        assertEquals(3, withCol.getColNum());
        assertEquals(5f, withCol.get(0, 2), 1e-6f);
        assertEquals(6f, withCol.get(1, 2), 1e-6f);

        var row = Linalg.vector(new float[]{7f, 8f, 9f});
        var withRow = withCol.addRow(row);
        assertEquals(3, withRow.getRowNum());
        assertEquals(3, withRow.getColNum());
        assertEquals(9f, withRow.get(2, 2), 1e-6f);
    }

    @Test
    void nullVectorThrows() {
        var m = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
        var exCol = assertThrows(IllegalArgumentException.class, () -> m.addColumn((IDoubleVector) null));
        assertTrue(exCol.getMessage().contains("null"));
        var exRow = assertThrows(IllegalArgumentException.class, () -> m.addRow((IDoubleVector) null));
        assertTrue(exRow.getMessage().contains("null"));
    }

    @Test
    void dimensionMismatchThrowsWithClearMessage() {
        var m = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
        var colEx = assertThrows(IllegalArgumentException.class, () -> m.addColumn(Linalg.vector(new double[]{1})));
        assertTrue(colEx.getMessage().contains("行数") || colEx.getMessage().contains("row count"));
        var rowEx = assertThrows(IllegalArgumentException.class, () -> m.addRow(Linalg.vector(new double[]{1, 2, 3})));
        assertTrue(rowEx.getMessage().contains("列数") || rowEx.getMessage().contains("column count"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void floatVectorOnDoubleMatrixThrows() {
        var m = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
        IVector<Double> floatColAsDouble = (IVector<Double>) (IVector<?>) Linalg.vector(new float[]{1f, 2f});
        var ex = assertThrows(IllegalArgumentException.class, () -> m.addColumn(floatColAsDouble));
        assertTrue(ex.getMessage().contains("Float"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void doubleVectorOnFloatMatrixThrows() {
        var m = Linalg.matrix(new float[][]{{1f, 2f}, {3f, 4f}});
        IVector<Float> doubleColAsFloat = (IVector<Float>) (IVector<?>) Linalg.vector(new double[]{1, 2});
        var ex = assertThrows(IllegalArgumentException.class, () -> m.addColumn(doubleColAsFloat));
        assertTrue(ex.getMessage().contains("Double"));
    }

    @Test
    void biasColumnExample() {
        var X = Linalg.randn(10, 3);
        var Xc = X.addColumn(Linalg.ones(10));
        assertEquals(10, Xc.getRowNum());
        assertEquals(4, Xc.getColNum());
        for (int i = 0; i < 10; i++) {
            assertEquals(1.0, Xc.get(i, 3), 1e-12);
        }
    }
}
