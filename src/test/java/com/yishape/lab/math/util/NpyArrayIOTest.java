package com.yishape.lab.math.util;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.*;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link NpyArrayIO} utility class.
 */
class NpyArrayIOTest {

    // ==================== Round-trip 2D ====================

    @Test
    void roundTrip_doubleArray2D() throws IOException {
        double[][] data = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        byte[] bytes = NpyArrayIO.toByteArrayDouble2D(data);
        assertNotNull(bytes);
        assertTrue(bytes.length > 0);

        double[][] read = NpyArrayIO.readDouble2D(new ByteArrayInputStream(bytes));
        assertEquals(2, read.length);
        assertEquals(3, read[0].length);
        assertEquals(1.0, read[0][0], 1e-10);
        assertEquals(6.0, read[1][2], 1e-10);
    }

    @Test
    void roundTrip_imatrix() throws IOException {
        IMatrix<Double> mat = Linalg.matrix(new double[][]{{1.5, 2.5}, {3.5, 4.5}});
        byte[] bytes = NpyArrayIO.toByteArray(mat);
        IMatrix<Double> read = NpyArrayIO.fromByteArray(bytes);
        assertEquals(2, read.rows());
        assertEquals(2, read.cols());
        assertEquals(1.5, read.get(0, 0), 1e-10);
        assertEquals(4.5, read.get(1, 1), 1e-10);
    }

    @Test
    void roundTrip_1D_array() throws IOException {
        double[] data = {1.0, 2.0, 3.0, 4.0};
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        NpyArrayIO.writeDouble1D(baos, data);

        double[][] read = NpyArrayIO.readDouble2D(new ByteArrayInputStream(baos.toByteArray()));
        // 1D data returned as 1xN matrix
        assertEquals(1, read.length);
        assertEquals(4, read[0].length);
        assertEquals(1.0, read[0][0], 1e-10);
        assertEquals(4.0, read[0][3], 1e-10);
    }

    @Test
    void roundTrip_1D_ivector() throws IOException {
        IVector<Double> vec = Linalg.vector(new double[]{10.0, 20.0, 30.0});
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        NpyArrayIO.writeDouble1D(baos, vec);

        IMatrix<Double> read = NpyArrayIO.readMatrix(new ByteArrayInputStream(baos.toByteArray()));
        assertEquals(1, read.rows());
        assertEquals(3, read.cols());
        assertEquals(10.0, read.get(0, 0), 1e-10);
        assertEquals(30.0, read.get(0, 2), 1e-10);
    }

    // ==================== Write to OutputStream ====================

    @Test
    void writeDouble2D_withIMatrix() throws IOException {
        IMatrix<Double> mat = Linalg.matrix(new double[][]{{7.0, 8.0}, {9.0, 10.0}});
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        NpyArrayIO.writeDouble2D(baos, mat);

        double[][] read = NpyArrayIO.readDouble2D(new ByteArrayInputStream(baos.toByteArray()));
        assertEquals(7.0, read[0][0], 1e-10);
        assertEquals(10.0, read[1][1], 1e-10);
    }

    // ==================== Error Conditions ====================

    @Test
    void writeDouble2D_nullOut_throws() {
        assertThrows(NullPointerException.class,
            () -> NpyArrayIO.writeDouble2D(null, new double[][]{{1.0}}));
    }

    @Test
    void writeDouble2D_nullData_throws() throws IOException {
        assertThrows(NullPointerException.class,
            () -> NpyArrayIO.writeDouble2D(new ByteArrayOutputStream(), (double[][]) null));
    }

    @Test
    void writeDouble2D_raggedRows_throws() {
        double[][] ragged = {{1.0, 2.0}, {3.0}};
        assertThrows(IllegalArgumentException.class,
            () -> NpyArrayIO.writeDouble2D(new ByteArrayOutputStream(), ragged));
    }

    @Test
    void writeDouble1D_nullOut_throws() {
        assertThrows(NullPointerException.class,
            () -> NpyArrayIO.writeDouble1D(null, new double[]{1.0}));
    }

    @Test
    void writeDouble1D_nullData_throws() {
        assertThrows(NullPointerException.class,
            () -> NpyArrayIO.writeDouble1D(new ByteArrayOutputStream(), (double[]) null));
    }

    @Test
    void readDouble2D_invalidMagic_throws() {
        byte[] garbage = new byte[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        assertThrows(IOException.class,
            () -> NpyArrayIO.readDouble2D(new ByteArrayInputStream(garbage)));
    }

    // ==================== Single Row ====================

    @Test
    void singleRowMatrix() throws IOException {
        double[][] data = {{1.0, 2.0, 3.0, 4.0, 5.0}};
        byte[] bytes = NpyArrayIO.toByteArrayDouble2D(data);
        double[][] read = NpyArrayIO.readDouble2D(new ByteArrayInputStream(bytes));
        assertEquals(1, read.length);
        assertEquals(5, read[0].length);
    }

    // ==================== Edge Cases ====================

    @Test
    void singleElementMatrix() throws IOException {
        double[][] data = {{42.0}};
        byte[] bytes = NpyArrayIO.toByteArrayDouble2D(data);
        double[][] read = NpyArrayIO.readDouble2D(new ByteArrayInputStream(bytes));
        assertEquals(1, read.length);
        assertEquals(1, read[0].length);
        assertEquals(42.0, read[0][0], 1e-10);
    }

    @Test
    void largeValues() throws IOException {
        double[][] data = {{Double.MAX_VALUE, -Double.MAX_VALUE}};
        byte[] bytes = NpyArrayIO.toByteArrayDouble2D(data);
        double[][] read = NpyArrayIO.readDouble2D(new ByteArrayInputStream(bytes));
        assertEquals(Double.MAX_VALUE, read[0][0], 0);
        assertEquals(-Double.MAX_VALUE, read[0][1], 0);
    }

    @Test
    void specialValues() throws IOException {
        double[][] data = {{Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY}};
        byte[] bytes = NpyArrayIO.toByteArrayDouble2D(data);
        double[][] read = NpyArrayIO.readDouble2D(new ByteArrayInputStream(bytes));
        assertTrue(Double.isNaN(read[0][0]));
        assertEquals(Double.POSITIVE_INFINITY, read[0][1], 0);
        assertEquals(Double.NEGATIVE_INFINITY, read[0][2], 0);
    }
}
