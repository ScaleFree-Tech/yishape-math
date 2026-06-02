package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;

/**
 * {@link IMatrix#toDoubleArray()} / {@link IMatrix#toFloatArray()} 与向量同类方法返回防御性拷贝，修改返回值不改变原对象。
 */
class ToArrayCopySemanticsTest {

    @Test
    void doubleMatrixToDoubleArrayMutationDoesNotAffectMatrix() {
        IDoubleMatrix m = IDoubleMatrix.of(new double[][]{{1, 2}, {3, 4}});
        double[][] d = m.toDoubleArray();
        assertNotSame(m.getData(), d);
        assertNotSame(m.getData()[0], d[0]);
        d[0][0] = 99;
        assertEquals(1.0, m.get(0, 0), 0.0);
    }

    @Test
    void floatMatrixToFloatArrayMutationDoesNotAffectMatrix() {
        IFloatMatrix m = IFloatMatrix.of(new float[][]{{1, 2}, {3, 4}});
        float[][] f = m.toFloatArray();
        assertNotSame(m.getData(), f);
        assertNotSame(m.getData()[0], f[0]);
        f[0][0] = 99f;
        assertEquals(1f, m.get(0, 0), 0f);
    }

    @Test
    void doubleVectorToDoubleArrayMutationDoesNotAffectVector() {
        IDoubleVector v = IDoubleVector.of(new double[]{1, 2, 3});
        double[] d = v.toDoubleArray();
        assertNotSame(v.getData(), d);
        d[0] = 99;
        assertEquals(1.0, v.get(0), 0.0);
    }

    @Test
    void floatVectorToFloatArrayMutationDoesNotAffectVector() {
        IFloatVector v = IFloatVector.of(new float[]{1, 2, 3});
        float[] f = v.toFloatArray();
        assertNotSame(v.getData(), f);
        f[0] = 99f;
        assertEquals(1f, v.get(0), 0f);
    }
}
