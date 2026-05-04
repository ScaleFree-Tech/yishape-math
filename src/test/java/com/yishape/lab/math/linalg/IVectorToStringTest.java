package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class IVectorToStringTest {

    @Test
    void shortDoubleVectorBracketed() {
        IDoubleVector v = IDoubleVector.of(new double[]{0, 1.5, -2.25});
        assertEquals("[0, 1.5, -2.25]", v.toString());
    }

    @Test
    void longDoubleVectorEllipsisAndLength() {
        double[] d = new double[30];
        for (int i = 0; i < 30; i++) {
            d[i] = i;
        }
        String s = IDoubleVector.of(d).toString();
        assertTrue(s.startsWith("["), s);
        assertTrue(s.contains(", ..., "), s);
        assertTrue(s.endsWith("] (length=30)"), s);
    }

    @Test
    void emptyVector() {
        assertEquals("[]", IDoubleVector.of(new double[0]).toString());
    }

    @Test
    void floatVectorConsistent() {
        IFloatVector v = IFloatVector.of(new float[]{1f, 2f});
        assertEquals("[1, 2]", v.toString());
    }
}
