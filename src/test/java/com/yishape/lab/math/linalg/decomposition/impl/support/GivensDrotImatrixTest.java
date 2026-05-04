package com.yishape.lab.math.linalg.decomposition.impl.support;

import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GivensDrotImatrixTest {

    @Test
    void applyColumnsKeepsTwoNormOnRandomPlane() {
        var m = Linalg.eye(5);
        GivensDrotImatrix.applyColumns(m, 1, 3, 0.6, 0.8);
        double c1 = 0, c3 = 0;
        for (int i = 0; i < 5; i++) {
            double x = m.get(i, 1);
            double y = m.get(i, 3);
            c1 += x * x;
            c3 += y * y;
        }
        assertEquals(1.0, c1, 1e-14);
        assertEquals(1.0, c3, 1e-14);
        double dot = 0;
        for (int i = 0; i < 5; i++) {
            dot += m.get(i, 1) * m.get(i, 3);
        }
        assertEquals(0.0, dot, 1e-13);
    }

    @Test
    void applyRows1BasedOrthogonalColumnsOfIdentityAugmented() {
        double[][] a = new double[5][5];
        for (int i = 1; i <= 4; i++) {
            a[i][i] = 1.0;
        }
        GivensDrotImatrix.applyRows1Based(a, 2, 4, 4, Math.cos(0.3), Math.sin(0.3));
        double n2 = 0, n4 = 0, dot = 0;
        for (int j = 1; j <= 4; j++) {
            double x = a[2][j];
            double y = a[4][j];
            n2 += x * x;
            n4 += y * y;
            dot += x * y;
        }
        assertEquals(1.0, n2, 1e-14);
        assertEquals(1.0, n4, 1e-14);
        assertEquals(0.0, dot, 1e-13);
    }
}
