package com.yishape.lab.math.optimize.autodiff;

import com.yishape.lab.math.autodiff.MixedMode;
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;

public class MixedModeTest {

    private static final double TOL = 1e-7;

    @Test
    void testHvpSimpleQuadratic() {
        // f(x) = sum(x_i^2) = x·x, H = 2I, H @ v = 2*v
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDoubleVector v = IDoubleVector.of(new double[] { 3.0, -1.0, 2.0 });

        double[] hvp = MixedMode.hvp(z -> z.dot(z), x, v);

        assertArrayEquals(new double[] { 6.0, -2.0, 4.0 }, hvp, TOL);
    }

    @Test
    void testHvpWeightedQuadratic() {
        // f(x) = x1^2 + 3*x2^2, H = [[2, 0], [0, 6]]
        // H @ [1, 1] = [2, 6]
        IDiffVector x = AD.vector(new double[] { 1.5, -2.0 });
        IDoubleVector v = IDoubleVector.of(new double[] { 1.0, 1.0 });

        // f(z) = z1^2*1 + z2^2*3 = z^2 · [1, 3]
        double[] hvp = MixedMode.hvp(
            z -> z.pow(2).dot(AD.vector(new double[] { 1.0, 3.0 })),
            x, v);

        assertArrayEquals(new double[] { 2.0, 6.0 }, hvp, TOL);
    }

    @Test
    void testJvpSquare() {
        // f(x) = x^2 (element-wise), J = diag(2*x)
        // At x=[2, 3]: J = [[4, 0], [0, 6]]
        // J @ v for v=[1, 2] = [4, 12]
        IDiffVector x = AD.vector(new double[] { 2.0, 3.0 });
        IDoubleVector v = IDoubleVector.of(new double[] { 1.0, 2.0 });

        double[] jvp = MixedMode.jvp(z -> z.pow(2), x, v);

        assertArrayEquals(new double[] { 4.0, 12.0 }, jvp, TOL);
    }

    @Test
    void testHessianDotSelf() {
        // f(x) = x·x = sum(x_i^2), H = 2I
        IDiffVector x = AD.vector(new double[] { 0.5, -1.0 });

        IDoubleMatrix H = MixedMode.hessian(z -> z.dot(z), x);

        double[][] expected = {{2, 0}, {0, 2}};
        double[][] actual = H.getData();
        for (int i = 0; i < 2; i++) {
            assertArrayEquals(expected[i], actual[i], 1e-5);
        }
    }

    @Test
    void testJacobianFullSquare() {
        // f(x) = x^2 (element-wise), J = diag(2*x)
        // At x=[2, 3]: J = [[4, 0], [0, 6]]
        IDiffVector x = AD.vector(new double[] { 2.0, 3.0 });

        IDoubleMatrix J = MixedMode.jacobianFull(z -> z.pow(2), x);

        double[][] expected = {{4, 0}, {0, 6}};
        double[][] actual = J.getData();
        assertEquals(2, actual.length);
        assertEquals(2, actual[0].length);
        for (int i = 0; i < 2; i++) {
            assertArrayEquals(expected[i], actual[i], 1e-5);
        }
    }
}
