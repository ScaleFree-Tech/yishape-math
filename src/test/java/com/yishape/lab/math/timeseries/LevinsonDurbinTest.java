package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.timeseries.solver.LevinsonDurbinSolver;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class LevinsonDurbinTest {

    @Test
    public void yuleWalker_AR1_matchesExpected() {
        double[] r = {1.0, 0.8};
        LevinsonDurbinSolver.YuleWalkerResult result =
                LevinsonDurbinSolver.solveYuleWalkerDetailed(r, 1);

        assertEquals(-0.8, result.arCoefficients().get(0), 1e-10);
        assertEquals(0.36, result.noiseVariance(), 1e-10);
        assertEquals(-0.8, result.reflectionCoefficients().get(0), 1e-10);
    }

    @Test
    public void yuleWalker_AR2_matchesKnown() {
        double[] r = {1.0, 0.5, 0.3};
        LevinsonDurbinSolver.YuleWalkerResult result =
                LevinsonDurbinSolver.solveYuleWalkerDetailed(r, 2);

        double[] ar = result.arCoefficients().toDoubleArray();
        double[] refl = result.reflectionCoefficients().toDoubleArray();

        assertEquals(-0.5, refl[0], 1e-10);
        assertEquals(2, ar.length);
        assertTrue(Math.abs(result.noiseVariance()) > 0);
    }

    @Test
    public void yuleWalker_whiteNoise_givesZeroAr() {
        double[] r = {1.0, 0.0, 0.0, 0.0};
        LevinsonDurbinSolver.YuleWalkerResult result =
                LevinsonDurbinSolver.solveYuleWalkerDetailed(r, 3);

        for (int i = 0; i < 3; i++) {
            assertEquals(0.0, result.arCoefficients().get(i), 1e-10);
        }
        assertEquals(1.0, result.noiseVariance(), 1e-10);
    }

    @Test
    public void toeplitzSolve_vs_denseSolve() {
        double[] c = {5, 2, 1, 0.5};
        double[] b = {10, 8, 6, 4};

        IVector<Double> xLD = LevinsonDurbinSolver.solve(c, b);

        IMatrix<Double> T = Linalg.toeplitz(c);
        IVector<Double> xDense = Linalg.solve(T, Linalg.vector(b));

        assertEquals(xDense.length(), xLD.length());
        for (int i = 0; i < xLD.length(); i++) {
            assertEquals(xDense.get(i), xLD.get(i), 1e-8);
        }
    }

    @Test
    public void toeplitzSolve_matrixAPI_vs_denseSolve() {
        double[] c = {4, 1, 0};
        IMatrix<Double> T = Linalg.toeplitz(c);
        IVector<Double> b = Linalg.vector(new double[]{1, 2, 3});

        IVector<Double> xLD = LevinsonDurbinSolver.solve(T, b);
        IVector<Double> xRef = Linalg.solve(T, b);

        for (int i = 0; i < 3; i++) {
            assertEquals(xRef.get(i), xLD.get(i), 1e-8);
        }
    }

    @Test
    public void pacf_usesLevinsonDurbin() {
        // AR(1) process: x[n] = 0.7*x[n-1] + noise
        // PACF should be ~0.7 at lag 1, ~0 at lag > 1
        int n = 500;
        double[] data = new double[n];
        data[0] = 0;
        java.util.Random rng = new java.util.Random(42);
        for (int i = 1; i < n; i++) {
            data[i] = 0.7 * data[i - 1] + rng.nextGaussian();
        }
        IVector<Double> tsData = Linalg.vector(data);

        IVector<Double> pacf = TimeSeriesUtils.calculatePartialAutocorrelation(tsData, 5);

        assertEquals(1.0, pacf.get(0), 1e-10);
        assertTrue(Math.abs(pacf.get(1)) > 0.3, "PACF(1) should be significantly non-zero");
        // For AR(1), PACF at lag 2+ should be close to 0
        assertTrue(Math.abs(pacf.get(3)) < 0.3, "PACF(3) should be near zero for AR(1)");
    }
}
