package com.yishape.lab.math.signal.core;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ThreadLocalRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RereFastDct2Test {

    private static IVector<Double> naiveDct2(IVector<Double> signal) {
        int n = signal.length();
        IVector<Double> result = Linalg.zeros(n);
        for (int k = 0; k < n; k++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += signal.get(i) * Math.cos(Math.PI * k * (2 * i + 1) / (2 * n));
            }
            double alpha = (k == 0) ? Math.sqrt(1.0 / n) : Math.sqrt(2.0 / n);
            result.set(k, alpha * sum);
        }
        return result;
    }

    private static IVector<Double> naiveIdct2(IVector<Double> dct) {
        int n = dct.length();
        IVector<Double> result = Linalg.zeros(n);
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                double alpha = (k == 0) ? Math.sqrt(1.0 / n) : Math.sqrt(2.0 / n);
                sum += alpha * dct.get(k) * Math.cos(Math.PI * k * (2 * i + 1) / (2 * n));
            }
            result.set(i, sum);
        }
        return result;
    }

    @Test
    void fastDct2MatchesNaivePowersOfTwoThrough1024() {
        ThreadLocalRandom rnd = ThreadLocalRandom.current();
        for (int n = 1; n <= 1024; n <<= 1) {
            double[] raw = new double[n];
            for (int i = 0; i < n; i++) {
                raw[i] = rnd.nextDouble(-1, 1);
            }
            IVector<Double> sig = Linalg.vector(raw);
            IVector<Double> naive = naiveDct2(sig);
            IVector<Double> fast = RereDCT.dct2(sig);
            for (int i = 0; i < n; i++) {
                assertEquals(naive.get(i), fast.get(i), 1e-9 * Math.max(1.0, Math.abs(naive.get(i))),
                        "n=" + n + " i=" + i);
            }
        }
    }

    @Test
    void roundtripDctIdct16384() {
        int n = 16384;
        double[] raw = new double[n];
        ThreadLocalRandom rnd = ThreadLocalRandom.current();
        for (int i = 0; i < n; i++) {
            raw[i] = rnd.nextGaussian();
        }
        IVector<Double> sig = Linalg.vector(raw);
        IVector<Double> c = RereDCT.dct2(sig);
        IVector<Double> back = RereDCT.idct2(c);
        double err = 0;
        for (int i = 0; i < n; i++) {
            double d = raw[i] - back.get(i);
            err += d * d;
        }
        assertTrue(err < 1e-6 * n, "roundtrip rmse too large: " + err);
    }

    @Test
    void idct2MatchesNaive() {
        int n = 512;
        double[] raw = new double[n];
        for (int i = 0; i < n; i++) {
            raw[i] = ThreadLocalRandom.current().nextDouble(-1, 1);
        }
        IVector<Double> x = Linalg.vector(raw);
        IVector<Double> naive = naiveIdct2(x);
        IVector<Double> fast = RereDCT.idct2(x);
        for (int i = 0; i < n; i++) {
            assertEquals(naive.get(i), fast.get(i), 1e-9);
        }
    }

    @Test
    void dct2DRoundtripPowerOfTwo() {
        int r = 32;
        int c = 64;
        double[][] m = new double[r][c];
        var rnd = ThreadLocalRandom.current();
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                m[i][j] = rnd.nextDouble(-1, 1);
            }
        }
        double[][] d = RereDCT.dct2D(m);
        double[][] back = RereDCT.idct2D(d);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                assertEquals(m[i][j], back[i][j], 1e-8);
            }
        }
    }

    @Test
    void nonPowerOfTwoStillUsesNaivePath() {
        int n = 15;
        double[] raw = new double[n];
        for (int i = 0; i < n; i++) {
            raw[i] = 0.1 * i;
        }
        IVector<Double> sig = Linalg.vector(raw);
        assertEquals(naiveDct2(sig).get(0), RereDCT.dct2(sig).get(0), 1e-12);
    }
}
