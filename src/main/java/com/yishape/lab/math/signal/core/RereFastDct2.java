package com.yishape.lab.math.signal.core;

/**
 * 长度 n 为 2 的幂时的快速 DCT-II / 逆 DCT-II（Lee 1984 递推），复杂度 O(n log n)。
 * 正交归一化与 {@link RereDCT#dct2} / {@link RereDCT#idct2} 的朴素实现一致；非 2 的幂仍由 {@link RereDCT} 走 O(n²) 路径。
 */
public final class RereFastDct2 {

    private RereFastDct2() {
    }

    public static boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    /**
     * 与 RereDCT 一致的正交 DCT-II，原地覆盖 {@code x[0..n)}。
     */
    public static void dct2OrthoInPlace(double[] x, int n) {
        dct2LeeUnnormalized(x, n);
        double invSqrtN = 1.0 / Math.sqrt(n);
        double sqrt2OverN = Math.sqrt(2.0 / n);
        for (int k = 0; k < n; k++) {
            x[k] *= (k == 0) ? invSqrtN : sqrt2OverN;
        }
    }

    /**
     * 与 RereDCT 一致的逆 DCT-II，原地覆盖。
     */
    public static void idct2OrthoInPlace(double[] x, int n) {
        double invSqrtN = 1.0 / Math.sqrt(n);
        double sqrt2OverN = Math.sqrt(2.0 / n);
        for (int k = 0; k < n; k++) {
            x[k] /= (k == 0) ? invSqrtN : sqrt2OverN;
        }
        idct2LeeUnnormalized(x, n);
    }

    /** 未归一化 DCT-II，Lee 算法。 */
    static void dct2LeeUnnormalized(double[] x, int n) {
        if (n == 1) {
            return;
        }
        int h = n / 2;
        double[] a = new double[h];
        double[] b = new double[h];
        for (int i = 0; i < h; i++) {
            a[i] = x[i] + x[n - 1 - i];
            b[i] = (x[i] - x[n - 1 - i])
                    / (2.0 * Math.cos((2 * i + 1) * Math.PI / (2 * n)));
        }
        dct2LeeUnnormalized(a, h);
        dct2LeeUnnormalized(b, h);
        for (int i = 0; i < h; i++) {
            x[2 * i] = a[i];
            x[2 * i + 1] = b[i] + ((i + 1 < h) ? b[i + 1] : 0.0);
        }
    }

    /** {@link #dct2LeeUnnormalized} 的逆。 */
    static void idct2LeeUnnormalized(double[] x, int n) {
        if (n == 1) {
            return;
        }
        int h = n / 2;
        double[] a = new double[h];
        double[] b = new double[h];
        for (int i = 0; i < h; i++) {
            a[i] = x[2 * i];
        }
        b[h - 1] = x[2 * h - 1];
        for (int i = h - 2; i >= 0; i--) {
            b[i] = x[2 * i + 1] - b[i + 1];
        }
        idct2LeeUnnormalized(a, h);
        idct2LeeUnnormalized(b, h);
        for (int i = 0; i < h; i++) {
            double c = 2.0 * Math.cos((2 * i + 1) * Math.PI / (2 * n));
            x[i] = (a[i] + c * b[i]) / 2.0;
            x[n - 1 - i] = (a[i] - c * b[i]) / 2.0;
        }
    }
}
