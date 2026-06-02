package com.yishape.lab.math.compute.hpc;

/**
 * 大块双精度矩阵乘可选委托扩展库中的 faer / 原生实现；失败或未达阈值时返回 {@code null}。
 */
public final class HpcGemm {

    private HpcGemm() {
    }

    /**
     * @param a m×n
     * @param b n×p
     * @return 新的 m×p，或 {@code null} 表示应回退 Java
     */
    public static double[][] tryMatMul(double[][] a, double[][] b) {
        if (a == null || b == null || a.length == 0 || b.length == 0) {
            return null;
        }
        int m = a.length;
        int n = a[0].length;
        int p = b[0].length;
        if (b.length != n) {
            return null;
        }
        for (int i = 0; i < m; i++) {
            if (a[i] == null || a[i].length != n) {
                return null;
            }
        }
        for (int i = 0; i < n; i++) {
            if (b[i] == null || b[i].length != p) {
                return null;
            }
        }
        long flops = (long) m * n * p;
        if (flops < HpcConfig.gemmMinFlops()) {
            return null;
        }
        if (!HpcConfig.allowAttempts()) {
            return null;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return null;
        }
        try {
            return HpcOptionalRuntime.tryMatMul(a, b);
        } catch (Throwable t) {
            return null;
        }
    }
}
