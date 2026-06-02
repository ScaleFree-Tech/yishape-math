package com.yishape.lab.math.compute.hpc;

import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * yishape-math-hpc 为可选原生路径：无 DLL/so 时测试仍通过；存在时校验大块 matmul 数值一致。
 */
class HpcGemmOptionalTest {

    private static double[][] naiveMul(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        int p = b[0].length;
        double[][] c = new double[m][p];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += a[i][k] * b[k][j];
                }
                c[i][j] = s;
            }
        }
        return c;
    }

    @Test
    void linalgReportsHpcOnlyWhenNativeLoads() {
        assertEquals(Linalg.isHpcNativeRuntimeLoaded(), HpcOptionalRuntime.isNativeRuntimeAvailable());
    }

    @Test
    void hpcMatmulMatchesNaiveWhenAvailable() {
        assumeTrue(HpcOptionalRuntime.isNativeRuntimeAvailable(), "skip when yishape_math_rust native not available");
        int n = 180;
        double[][] a = new double[n][n];
        double[][] b = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = (i + 2 * j) * 0.001;
                b[i][j] = (i - j) * 0.002;
            }
        }
        double[][] expect = naiveMul(a, b);
        double[][] got = HpcGemm.tryMatMul(a, b);
        assertNotNull(got);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                assertEquals(expect[i][j], got[i][j], 1e-9, "at " + i + "," + j);
            }
        }
    }
}
