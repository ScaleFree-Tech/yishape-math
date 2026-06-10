package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.TensorFusedOps;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Simple timing benchmarks for TensorFusedOps accelerated forward paths.
 * Verifies that the SIMD-accelerated path is not slower than scalar fallback.
 */
public class TensorFusedOpsBenchmark {

    /** Large tensor for meaningful timing: 1M elements. */
    private RereDiffTensor largeTensor() {
        int n = 1_000_000;
        double[] data = new double[n];
        for (int i = 0; i < n; i++) data[i] = (i % 10) - 5;
        return new RereDiffTensor(data, n);
    }

    @Test
    void benchmarkExpFusedVsUnfused() {
        // Fused path: SIMD-accelerated via UniversalOperation.EXP
        RereDiffTensor x1 = largeTensor();
        long t0 = System.nanoTime();
        for (int iter = 0; iter < 50; iter++) {
            x1 = new RereDiffTensor(x1.toDoubleArray(), x1.shape());
            IDiffTensor fused = new TensorFusedOps(x1).exp().done();
            fused.sum().backward();
        }
        long fusedTime = System.nanoTime() - t0;

        // Unfused path: scalar loop via RereDiffTensor.exp()
        RereDiffTensor x2 = largeTensor();
        long t1 = System.nanoTime();
        for (int iter = 0; iter < 50; iter++) {
            x2 = new RereDiffTensor(x2.toDoubleArray(), x2.shape());
            IDiffTensor unfused = x2.exp();
            unfused.sum().backward();
        }
        long unfusedTime = System.nanoTime() - t1;

        // Fused path should not be significantly slower than unfused
        // (It saves memory allocation, so should be faster, but we only check it's not 2x slower)
        assertTrue(fusedTime < unfusedTime * 2,
            "Fused exp should not be >2x slower than unfused: fused="
            + (fusedTime / 1_000_000) + "ms, unfused=" + (unfusedTime / 1_000_000) + "ms");
    }

    @Test
    void benchmarkReluFusedVsUnfused() {
        RereDiffTensor x1 = largeTensor();
        long t0 = System.nanoTime();
        for (int iter = 0; iter < 50; iter++) {
            x1 = new RereDiffTensor(x1.toDoubleArray(), x1.shape());
            IDiffTensor fused = new TensorFusedOps(x1).relu().done();
            fused.sum().backward();
        }
        long fusedTime = System.nanoTime() - t0;

        RereDiffTensor x2 = largeTensor();
        long t1 = System.nanoTime();
        for (int iter = 0; iter < 50; iter++) {
            x2 = new RereDiffTensor(x2.toDoubleArray(), x2.shape());
            IDiffTensor unfused = x2.relu();
            unfused.sum().backward();
        }
        long unfusedTime = System.nanoTime() - t1;

        assertTrue(fusedTime < unfusedTime * 2,
            "Fused relu should not be >2x slower than unfused: fused="
            + (fusedTime / 1_000_000) + "ms, unfused=" + (unfusedTime / 1_000_000) + "ms");
    }

    @Test
    void benchmarkChainFusedVsUnfused() {
        // Chain: exp().relu().tanh().sigmoid() — all SIMD-accelerated
        RereDiffTensor x1 = largeTensor();
        long t0 = System.nanoTime();
        for (int iter = 0; iter < 20; iter++) {
            x1 = new RereDiffTensor(x1.toDoubleArray(), x1.shape());
            IDiffTensor fused = new TensorFusedOps(x1).exp().relu().tanh().sigmoid().done();
            fused.sum().backward();
        }
        long fusedTime = System.nanoTime() - t0;

        RereDiffTensor x2 = largeTensor();
        long t1 = System.nanoTime();
        for (int iter = 0; iter < 20; iter++) {
            x2 = new RereDiffTensor(x2.toDoubleArray(), x2.shape());
            IDiffTensor unfused = x2.exp().relu().tanh().sigmoid();
            unfused.sum().backward();
        }
        long unfusedTime = System.nanoTime() - t1;

        // Fused chain is expected to be faster (fewer intermediate tensor allocations)
        assertTrue(fusedTime < unfusedTime * 2,
            "Fused chain should not be >2x slower: fused="
            + (fusedTime / 1_000_000) + "ms, unfused=" + (unfusedTime / 1_000_000) + "ms");
    }
}
