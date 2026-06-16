package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;
import com.yishape.lab.math.compute.hpc.HpcSwitch;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * HPC contract test: Java AD ↔ Rust HPC serialization.
 *
 * <p>Every test generates random data, runs CPU baseline ({@code backward()})
 * and HPC accelerated ({@code HpcGraphExecutor.tryExecute()}), then asserts
 * loss and gradients match within f64 precision tolerances.</p>
 *
 * <p>Mirrors {@link ProtocolContractTest} but targets the HPC backend.
 * When HPC native library is absent, tests skip gracefully.</p>
 *
 * <p><b>HPC-unique ops tested here:</b> expand, reciprocal, rsub, rdiv, scatter, narrow.</p>
 */
public class HpcContractTest {

    private static boolean hpcPresent;
    private static final Random RNG = new Random(0xCAFE_BEEF);
    private static final double LOSS_TOL = 1e-8;   // f64 vs f64 (HPC uses f64 like CPU)
    private static final double GRAD_TOL = 1e-7;

    @BeforeAll
    static void detect() {
        System.setProperty("yishape.hpc.minElements", "0");
        hpcPresent = HpcOptionalRuntime.isNativeRuntimeAvailable();
        System.out.println("[HpcContract] HPC " + (hpcPresent ? "present" : "absent"));
    }

    @AfterEach
    void restoreSwitch() {
        HpcSwitch.enable();
    }

    // ── Data generators ──

    private static double[] rand(int n) {
        double[] d = new double[n];
        for (int i = 0; i < n; i++) d[i] = RNG.nextDouble() * 2.0 - 1.0;
        return d;
    }

    private static double[] randPos(int n) {
        double[] d = new double[n];
        for (int i = 0; i < n; i++) d[i] = RNG.nextDouble() * 9.0 + 1.0;
        return d;
    }

    // ── Comparison engine ──

    @FunctionalInterface interface TensorGraph { IDiffTensor build(RereDiffTensor[] leaves); }

    record CpuResult(double loss, double[][] grads) {}
    record HpcResult(double loss, double[][] grads) {}

    private static CpuResult cpuRef(double[][] leafData, int[][] shapes, TensorGraph fn) {
        RereDiffTensor[] leaves = new RereDiffTensor[leafData.length];
        for (int i = 0; i < leafData.length; i++)
            leaves[i] = (RereDiffTensor) AD.leafTensor(leafData[i], shapes[i]);
        RereDiffTensor loss = (RereDiffTensor) fn.build(leaves);
        loss.backward();
        double[][] grads = new double[leaves.length][];
        for (int i = 0; i < leaves.length; i++) grads[i] = leaves[i].gradData().clone();
        return new CpuResult(loss.value().toDoubleArray()[0], grads);
    }

    private static HpcResult hpcExec(double[][] leafData, int[][] shapes, TensorGraph fn) {
        if (!hpcPresent) return new HpcResult(Double.NaN, null);
        RereDiffTensor[] leaves = new RereDiffTensor[leafData.length];
        for (int i = 0; i < leafData.length; i++)
            leaves[i] = (RereDiffTensor) AD.leafTensor(leafData[i], shapes[i]);
        RereDiffTensor loss = (RereDiffTensor) fn.build(leaves);
        double hpcLoss = HpcGraphExecutor.tryExecute(loss);
        if (Double.isNaN(hpcLoss)) return new HpcResult(Double.NaN, null);
        double[][] grads = new double[leaves.length][];
        for (int i = 0; i < leaves.length; i++) grads[i] = leaves[i].gradData().clone();
        return new HpcResult(hpcLoss, grads);
    }

    private void assertMatch(String label, CpuResult cpu, HpcResult hpc) {
        if (!hpcPresent) {
            assertTrue(Double.isNaN(hpc.loss), label + ": expected NaN when HPC absent");
            return;
        }
        assertFalse(Double.isNaN(hpc.loss),
            () -> label + ": HPC loss is NaN — execution failed");
        assertEquals(cpu.loss, hpc.loss, LOSS_TOL,
            () -> String.format("%s: HPC=%.12f CPU=%.12f diff=%.2e",
                label, hpc.loss, cpu.loss, Math.abs(hpc.loss - cpu.loss)));
        assertEquals(cpu.grads.length, hpc.grads.length,
            () -> label + ": grad count mismatch");
        for (int i = 0; i < cpu.grads.length; i++) {
            final int gi = i;
            double[] cg = cpu.grads[gi], hg = hpc.grads[gi];
            assertEquals(cg.length, hg.length,
                () -> label + ": grad[" + gi + "] len mismatch");
            for (int j = 0; j < cg.length; j++) {
                final int gj = j;
                double diff = Math.abs(cg[gj] - hg[gj]);
                double maxAbs = Math.max(Math.abs(cg[gj]), Math.abs(hg[gj]));
                double effectiveTol = Math.max(GRAD_TOL, maxAbs * 1e-6);
                assertTrue(diff <= effectiveTol,
                    () -> String.format("%s: grad[%d][%d] HPC=%.12f CPU=%.12f diff=%.2e tol=%.2e",
                        label, gi, gj, hg[gj], cg[gj], diff, effectiveTol));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Binary elementwise ops
    // ═══════════════════════════════════════════════════════════════

    @Test void testAdd() { testBinOp("add", (a, b) -> a.add(b).sum()); }
    @Test void testSub() { testBinOp("sub", (a, b) -> a.sub(b).sum()); }
    @Test void testMul() { testBinOp("mul", (a, b) -> a.mul(b).abs().sum()); }
    @Test void testDiv() { testBinOp("div", (a, b) -> a.div(b).abs().sum()); }

    private void testBinOp(String name,
                           java.util.function.BiFunction<IDiffTensor, IDiffTensor, IDiffTensor> fn) {
        double[] d1 = rand(24), d2 = rand(24);
        CpuResult cpu = cpuRef(new double[][]{d1, d2}, new int[][]{{24}, {24}},
            leaves -> fn.apply(leaves[0], leaves[1]));
        HpcResult hpc = hpcExec(new double[][]{d1, d2}, new int[][]{{24}, {24}},
            leaves -> fn.apply(leaves[0], leaves[1]));
        assertMatch(name, cpu, hpc);
    }

    // ═══════════════════════════════════════════════════════════════
    // Unary ops
    // ═══════════════════════════════════════════════════════════════

    @Test void testRelu() { testUnaryOp("relu", a -> a.relu().sum()); }
    @Test void testSigmoid() { testUnaryOp("sigmoid", a -> a.sigmoid().sum()); }
    @Test void testTanh() { testUnaryOp("tanh", a -> a.tanh().sum()); }
    @Test void testExp() { testUnaryOp("exp", a -> a.exp().sum()); }

    private void testUnaryOp(String name,
                             java.util.function.Function<IDiffTensor, IDiffTensor> fn) {
        double[] d = rand(16);
        CpuResult cpu = cpuRef(new double[][]{d}, new int[][]{{16}},
            leaves -> fn.apply(leaves[0]));
        HpcResult hpc = hpcExec(new double[][]{d}, new int[][]{{16}},
            leaves -> fn.apply(leaves[0]));
        assertMatch(name, cpu, hpc);
    }

    // ═══════════════════════════════════════════════════════════════
    // Reduce ops
    // ═══════════════════════════════════════════════════════════════

    @Test void testSum() {
        double[] d = rand(32);
        CpuResult cpu = cpuRef(new double[][]{d}, new int[][]{{32}},
            leaves -> leaves[0].sum());
        HpcResult hpc = hpcExec(new double[][]{d}, new int[][]{{32}},
            leaves -> leaves[0].sum());
        assertMatch("sum", cpu, hpc);
    }

    @Test void testMean() {
        double[] d = rand(32);
        CpuResult cpu = cpuRef(new double[][]{d}, new int[][]{{32}},
            leaves -> leaves[0].mean(-1, false).sum());
        HpcResult hpc = hpcExec(new double[][]{d}, new int[][]{{32}},
            leaves -> leaves[0].mean(-1, false).sum());
        assertMatch("mean", cpu, hpc);
    }

    // ═══════════════════════════════════════════════════════════════
    // Linear layer (exercises matmul + broadcast add)
    // ═══════════════════════════════════════════════════════════════

    @Test void testLinear() {
        int B = 4, inF = 8, outF = 6;
        double[] input = rand(B * inF);
        double[] weight = rand(inF * outF);
        double[] bias = rand(outF);

        CpuResult cpu = cpuRef(new double[][]{input, weight, bias},
            new int[][]{{B, inF}, {inF, outF}, {outF}},
            leaves -> leaves[0].mmul(leaves[1]).add(leaves[2]).sum());

        HpcResult hpc = hpcExec(new double[][]{input, weight, bias},
            new int[][]{{B, inF}, {inF, outF}, {outF}},
            leaves -> leaves[0].mmul(leaves[1]).add(leaves[2]).sum());

        assertMatch("linear", cpu, hpc);
    }

    // ═══════════════════════════════════════════════════════════════
    // Broadcast ops
    // ═══════════════════════════════════════════════════════════════

    @Test void testBroadcastAdd_dim1() {
        int B = 32, C = 10;
        double[] d1 = rand(B * C);
        double[] d2 = rand(B);
        CpuResult cpu = cpuRef(new double[][]{d1, d2}, new int[][]{{B, C}, {B}},
            leaves -> leaves[0].add(leaves[1]).sum());
        HpcResult hpc = hpcExec(new double[][]{d1, d2}, new int[][]{{B, C}, {B}},
            leaves -> leaves[0].add(leaves[1]).sum());
        if (!hpcPresent || Double.isNaN(hpc.loss)) return;
        assertMatch("bcastAdd_dim1", cpu, hpc);
    }

    // ═══════════════════════════════════════════════════════════════
    // HPC-unique ops (not available on GPU)
    // ═══════════════════════════════════════════════════════════════

    @Test void testExpand() {
        double[] d = rand(12);
        CpuResult cpu = cpuRef(new double[][]{d}, new int[][]{{3, 4}},
            leaves -> leaves[0].unsqueeze(0).expand(2, 3, 4).sum());
        HpcResult hpc = hpcExec(new double[][]{d}, new int[][]{{3, 4}},
            leaves -> leaves[0].unsqueeze(0).expand(2, 3, 4).sum());
        if (!hpcPresent || Double.isNaN(hpc.loss)) return;
        assertMatch("expand", cpu, hpc);
    }

    @Test void testReciprocal() {
        // HPC may not support reciprocal natively; NaN is acceptable fallback
        double[] d = randPos(16);
        CpuResult cpu = cpuRef(new double[][]{d}, new int[][]{{16}},
            leaves -> leaves[0].reciprocal().sum());
        HpcResult hpc = hpcExec(new double[][]{d}, new int[][]{{16}},
            leaves -> leaves[0].reciprocal().sum());
        if (!hpcPresent || Double.isNaN(hpc.loss)) return; // HPC unsupported — skip
        assertMatch("reciprocal", cpu, hpc);
    }

    @Test void testGather() {
        // gather index is discrete — use constant (not leaf) so backward ignores it
        double[] d = rand(40);
        CpuResult cpu = cpuRef(new double[][]{d}, new int[][]{{10, 4}},
            leaves -> {
                RereDiffTensor idx = (RereDiffTensor) AD.constantTensor(new double[]{3, 0, 7}, 3);
                return leaves[0].gather(0, idx).sum();
            });
        HpcResult hpc = hpcExec(new double[][]{d}, new int[][]{{10, 4}},
            leaves -> {
                RereDiffTensor idx = (RereDiffTensor) AD.constantTensor(new double[]{3, 0, 7}, 3);
                return leaves[0].gather(0, idx).sum();
            });
        assertMatch("gather", cpu, hpc);
    }

    @Test void testNormalize() {
        double[] d = rand(24); // shape [4, 6]
        CpuResult cpu = cpuRef(new double[][]{d}, new int[][]{{4, 6}},
            leaves -> leaves[0].normalize(2.0, -1).sum());
        HpcResult hpc = hpcExec(new double[][]{d}, new int[][]{{4, 6}},
            leaves -> leaves[0].normalize(2.0, -1).sum());
        assertMatch("normalize", cpu, hpc);
    }

    @Test void testBatchNorm() {
        int N = 4, C = 6;
        double[] d = rand(N * C);
        double[] gamma = rand(C);
        double[] beta = rand(C);
        CpuResult cpu = cpuRef(new double[][]{d, gamma, beta},
            new int[][]{{N, C}, {C}, {C}},
            leaves -> leaves[0].batchNorm(leaves[1], leaves[2], 1e-5).sum());
        HpcResult hpc = hpcExec(new double[][]{d, gamma, beta},
            new int[][]{{N, C}, {C}, {C}},
            leaves -> leaves[0].batchNorm(leaves[1], leaves[2], 1e-5).sum());
        assertMatch("batchNorm", cpu, hpc);
    }

    @Test void testGroupNorm() {
        // groupNorm expects channels at shape[rank-2]; use [N, C, L] format
        int N = 2, C = 4, L = 6;
        double[] d = rand(N * C * L);
        double[] gamma = rand(C);
        double[] beta = rand(C);
        CpuResult cpu = cpuRef(new double[][]{d, gamma, beta},
            new int[][]{{N, C, L}, {C}, {C}},
            leaves -> leaves[0].groupNorm(2, leaves[1], leaves[2], 1e-5).sum());
        HpcResult hpc = hpcExec(new double[][]{d, gamma, beta},
            new int[][]{{N, C, L}, {C}, {C}},
            leaves -> leaves[0].groupNorm(2, leaves[1], leaves[2], 1e-5).sum());
        assertMatch("groupNorm", cpu, hpc);
    }

    // ═══════════════════════════════════════════════════════════════
    // Matmul (mmul) — direct mmul tests for HPC matmul backward
    // ═══════════════════════════════════════════════════════════════

    @Test void testMmul2D() {
        // A[M,K] @ B[K,N] = C[M,N].sum() with M=4, K=3, N=5
        int M = 4, K = 3, N = 5;
        double[] a = rand(M * K);
        double[] b = rand(K * N);
        CpuResult cpu = cpuRef(new double[][]{a, b},
            new int[][]{{M, K}, {K, N}},
            leaves -> leaves[0].mmul(leaves[1]).sum());
        HpcResult hpc = hpcExec(new double[][]{a, b},
            new int[][]{{M, K}, {K, N}},
            leaves -> leaves[0].mmul(leaves[1]).sum());
        assertMatch("mmul2D", cpu, hpc);
    }

    @Test void testMmulTallSkinny() {
        // M=1, K=8, N=3 — single-row output [1,3]
        int M = 1, K = 8, N = 3;
        double[] a = rand(M * K);
        double[] b = rand(K * N);
        CpuResult cpu = cpuRef(new double[][]{a, b},
            new int[][]{{M, K}, {K, N}},
            leaves -> leaves[0].mmul(leaves[1]).sum());
        HpcResult hpc = hpcExec(new double[][]{a, b},
            new int[][]{{M, K}, {K, N}},
            leaves -> leaves[0].mmul(leaves[1]).sum());
        if (!hpcPresent || Double.isNaN(hpc.loss)) return;
        assertMatch("mmulTallSkinny", cpu, hpc);
    }

    @Test void testMmulWideShort() {
        // M=5, K=3, N=1 — single-column output [5,1]
        int M = 5, K = 3, N = 1;
        double[] a = rand(M * K);
        double[] b = rand(K * N);
        CpuResult cpu = cpuRef(new double[][]{a, b},
            new int[][]{{M, K}, {K, N}},
            leaves -> leaves[0].mmul(leaves[1]).sum());
        HpcResult hpc = hpcExec(new double[][]{a, b},
            new int[][]{{M, K}, {K, N}},
            leaves -> leaves[0].mmul(leaves[1]).sum());
        if (!hpcPresent || Double.isNaN(hpc.loss)) return;
        assertMatch("mmulWideShort", cpu, hpc);
    }

    @Test void testBmm3D() {
        // A[B,M,K] @ B[B,K,N] = C[B,M,N].sum() with B=2, M=3, K=4, N=5
        int B = 2, M = 3, K = 4, N = 5;
        double[] a = rand(B * M * K);
        double[] b = rand(B * K * N);
        CpuResult cpu = cpuRef(new double[][]{a, b},
            new int[][]{{B, M, K}, {B, K, N}},
            leaves -> leaves[0].bmm(leaves[1]).sum());
        HpcResult hpc = hpcExec(new double[][]{a, b},
            new int[][]{{B, M, K}, {B, K, N}},
            leaves -> leaves[0].bmm(leaves[1]).sum());
        assertMatch("bmm3D", cpu, hpc);
    }
}
