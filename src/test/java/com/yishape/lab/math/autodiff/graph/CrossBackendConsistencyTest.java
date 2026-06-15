package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;
import com.yishape.lab.math.compute.gpu.GpuSwitch;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;
import com.yishape.lab.math.compute.hpc.HpcSwitch;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Cross-backend semantic consistency for broadcast/expand/reduce ops.
 *
 * <p>Systematically enumerates rank combinations × dimension matching patterns
 * and compares CPU (ground truth), HPC, and GPU results. Designed to catch
 * backend-specific broadcast alignment bugs that single-backend tests miss.</p>
 *
 * <h3>Tested dimension patterns</h3>
 * <ul>
 *   <li>Rank-1 matching first dim of 2D output (DL left-aligned)</li>
 *   <li>Rank-1 matching last dim of 2D output (DL right-aligned)</li>
 *   <li>Rank-1 with square 2D output (ambiguous — needs shape info)</li>
 *   <li>Rank-1 matching first/last dim of 3D output</li>
 *   <li>Expand first dim vs last dim</li>
 *   <li>Broadcast with wide/skinny aspect ratios</li>
 * </ul>
 *
 * @since 0.5.0
 */
public class CrossBackendConsistencyTest {

    private static boolean gpuPresent;
    private static boolean hpcPresent;
    private static final Random RNG = new Random(0xBEEF_CAFE);
    private static final double GPU_LOSS_TOL = 1e-3;   // f32 GPU vs f64 CPU
    private static final double GPU_GRAD_TOL = 5e-4;
    private static final double HPC_LOSS_TOL = 1e-8;   // f64 HPC vs f64 CPU
    private static final double HPC_GRAD_TOL = 1e-7;

    @BeforeAll
    static void detect() {
        System.setProperty("yishape.gpu.minElements", "0");
        System.setProperty("yishape.hpc.minElements", "0");
        gpuPresent = GpuOptionalRuntime.isGpuAvailable();
        hpcPresent = HpcOptionalRuntime.isNativeRuntimeAvailable();
        System.out.println("[CrossBackend] GPU=" + (gpuPresent ? "present" : "absent")
            + " HPC=" + (hpcPresent ? "present" : "absent"));
    }

    @AfterEach
    void restore() {
        GpuSwitch.enable();
        HpcSwitch.enable();
    }

    // ── Data ──────────────────────────────────────────────────────────

    private static double[] rand(int n) {
        double[] d = new double[n];
        for (int i = 0; i < n; i++) d[i] = RNG.nextDouble() * 2.0 - 1.0;
        return d;
    }

    // ── Comparison engine ─────────────────────────────────────────────

    @FunctionalInterface
    interface TensorGraph { IDiffTensor build(RereDiffTensor[] leaves); }

    record BackendResult(double loss, double[][] grads) {}

    private static BackendResult cpuRef(double[][] leafData, int[][] shapes, TensorGraph fn) {
        RereDiffTensor[] leaves = new RereDiffTensor[leafData.length];
        for (int i = 0; i < leafData.length; i++)
            leaves[i] = (RereDiffTensor) AD.leafTensor(leafData[i], shapes[i]);
        RereDiffTensor loss = (RereDiffTensor) fn.build(leaves);
        loss.backward();
        double[][] grads = new double[leaves.length][];
        for (int i = 0; i < leaves.length; i++)
            grads[i] = leaves[i].gradData().clone();
        return new BackendResult(loss.value().toDoubleArray()[0], grads);
    }

    private static BackendResult gpuExec(double[][] leafData, int[][] shapes, TensorGraph fn) {
        if (!gpuPresent) return new BackendResult(Double.NaN, null);
        RereDiffTensor[] leaves = new RereDiffTensor[leafData.length];
        for (int i = 0; i < leafData.length; i++)
            leaves[i] = (RereDiffTensor) AD.leafTensor(leafData[i], shapes[i]);
        RereDiffTensor loss = (RereDiffTensor) fn.build(leaves);
        double gpuLoss = GpuGraphExecutor.tryExecute(loss);
        if (Double.isNaN(gpuLoss)) return new BackendResult(Double.NaN, null);
        double[][] grads = new double[leaves.length][];
        for (int i = 0; i < leaves.length; i++)
            grads[i] = leaves[i].gradData().clone();
        return new BackendResult(gpuLoss, grads);
    }

    private static BackendResult hpcExec(double[][] leafData, int[][] shapes, TensorGraph fn) {
        if (!hpcPresent) return new BackendResult(Double.NaN, null);
        RereDiffTensor[] leaves = new RereDiffTensor[leafData.length];
        for (int i = 0; i < leafData.length; i++)
            leaves[i] = (RereDiffTensor) AD.leafTensor(leafData[i], shapes[i]);
        RereDiffTensor loss = (RereDiffTensor) fn.build(leaves);
        double hpcLoss = HpcGraphExecutor.tryExecute(loss);
        if (Double.isNaN(hpcLoss)) return new BackendResult(Double.NaN, null);
        double[][] grads = new double[leaves.length][];
        for (int i = 0; i < leaves.length; i++)
            grads[i] = leaves[i].gradData().clone();
        return new BackendResult(hpcLoss, grads);
    }

    // ── Assertion helpers ─────────────────────────────────────────────

    /** Compare GPU result to CPU, tolerating f32 precision. */
    private void assertGpuMatch(String label, BackendResult cpu, BackendResult gpu) {
        if (!gpuPresent) { assertTrue(Double.isNaN(gpu.loss), label + ": NaN expected absent"); return; }
        assertFalse(Double.isNaN(gpu.loss), () -> label + ": GPU loss NaN");
        assertEquals(cpu.loss, gpu.loss, GPU_LOSS_TOL,
            () -> String.format("%s GPU=%.8f CPU=%.8f diff=%.2e", label, gpu.loss, cpu.loss,
                Math.abs(gpu.loss - cpu.loss)));
        assertGradsMatch(label, cpu, gpu, GPU_GRAD_TOL);
    }

    /** Compare HPC result to CPU, tolerating f64 precision. */
    private void assertHpcMatch(String label, BackendResult cpu, BackendResult hpc) {
        if (!hpcPresent) { assertTrue(Double.isNaN(hpc.loss), label + ": NaN expected absent"); return; }
        assertFalse(Double.isNaN(hpc.loss), () -> label + ": HPC loss NaN");
        assertEquals(cpu.loss, hpc.loss, HPC_LOSS_TOL,
            () -> String.format("%s HPC=%.12f CPU=%.12f diff=%.2e", label, hpc.loss, cpu.loss,
                Math.abs(hpc.loss - cpu.loss)));
        assertGradsMatch(label, cpu, hpc, HPC_GRAD_TOL);
    }

    /** Compare HPC to GPU directly (both native, tolerating f32 GPU precision). */
    private void assertGpuHpcMatch(String label, BackendResult gpu, BackendResult hpc) {
        if (!gpuPresent || !hpcPresent) return;
        if (Double.isNaN(gpu.loss) || Double.isNaN(hpc.loss)) return;
        assertEquals(gpu.loss, hpc.loss, GPU_LOSS_TOL,
            () -> String.format("%s GPU=%.8f HPC=%.12f diff=%.2e", label, gpu.loss, hpc.loss,
                Math.abs(gpu.loss - hpc.loss)));
    }

    private void assertGradsMatch(String label, BackendResult ref, BackendResult tst, double tol) {
        assertEquals(ref.grads.length, tst.grads.length, label + ": grad count mismatch");
        for (int i = 0; i < ref.grads.length; i++) {
            final int gi = i;
            double[] rg = ref.grads[gi], tg = tst.grads[gi];
            assertEquals(rg.length, tg.length, label + ": grad[" + gi + "] len");
            for (int j = 0; j < rg.length; j++) {
                final int gj = j;
                double diff = Math.abs(rg[gj] - tg[gj]);
                double maxAbs = Math.max(Math.abs(rg[gj]), Math.abs(tg[gj]));
                double effectiveTol = Math.max(tol, maxAbs * 1e-4);
                assertTrue(diff <= effectiveTol,
                    () -> String.format("%s: grad[%d][%d] ref=%.12f test=%.12f diff=%.2e tol=%.2e",
                        label, gi, gj, rg[gj], tg[gj], diff, effectiveTol));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pattern 1: Rank-1 broadcast matching FIRST dim of 2D output
    //   [B,C] + [B]  — DL left-aligned
    //   Gradient: dL/d[B] = sum over dim-1 (C columns per row)
    // ═══════════════════════════════════════════════════════════════════

    @Test void testBroadcastAdd_1Dto2D_firstDim() {
        doBroadcast1Dto2D("add", false, 32, 10);  // wide: 32x10
    }
    @Test void testBroadcastAdd_1Dto2D_firstDim_wide() {
        doBroadcast1Dto2D("add", false, 64, 4);   // very wide: 64x4
    }
    @Test void testBroadcastAdd_1Dto2D_firstDim_tall() {
        doBroadcast1Dto2D("add", false, 4, 64);   // tall: 4x64
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pattern 2: Rank-1 broadcast matching LAST dim of 2D output
    //   [B,C] + [C]  — DL right-aligned
    //   Gradient: dL/d[C] = sum over dim-0 (B rows)
    // ═══════════════════════════════════════════════════════════════════

    @Test void testBroadcastAdd_1Dto2D_lastDim() {
        doBroadcast1Dto2D("add", true, 32, 10);
    }
    @Test void testBroadcastAdd_1Dto2D_lastDim_wide() {
        doBroadcast1Dto2D("add", true, 64, 4);
    }
    @Test void testBroadcastAdd_1Dto2D_lastDim_tall() {
        doBroadcast1Dto2D("add", true, 4, 64);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pattern 3: Rank-1 with SQUARE 2D output (ambiguous — needs shape info)
    //   [N,N] + [N]  — matches both first AND last dim
    // ═══════════════════════════════════════════════════════════════════

    @Test void testBroadcastAdd_1Dto2D_square_first() {
        doBroadcast1Dto2D("add", false, 8, 8);  // B matches first dim
    }
    @Test void testBroadcastAdd_1Dto2D_square_last() {
        doBroadcast1Dto2D("add", true, 8, 8);   // C matches last dim
    }

    private void doBroadcast1Dto2D(String op, boolean lastDim, int B, int C) {
        double[] d1 = rand(B * C);  // [B, C] — 2D
        double[] d2 = lastDim ? rand(C) : rand(B);  // [C] or [B] — 1D

        TensorGraph fn = lastDim
            ? leaves -> leaves[0].add(leaves[1]).sum()
            : leaves -> leaves[0].add(leaves[1]).sum();

        String label = String.format("add[%dx%d]+[%d]", B, C, lastDim ? C : B);
        BackendResult cpu = cpuRef(new double[][]{d1, d2},
            new int[][]{{B, C}, {lastDim ? C : B}}, fn);

        // HPC
        BackendResult hpc = hpcExec(new double[][]{d1, d2},
            new int[][]{{B, C}, {lastDim ? C : B}}, fn);
        assertHpcMatch(label + "/HPC", cpu, hpc);

        // GPU
        BackendResult gpu = gpuExec(new double[][]{d1, d2},
            new int[][]{{B, C}, {lastDim ? C : B}}, fn);
        assertGpuMatch(label + "/GPU", cpu, gpu);

        // Cross-native: GPU vs HPC
        assertGpuHpcMatch(label + "/GPUvsHPC", gpu, hpc);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pattern 4: Binary op variants — sub, mul, div with broadcast
    //   Tested on representative patterns only (not all rank combos)
    // ═══════════════════════════════════════════════════════════════════

    @Test void testBroadcastSub_firstDim() {
        int B = 32, C = 10;
        double[] d1 = rand(B * C), d2 = rand(B);
        TensorGraph fn = leaves -> leaves[0].sub(leaves[1]).sum();
        BackendResult cpu = cpuRef(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn);
        assertHpcMatch("sub[32x10]+[32]/HPC", cpu,
            hpcExec(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn));
        assertGpuMatch("sub[32x10]+[32]/GPU", cpu,
            gpuExec(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn));
    }

    @Test void testBroadcastMul_firstDim() {
        int B = 32, C = 10;
        double[] d1 = rand(B * C), d2 = rand(B);
        TensorGraph fn = leaves -> leaves[0].mul(leaves[1]).abs().sum();
        BackendResult cpu = cpuRef(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn);
        assertHpcMatch("mul[32x10]*[32]/HPC", cpu,
            hpcExec(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn));
        assertGpuMatch("mul[32x10]*[32]/GPU", cpu,
            gpuExec(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn));
    }

    @Test void testBroadcastDiv_lastDim() {
        int B = 32, C = 10;
        double[] d1 = rand(B * C);
        double[] d2 = new double[C];
        for (int i = 0; i < C; i++) d2[i] = RNG.nextDouble() * 9.0 + 1.0; // positive
        TensorGraph fn = leaves -> leaves[0].div(leaves[1]).abs().sum();
        BackendResult cpu = cpuRef(new double[][]{d1, d2}, new int[][]{{B, C}, {C}}, fn);
        assertHpcMatch("div[32x10]/[10]/HPC", cpu,
            hpcExec(new double[][]{d1, d2}, new int[][]{{B, C}, {C}}, fn));
        assertGpuMatch("div[32x10]/[10]/GPU", cpu,
            gpuExec(new double[][]{d1, d2}, new int[][]{{B, C}, {C}}, fn));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pattern 5: Rank-1 broadcast to 3D output
    //   [A,B,C] + [B] — matching middle dim (first dim of rank-1 maps to dim 1)
    // ═══════════════════════════════════════════════════════════════════

    @Test void testBroadcastAdd_1Dto3D_firstDim3D() {
        int A = 4, B = 8, C = 6;
        double[] d1 = rand(A * B * C);  // [4, 8, 6]
        double[] d2 = rand(A);           // [4] — matches first dim
        TensorGraph fn = leaves -> leaves[0].add(leaves[1]).sum();
        String label = String.format("add[%dx%dx%d]+[%d]", A, B, C, A);
        BackendResult cpu = cpuRef(new double[][]{d1, d2}, new int[][]{{A, B, C}, {A}}, fn);
        assertHpcMatch(label + "/HPC", cpu,
            hpcExec(new double[][]{d1, d2}, new int[][]{{A, B, C}, {A}}, fn));
        assertGpuMatch(label + "/GPU", cpu,
            gpuExec(new double[][]{d1, d2}, new int[][]{{A, B, C}, {A}}, fn));
    }

    @Test void testBroadcastAdd_1Dto3D_lastDim3D() {
        int A = 4, B = 8, C = 6;
        double[] d1 = rand(A * B * C);  // [4, 8, 6]
        double[] d2 = rand(C);           // [6] — matches last dim
        TensorGraph fn = leaves -> leaves[0].add(leaves[1]).sum();
        String label = String.format("add[%dx%dx%d]+[%d]", A, B, C, C);
        BackendResult cpu = cpuRef(new double[][]{d1, d2}, new int[][]{{A, B, C}, {C}}, fn);
        assertHpcMatch(label + "/HPC", cpu,
            hpcExec(new double[][]{d1, d2}, new int[][]{{A, B, C}, {C}}, fn));
        assertGpuMatch(label + "/GPU", cpu,
            gpuExec(new double[][]{d1, d2}, new int[][]{{A, B, C}, {C}}, fn));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pattern 6: Rank-2 broadcast to 3D output (trailing dims)
    //   [A,B,C] + [B,C] — rank-2 matches trailing dims (valid broadcast)
    //   [A,B,C] + [1,B,C] — explicit unsqueeze
    // ═══════════════════════════════════════════════════════════════════

    @Test void testBroadcastAdd_2Dto3D_trailing() {
        int A = 4, B = 8, C = 6;
        double[] d1 = rand(A * B * C);   // [4, 8, 6]
        double[] d2 = rand(B * C);        // [8, 6] — trailing dims match
        TensorGraph fn = leaves -> leaves[0].add(leaves[1]).sum();
        String label = String.format("add[%dx%dx%d]+[%dx%d]", A, B, C, B, C);
        BackendResult cpu = cpuRef(new double[][]{d1, d2}, new int[][]{{A, B, C}, {B, C}}, fn);
        assertHpcMatch(label + "/HPC", cpu,
            hpcExec(new double[][]{d1, d2}, new int[][]{{A, B, C}, {B, C}}, fn));
        assertGpuMatch(label + "/GPU", cpu,
            gpuExec(new double[][]{d1, d2}, new int[][]{{A, B, C}, {B, C}}, fn));
    }

    @Test void testBroadcastAdd_2Dto3D_viaUnsqueeze() {
        int A = 4, B = 8, C = 6;
        double[] d1 = rand(A * B * C);   // [4, 8, 6]
        double[] d2 = rand(B * C);        // [8, 6] → unsqueeze(0) → [1, 8, 6]
        TensorGraph fn = leaves -> leaves[0].add(leaves[1].unsqueeze(0)).sum();
        String label = String.format("add[%dx%dx%d]+unsqueeze(0)[%dx%d]", A, B, C, B, C);
        BackendResult cpu = cpuRef(new double[][]{d1, d2}, new int[][]{{A, B, C}, {B, C}}, fn);
        assertHpcMatch(label + "/HPC", cpu,
            hpcExec(new double[][]{d1, d2}, new int[][]{{A, B, C}, {B, C}}, fn));
        assertGpuMatch(label + "/GPU", cpu,
            gpuExec(new double[][]{d1, d2}, new int[][]{{A, B, C}, {B, C}}, fn));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pattern 7: Expand ops
    //   expand([C], 0) → [1,C] → expand(B,C)  — first dim expand
    //   expand([B], 1) → [B,1] → expand(B,C)  — last dim expand
    // ═══════════════════════════════════════════════════════════════════

    @Test void testExpand_firstDim() {
        int B = 3, C = 5;
        double[] d = rand(C);  // [C]
        TensorGraph fn = leaves -> leaves[0].unsqueeze(0).expand(B, C).sum();
        String label = String.format("expand[%d]→unsqueeze(0)→expand(%d,%d)", C, B, C);
        BackendResult cpu = cpuRef(new double[][]{d}, new int[][]{{C}}, fn);
        assertHpcMatch(label + "/HPC", cpu,
            hpcExec(new double[][]{d}, new int[][]{{C}}, fn));
        assertGpuMatch(label + "/GPU", cpu,
            gpuExec(new double[][]{d}, new int[][]{{C}}, fn));
    }

    @Test void testExpand_lastDim() {
        int B = 3, C = 5;
        double[] d = rand(B);  // [B]
        TensorGraph fn = leaves -> leaves[0].unsqueeze(1).expand(B, C).sum();
        String label = String.format("expand[%d]→unsqueeze(1)→expand(%d,%d)", B, B, C);
        BackendResult cpu = cpuRef(new double[][]{d}, new int[][]{{B}}, fn);
        assertHpcMatch(label + "/HPC", cpu,
            hpcExec(new double[][]{d}, new int[][]{{B}}, fn));
        assertGpuMatch(label + "/GPU", cpu,
            gpuExec(new double[][]{d}, new int[][]{{B}}, fn));
    }

    @Test void testExpand_to3D() {
        int B = 2, C = 3, D = 4;
        double[] d = rand(C * D);  // [3, 4] → [2, 3, 4]
        TensorGraph fn = leaves -> leaves[0].unsqueeze(0).expand(B, C, D).sum();
        String label = String.format("expand[%dx%d]→unsqueeze(0)→expand(%d,%d,%d)", C, D, B, C, D);
        BackendResult cpu = cpuRef(new double[][]{d}, new int[][]{{C, D}}, fn);
        assertHpcMatch(label + "/HPC", cpu,
            hpcExec(new double[][]{d}, new int[][]{{C, D}}, fn));
        assertGpuMatch(label + "/GPU", cpu,
            gpuExec(new double[][]{d}, new int[][]{{C, D}}, fn));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pattern 8: Broadcast with unequal aspect ratios (stress test)
    //   Very wide ([256,2]), very tall ([2,256]), prime dims
    // ═══════════════════════════════════════════════════════════════════

    @Test void testBroadcastAdd_primeDims() {
        int B = 17, C = 23;  // primes — no factorization ambiguity
        double[] d1 = rand(B * C), d2 = rand(B);
        TensorGraph fn = leaves -> leaves[0].add(leaves[1]).sum();
        String label = String.format("add[%dx%d]+[%d]", B, C, B);
        BackendResult cpu = cpuRef(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn);
        assertHpcMatch(label + "/HPC", cpu,
            hpcExec(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn));
        assertGpuMatch(label + "/GPU", cpu,
            gpuExec(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn));
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pattern 9: Chain ops — broadcast then reduce
    //   This tests that backward through fused/higher-level ops is correct
    // ═══════════════════════════════════════════════════════════════════

    @Test void testChain_BroadcastThenPow() {
        int B = 16, C = 8;
        double[] d1 = rand(B * C), d2 = rand(B);
        TensorGraph fn = leaves -> leaves[0].add(leaves[1]).pow(2.0).sum();
        BackendResult cpu = cpuRef(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn);
        assertHpcMatch("chain(+,pow)[16x8]+[16]/HPC", cpu,
            hpcExec(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn));
        assertGpuMatch("chain(+,pow)[16x8]+[16]/GPU", cpu,
            gpuExec(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn));
    }

    @Test void testChain_BroadcastThenSumDim() {
        int B = 8, C = 12;
        double[] d1 = rand(B * C), d2 = rand(B);
        // a + b → sum over dim 1 → sum (add-sum chain)
        TensorGraph fn = leaves -> leaves[0].add(leaves[1]).sum(1, true).sum();
        BackendResult cpu = cpuRef(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn);
        assertHpcMatch("chain(+,sumDim)[8x12]+[8]/HPC", cpu,
            hpcExec(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn));
        assertGpuMatch("chain(+,sumDim)[8x12]+[8]/GPU", cpu,
            gpuExec(new double[][]{d1, d2}, new int[][]{{B, C}, {B}}, fn));
    }
}
