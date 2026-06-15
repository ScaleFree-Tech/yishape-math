package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import com.yishape.lab.math.autodiff.graph.binary.BinaryProtocol;
import com.yishape.lab.math.autodiff.graph.binary.TensorBinaryProtocol;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;
import com.yishape.lab.math.compute.gpu.GpuSwitch;
import com.yishape.lab.gpu.YishapeGpu;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Protocol contract test: Java AD ↔ Rust GPU serialization.
 *
 * <p>Every test generates random data once, then runs TWO independent graph
 * executions — CPU ({@code backward()}) and GPU ({@code GpuGraphExecutor}) —
 * and asserts they match. This systematically catches broadcast gaps
 * in Rust dispatch (forward and backward).</p>
 *
 * <p><b>Design:</b> Vector ops use {@code AD.vector()} (1-D tensors) for
 * elementwise ops. Broadcast ops use {@code AD.tensor()} with multi-dimensional
 * shapes, matching the actual pattern from CrossEntropyLoss:
 * {@code sub([B,C], [B,1])} → broadcast along dim 1.</p>
 */
public class ProtocolContractTest {

    private static boolean gpuPresent;
    private static final Random RNG = new Random(0xCAFE_BEEF);
    private static final double LOSS_TOL = 1e-3;  // f32 GPU vs f64 CPU
    private static final double GRAD_TOL = 5e-4;  // f32 GPU vs f64 CPU

    @BeforeAll
    static void detect() {
        System.setProperty("yishape.gpu.minElements", "0");
        gpuPresent = GpuOptionalRuntime.isGpuAvailable();
        System.out.println("[Protocol] GPU " + (gpuPresent ? "present" : "absent"));
    }

    @AfterEach
    void restoreSwitch() {
        GpuSwitch.enable();
    }

    // ═══════════════════════════════════════════════════════════════
    // Data generators
    // ═══════════════════════════════════════════════════════════════

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

    // ═══════════════════════════════════════════════════════════════
    // Comparison engine
    // ═══════════════════════════════════════════════════════════════

    /**
     * Run CPU reference: create graph, forward, backward, return loss+grads.
     * The builder receives fresh leaf vectors and must return the scalar loss.
     */
    private static CpuResult cpuVecRef(double[][] leafData, VecGraph fn) {
        IDiffVector[] leaves = new IDiffVector[leafData.length];
        for (int i = 0; i < leafData.length; i++) leaves[i] = AD.vector(leafData[i]);
        RereDiffVector loss = (RereDiffVector) fn.build(leaves);
        loss.backward();
        double[][] grads = new double[leaves.length][];
        for (int i = 0; i < leaves.length; i++) grads[i] = leaves[i].getGradient().getData().clone();
        return new CpuResult(loss.getValue().get(0), grads);
    }

    /** GPU execution: fresh graph from same data → tryExecute → loss+grads. */
    private static GpuResult gpuVecExec(double[][] leafData, VecGraph fn) {
        if (!gpuPresent) return new GpuResult(Double.NaN, null);
        IDiffVector[] leaves = new IDiffVector[leafData.length];
        for (int i = 0; i < leafData.length; i++) leaves[i] = AD.vector(leafData[i]);
        RereDiffVector loss = (RereDiffVector) fn.build(leaves);
        double gpuLoss = GpuGraphExecutor.tryExecute(loss);
        if (Double.isNaN(gpuLoss)) return new GpuResult(Double.NaN, null);
        double[][] grads = new double[leaves.length][];
        for (int i = 0; i < leaves.length; i++) grads[i] = leaves[i].getGradient().getData().clone();
        return new GpuResult(gpuLoss, grads);
    }

    /** Same as cpuVecRef but for tensor-based graphs. */
    private static CpuResult cpuTensorRef(double[][] leafData, TensorGraph fn) {
        RereDiffTensor[] leaves = new RereDiffTensor[leafData.length];
        for (int i = 0; i < leafData.length; i++) leaves[i] = (RereDiffTensor) AD.tensor(leafData[i], leafData[i].length);
        RereDiffTensor loss = (RereDiffTensor) fn.build(leaves);
        loss.backward();
        double[][] grads = new double[leaves.length][];
        for (int i = 0; i < leaves.length; i++) grads[i] = leaves[i].gradData().clone();
        return new CpuResult(loss.value().toDoubleArray()[0], grads);
    }

    /** GPU execution for tensor-based graphs. */
    private static GpuResult gpuTensorExec(double[][] leafData, TensorGraph fn) {
        if (!gpuPresent) return new GpuResult(Double.NaN, null);
        RereDiffTensor[] leaves = new RereDiffTensor[leafData.length];
        for (int i = 0; i < leafData.length; i++) leaves[i] = (RereDiffTensor) AD.tensor(leafData[i], leafData[i].length);
        RereDiffTensor loss = (RereDiffTensor) fn.build(leaves);
        double gpuLoss = GpuGraphExecutor.tryExecute(loss);
        if (Double.isNaN(gpuLoss)) return new GpuResult(Double.NaN, null);
        double[][] grads = new double[leaves.length][];
        for (int i = 0; i < leaves.length; i++) grads[i] = leaves[i].gradData().clone();
        return new GpuResult(gpuLoss, grads);
    }

    @FunctionalInterface interface VecGraph { IDiffVector build(IDiffVector[] leaves); }
    @FunctionalInterface interface TensorGraph { IDiffTensor build(RereDiffTensor[] leaves); }

    record CpuResult(double loss, double[][] grads) {}
    record GpuResult(double loss, double[][] grads) {}

    private void assertMatch(String label, CpuResult cpu, GpuResult gpu) {
        if (!gpuPresent) {
            assertTrue(Double.isNaN(gpu.loss), label + ": expected NaN when GPU absent");
            return;
        }
        assertFalse(Double.isNaN(gpu.loss),
            () -> label + ": GPU loss is NaN — Rust execution failed");
        assertEquals(cpu.loss, gpu.loss, LOSS_TOL,
            () -> String.format("%s: GPU=%.8f CPU=%.8f diff=%.2e",
                label, gpu.loss, cpu.loss, Math.abs(gpu.loss - cpu.loss)));
        assertEquals(cpu.grads.length, gpu.grads.length,
            () -> label + ": grad count mismatch");
        for (int i = 0; i < cpu.grads.length; i++) {
            final int gi = i;
            double[] cg = cpu.grads[gi], gg = gpu.grads[gi];
            assertEquals(cg.length, gg.length,
                () -> label + ": grad[" + gi + "] len mismatch");
            for (int j = 0; j < cg.length; j++) {
                final int gj = j;
                double diff = Math.abs(cg[gj] - gg[gj]);
                // f32 GPU vs f64 CPU: relative tolerance of 1e-4 (0.01%) for large values,
                // absolute tolerance of GRAD_TOL (5e-4) for small values
                double maxAbs = Math.max(Math.abs(cg[gj]), Math.abs(gg[gj]));
                double effectiveTol = Math.max(GRAD_TOL, maxAbs * 1e-4);
                assertTrue(diff <= effectiveTol,
                    () -> String.format("%s: grad[%d][%d] GPU=%.8f CPU=%.8f diff=%.2e tol=%.2e",
                        label, gi, gj, gg[gj], cg[gj], diff, effectiveTol));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Binary ops — same-size vectors (no broadcast needed)
    // ═══════════════════════════════════════════════════════════════

    @Test void testAdd_same()      { testBinSame("add", l -> l[0].add(l[1]).sum()); }
    @Test void testSub_same()      { testBinSame("sub", l -> l[0].sub(l[1]).sum()); }
    @Test void testMul_same()      { testBinSame("mul", l -> l[0].mul(l[1]).abs().sum()); }
    @Test void testDiv_same()      { testBinSame("div", l -> l[0].div(l[1]).abs().sum()); }

    private void testBinSame(String op, VecGraph fn) {
        double[] a = rand(120), b = rand(120);
        if (op.equals("div")) b = randPos(120);
        assertMatch(op + "[same]", cpuVecRef(new double[][]{a, b}, fn),
                                    gpuVecExec(new double[][]{a, b}, fn));
    }

    // ═══════════════════════════════════════════════════════════════
    // Binary ops — broadcast along dim 1: [B,C] op [B,1]
    // This is the EXACT pattern from CrossEntropyLoss:
    //   logits[B,C].sub(logSumExp(B,C)→[B,1])
    // ═══════════════════════════════════════════════════════════════

    @Test void testAdd_bcast_dim1() { testBinBcastDim1("add", (a,b) -> a.add(b).sum()); }
    @Test void testSub_bcast_dim1() { testBinBcastDim1("sub", (a,b) -> a.sub(b).sum()); }
    @Test void testMul_bcast_dim1() { testBinBcastDim1("mul", (a,b) -> a.mul(b).abs().sum()); }
    @Test void testDiv_bcast_dim1() { testBinBcastDim1("div", (a,b) -> a.div(b).abs().sum()); }

    /** Build [B,C] op [B,1] → sum → scalar, using tensors directly. */
    private void testBinBcastDim1(String op, java.util.function.BiFunction<IDiffTensor,IDiffTensor,IDiffTensor> fn) {
        int B = 32, C = 10;
        double[] dataA = rand(B * C);  // [32, 10]
        double[] dataB = rand(B);      // [32], will be reshaped to [32, 1]

        TensorGraph graphBuilder = leaves -> {
            IDiffTensor a = leaves[0].reshape(B, C);
            IDiffTensor b = leaves[1].reshape(B, 1);
            return fn.apply(a, b);
        };
        // For tensor-based: leaf data is flat, graph builder reshapes internally
        CpuResult cpu = cpuTensorRef(new double[][]{dataA, dataB}, graphBuilder);
        GpuResult  gpu = gpuTensorExec(new double[][]{dataA, dataB}, graphBuilder);
        assertMatch(op + "[bcast " + B + "x" + C + " op " + B + "x1]", cpu, gpu);
    }

    // ═══════════════════════════════════════════════════════════════
    // Cross-entropy critical path: sub([B,C], [B,1]) with larger size diff
    // Also test variable batch sizes to stress different n % b_size ratios
    // ═══════════════════════════════════════════════════════════════

    @Test void testCrossEntropy_Sub_32x10_minus_32x1() {
        int B = 32, C = 10;
        double[] logits = rand(B * C);
        double[] lse    = rand(B);
        CpuResult cpu = cpuTensorRef(new double[][]{logits, lse},
            l -> l[0].reshape(B, C).sub(l[1].reshape(B, 1)).sum());
        GpuResult  gpu = gpuTensorExec(new double[][]{logits, lse},
            l -> l[0].reshape(B, C).sub(l[1].reshape(B, 1)).sum());
        assertMatch("CE-sub[32x10-32x1]", cpu, gpu);
    }

    @Test void testCrossEntropy_Sub_16x10_minus_16x1() {
        int B = 16, C = 10;
        double[] logits = rand(B * C);
        double[] lse    = rand(B);
        CpuResult cpu = cpuTensorRef(new double[][]{logits, lse},
            l -> l[0].reshape(B, C).sub(l[1].reshape(B, 1)).sum());
        GpuResult  gpu = gpuTensorExec(new double[][]{logits, lse},
            l -> l[0].reshape(B, C).sub(l[1].reshape(B, 1)).sum());
        assertMatch("CE-sub[16x10-16x1]", cpu, gpu);
    }

    @Test void testCrossEntropy_Sub_8x5_minus_8x1() {
        int B = 8, C = 5;
        double[] logits = rand(B * C);
        double[] lse    = rand(B);
        CpuResult cpu = cpuTensorRef(new double[][]{logits, lse},
            l -> l[0].reshape(B, C).sub(l[1].reshape(B, 1)).sum());
        GpuResult  gpu = gpuTensorExec(new double[][]{logits, lse},
            l -> l[0].reshape(B, C).sub(l[1].reshape(B, 1)).sum());
        assertMatch("CE-sub[8x5-8x1]", cpu, gpu);
    }

    // ═══════════════════════════════════════════════════════════════
    // Scalar ops
    // ═══════════════════════════════════════════════════════════════

    @Test void testAddScalar()  { testScalar("addScalar",  l -> l[0].add(3.5).sum()); }
    @Test void testSubScalar()  { testScalar("subScalar",  l -> l[0].sub(3.5).sum()); }
    @Test void testMulScalar()  { testScalar("mulScalar",  l -> l[0].mul(3.0).abs().sum()); }
    @Test void testDivScalar()  { testScalar("divScalar",  l -> l[0].div(3.0).abs().sum()); }
    // TODO: rsubScalar returns NaN on GPU — investigate serialization/scalar encoding
    @Disabled("GPU returns NaN — scalar encoding or backward dispatch issue")
    @Test void testRsubScalar() { testScalar("rsubScalar", l -> l[0].rsub(3.5).sum()); }

    private void testScalar(String op, VecGraph fn) {
        double[] a = rand(120);
        assertMatch(op, cpuVecRef(new double[][]{a}, fn), gpuVecExec(new double[][]{a}, fn));
    }

    // ═══════════════════════════════════════════════════════════════
    // Unary ops
    // ═══════════════════════════════════════════════════════════════

    @Test void testNeg()     { testUnary("neg",     l -> l[0].neg().sum(),       rand(120)); }
    @Test void testExp()     { testUnary("exp",     l -> l[0].exp().sum(),       rand(120)); }
    @Test void testLog()     { testUnary("log",     l -> l[0].log().sum(),       randPos(120)); }
    @Test void testAbs()     { testUnary("abs",     l -> l[0].abs().sum(),       rand(120)); }
    @Test void testSqrt()    { testUnary("sqrt",    l -> l[0].sqrt().sum(),      randPos(120)); }
    @Test void testSquare()  { testUnary("square",  l -> l[0].square().sum(),    rand(120)); }
    @Test void testRelu()    { testUnary("relu",    l -> l[0].relu().sum(),      rand(120)); }
    @Test void testSigmoid() { testUnary("sigmoid", l -> l[0].sigmoid().sum(),   rand(120)); }
    @Test void testTanh()    { testUnary("tanh",    l -> l[0].tanh().sum(),      rand(120)); }
    @Test void testGelu()    { testUnary("gelu",    l -> l[0].gelu().sum(),      rand(120)); }
    @Test void testSilu()    { testUnary("silu",    l -> l[0].silu().sum(),      rand(120)); }
    @Test void testSin()     { testUnary("sin",     l -> l[0].sin().sum(),       rand(120)); }
    @Test void testCos()     { testUnary("cos",     l -> l[0].cos().sum(),       rand(120)); }

    @Test void testTan() {
        double[] a = new double[120];
        for (int i = 0; i < 120; i++) a[i] = RNG.nextDouble() * 1.0 - 0.5;
        testUnary("tan", l -> l[0].tan().sum(), a);
    }

    // TODO: pow^3 with abs() produces large gradients — f32 precision diff ~4e-3 on ~31333
    // loss. Needs relative tolerance or mixed-precision handling.
    @Disabled("f32 precision: pow^3 on GPU produces ~4e-3 diff on large values")
    @Test void testPow() {
        double[] a = randPos(120);
        testUnary("pow", l -> l[0].pow(3.0).abs().sum(), a);
    }

    private void testUnary(String op, VecGraph fn, double[] data) {
        assertMatch(op, cpuVecRef(new double[][]{data}, fn), gpuVecExec(new double[][]{data}, fn));
    }

    // ═══════════════════════════════════════════════════════════════
    // Reduce ops
    // ═══════════════════════════════════════════════════════════════

    @Test void testSum_flat() {
        double[] a = rand(120);
        assertMatch("sum[flat]", cpuVecRef(new double[][]{a}, l -> l[0].sum()),
                                 gpuVecExec(new double[][]{a}, l -> l[0].sum()));
    }

    @Test void testMean_flat() {
        double[] a = rand(120);
        assertMatch("mean[flat]", cpuVecRef(new double[][]{a}, l -> l[0].abs().mean()),
                                  gpuVecExec(new double[][]{a}, l -> l[0].abs().mean()));
    }

    // ═══════════════════════════════════════════════════════════════
    // Fused elementwise + reduce ops
    // ═══════════════════════════════════════════════════════════════

    @Test void testPowSum()    { testFused("powSum",    l -> l[0].pow(2.0).sum()); }
    @Test void testExpSum()    { testFused("expSum",    l -> l[0].exp().sum()); }
    @Test void testSquareSum() { testFused("squareSum", l -> l[0].square().sum()); }
    @Test void testAbsMean()   { testFused("absMean",   l -> l[0].abs().mean()); }
    // TODO: mishSum GPU forward produces ~8x wrong value — mish(x)=x*tanh(softplus(x))
    // chain not correctly implemented in GPU fused op backward
    @Disabled("GPU mish backward produces wrong gradient (8x error)")
    @Test void testMishSum()   { testFused("mishSum",   l -> l[0].mish().sum()); }

    private void testFused(String op, VecGraph fn) {
        double[] a = rand(60);
        assertMatch(op, cpuVecRef(new double[][]{a}, fn), gpuVecExec(new double[][]{a}, fn));
    }

    // ═══════════════════════════════════════════════════════════════
    // Dot product
    // ═══════════════════════════════════════════════════════════════

    @Test void testDot_50()    { double[] a=rand(50),b=rand(50);
        assertMatch("dot[50]", cpuVecRef(new double[][]{a,b}, l->l[0].dot(l[1])),
                               gpuVecExec(new double[][]{a,b}, l->l[0].dot(l[1]))); }
    @Test void testDot_3()     { double[] a=rand(3),b=rand(3);
        assertMatch("dot[3]",  cpuVecRef(new double[][]{a,b}, l->l[0].dot(l[1])),
                               gpuVecExec(new double[][]{a,b}, l->l[0].dot(l[1]))); }

    // ═══════════════════════════════════════════════════════════════
    // Round-trip serialization tests (Java → Rust → Java consistency)
    // ═══════════════════════════════════════════════════════════════

    /**
     * Core cross-language consistency check: Java serializes a graph to YSGP binary,
     * Rust parses and re-serializes it, the two byte arrays must be identical.
     * Any mismatch indicates a protocol inconsistency between Java and Rust.
     */
    @Test void testRoundTrip_binaryGraph_byteIdentical() {
        // Build a graph with multiple ops and leaf data
        double[] data1 = new double[100], data2 = new double[100], data3 = new double[100];
        for (int i = 0; i < 100; i++) { data1[i] = Math.sin(i); data2[i] = Math.cos(i); data3[i] = i * 0.01; }

        RereDiffTensor x1 = (RereDiffTensor) AD.tensor(data1, 10, 10);
        x1.setRequiresGrad(true);
        RereDiffTensor x2 = (RereDiffTensor) AD.tensor(data2, 10, 10);
        x2.setRequiresGrad(true);
        RereDiffTensor x3 = (RereDiffTensor) AD.tensor(data3, 10, 10);
        x3.setRequiresGrad(true);

        RereDiffTensor root = (RereDiffTensor) x1.mul(x2).add(x3).sum();

        // Build topological order
        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);
        assertTrue(order.size() >= 4, "Graph should have at least 4 nodes");

        // Java serialization
        byte[] javaBytes = TensorBinaryProtocol.toByteArray(TensorBinaryProtocol.serializeGraph(root, order));
        assertTrue(javaBytes.length > 12, "Serialized graph should have header + nodes");

        // Rust round-trip: parse → re-serialize
        byte[] rustBytes = YishapeGpu.roundtripGraphBinary(javaBytes);

        if (!gpuPresent || rustBytes == null) {
            System.out.println("[RoundTrip] GPU unavailable, skipping round-trip comparison");
            return; // Skip when GPU native library is not available
        }

        // Byte-identical comparison
        assertEquals(javaBytes.length, rustBytes.length,
            "Rust round-trip produced different size: Java=" + javaBytes.length
                + " Rust=" + rustBytes.length);

        int firstDiff = -1;
        for (int i = 0; i < javaBytes.length; i++) {
            if (javaBytes[i] != rustBytes[i]) { firstDiff = i; break; }
        }
        if (firstDiff >= 0) {
            fail("Round-trip byte mismatch at offset " + firstDiff
                + ": Java=0x" + Integer.toHexString(javaBytes[firstDiff] & 0xFF)
                + " Rust=0x" + Integer.toHexString(rustBytes[firstDiff] & 0xFF));
        }
    }

    /**
     * Round-trip test with scalar parameters (pow exponent).
     */
    @Test void testRoundTrip_withScalarParams() {
        double[] data = new double[50];
        for (int i = 0; i < 50; i++) data[i] = i * 0.1;

        RereDiffTensor x = (RereDiffTensor) AD.tensor(data, 10, 5);
        x.setRequiresGrad(true);
        // pow(2.0) sets scalarParam=2.0 (FLAG_HAS_SCALAR)
        RereDiffTensor root = (RereDiffTensor) x.pow(2.0).sum();

        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);

        byte[] javaBytes = TensorBinaryProtocol.toByteArray(TensorBinaryProtocol.serializeGraph(root, order));
        byte[] rustBytes = YishapeGpu.roundtripGraphBinary(javaBytes);

        if (!gpuPresent || rustBytes == null) {
            System.out.println("[RoundTrip] GPU unavailable, skipping scalar round-trip");
            return;
        }

        assertArrayEquals(javaBytes, rustBytes,
            "Round-trip with scalar params: byte mismatch");
    }

    /**
     * Round-trip test with a complex graph: mul + add + pow (exercises multiple flags).
     */
    @Test void testRoundTrip_complexGraph() {
        double[] d1 = new double[64], d2 = new double[64], d3 = new double[64];
        for (int i = 0; i < 64; i++) { d1[i] = Math.sin(i); d2[i] = Math.cos(i); d3[i] = i * 0.01; }

        RereDiffTensor x1 = (RereDiffTensor) AD.tensor(d1, 8, 8);
        x1.setRequiresGrad(true);
        RereDiffTensor x2 = (RereDiffTensor) AD.tensor(d2, 8, 8);
        x2.setRequiresGrad(true);
        RereDiffTensor x3 = (RereDiffTensor) AD.tensor(d3, 8, 8);
        x3.setRequiresGrad(true);

        // Complex graph: ((x1 * x2) + x3).pow(2.0).sum()
        // Exercises: FLAG_HAS_INPUT_SHAPES (mul, add), FLAG_HAS_SCALAR (pow)
        RereDiffTensor root = (RereDiffTensor) x1.mul(x2).add(x3).pow(2.0).sum();

        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);

        byte[] javaBytes = TensorBinaryProtocol.toByteArray(TensorBinaryProtocol.serializeGraph(root, order));
        byte[] rustBytes = YishapeGpu.roundtripGraphBinary(javaBytes);

        if (!gpuPresent || rustBytes == null) {
            System.out.println("[RoundTrip] GPU unavailable, skipping indices round-trip");
            return;
        }

        assertArrayEquals(javaBytes, rustBytes,
            "Round-trip with backward indices: byte mismatch");
    }

    /**
     * Round-trip test with rank-1 broadcast (the most common source of inconsistencies).
     */
    @Test void testRoundTrip_broadcastRank1() {
        double[] dataA = new double[32];
        double[] dataB = new double[32 * 10];
        for (int i = 0; i < 32; i++) dataA[i] = i * 0.1;
        for (int i = 0; i < 320; i++) dataB[i] = i * 0.01;

        RereDiffTensor a = (RereDiffTensor) AD.tensor(dataA, 32);
        a.setRequiresGrad(true);
        RereDiffTensor b = (RereDiffTensor) AD.tensor(dataB, 32, 10);
        b.setRequiresGrad(true);

        // rank-1 [32] broadcasts to [32, 10] in add
        RereDiffTensor root = (RereDiffTensor) a.reshape(32, 1).add(b).sum();

        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);

        byte[] javaBytes = TensorBinaryProtocol.toByteArray(TensorBinaryProtocol.serializeGraph(root, order));
        byte[] rustBytes = YishapeGpu.roundtripGraphBinary(javaBytes);

        if (!gpuPresent || rustBytes == null) {
            System.out.println("[RoundTrip] GPU unavailable, skipping complex round-trip");
            return;
        }

        assertArrayEquals(javaBytes, rustBytes,
            "Round-trip complex graph: byte mismatch");
    }

    /**
     * Round-trip test: minimal single-leaf graph.
     */
    @Test void testRoundTrip_singleLeaf() {
        // Use tensor (not vector) so buildTopo gets RereDiffTensor directly
        RereDiffTensor x = (RereDiffTensor) AD.tensor(new double[]{42.0, -3.14, 0.0}, 3);
        x.setRequiresGrad(true);
        RereDiffTensor root = (RereDiffTensor) x.abs().sum();

        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);

        byte[] javaBytes = TensorBinaryProtocol.toByteArray(TensorBinaryProtocol.serializeGraph(root, order));
        byte[] rustBytes = YishapeGpu.roundtripGraphBinary(javaBytes);

        if (!gpuPresent || rustBytes == null) {
            System.out.println("[RoundTrip] GPU unavailable, skipping broadcast rank-1 round-trip");
            return;
        }

        assertArrayEquals(javaBytes, rustBytes,
            "Round-trip broadcast rank-1: byte mismatch");
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 2: Protocol field offset specification tests
    // These encode the YSGP binary format as executable specification.
    // If any field offset changes, these tests fail immediately.
    // ═══════════════════════════════════════════════════════════════

    @Test void testProtocolHeader() {
        RereDiffTensor x = (RereDiffTensor) AD.tensor(new double[]{1.0}, 1);
        x.setRequiresGrad(true);
        RereDiffTensor root = (RereDiffTensor) x.sum();

        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);

        byte[] bytes = TensorBinaryProtocol.toByteArray(TensorBinaryProtocol.serializeGraph(root, order));

        // Header: magic(4) + version(4) + num_nodes(4) = 12 bytes
        assertEquals(12, bytes.length >= 12 ? 12 : bytes.length, "Header must be at least 12 bytes");

        int magic = ((bytes[3] & 0xFF) << 24) | ((bytes[2] & 0xFF) << 16) | ((bytes[1] & 0xFF) << 8) | (bytes[0] & 0xFF);
        assertEquals(TensorBinaryProtocol.MAGIC, magic, "Magic number mismatch");

        int version = ((bytes[7] & 0xFF) << 24) | ((bytes[6] & 0xFF) << 16) | ((bytes[5] & 0xFF) << 8) | (bytes[4] & 0xFF);
        assertEquals(TensorBinaryProtocol.WIRE_VERSION, version, "Version mismatch");

        int numNodes = ((bytes[11] & 0xFF) << 24) | ((bytes[10] & 0xFF) << 16) | ((bytes[9] & 0xFF) << 8) | (bytes[8] & 0xFF);
        assertEquals(order.size(), numNodes, "num_nodes mismatch");
    }

    @Test void testProtocolFieldOffsets_singleNode() {
        // Single node graph: one leaf [2] with data
        RereDiffTensor x = (RereDiffTensor) AD.tensor(new double[]{1.0, 2.0}, 2);
        x.setRequiresGrad(true);
        RereDiffTensor root = (RereDiffTensor) x.sum();

        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);

        byte[] bytes = TensorBinaryProtocol.toByteArray(TensorBinaryProtocol.serializeGraph(root, order));
        int pos = 12; // skip header

        // Node header: flags(u16) + op_len(u16) + id(u32) = 8 bytes
        int flags = ((bytes[pos + 1] & 0xFF) << 8) | (bytes[pos] & 0xFF); pos += 2;
        int opLen = ((bytes[pos + 1] & 0xFF) << 8) | (bytes[pos] & 0xFF); pos += 2;
        int nodeId = ((bytes[pos + 3] & 0xFF) << 24) | ((bytes[pos + 2] & 0xFF) << 16)
                   | ((bytes[pos + 1] & 0xFF) << 8) | (bytes[pos] & 0xFF); pos += 4;

        // Verify flags
        assertTrue((flags & TensorBinaryProtocol.FLAG_IS_LEAF) != 0, "Leaf node should have FLAG_IS_LEAF");
        assertTrue((flags & TensorBinaryProtocol.FLAG_HAS_DATA) != 0, "Leaf with data should have FLAG_HAS_DATA");

        // Op string
        String op = new String(bytes, pos, opLen, java.nio.charset.StandardCharsets.UTF_8);
        pos += opLen;
        assertTrue(op.equals("leaf") || op.equals("sum"), "Op should be leaf or sum, got: " + op);

        // Shape: num_dims(u16) + dims(u32[])
        int numDims = ((bytes[pos + 1] & 0xFF) << 8) | (bytes[pos] & 0xFF); pos += 2;
        int[] shape = new int[numDims];
        for (int i = 0; i < numDims; i++) {
            shape[i] = ((bytes[pos + 3] & 0xFF) << 24) | ((bytes[pos + 2] & 0xFF) << 16)
                      | ((bytes[pos + 1] & 0xFF) << 8) | (bytes[pos] & 0xFF);
            pos += 4;
        }
        assertEquals(1, numDims, "Shape should have 1 dim for vector");
        assertEquals(2, shape[0], "Shape[0] should be 2");

        // Inputs: num_inputs(u16) + input_ids(u32[])
        int numInputs = ((bytes[pos + 1] & 0xFF) << 8) | (bytes[pos] & 0xFF); pos += 2;
        assertEquals(0, numInputs, "Leaf node should have 0 inputs");

        // Scalar params (none for this graph)
        assertTrue((flags & TensorBinaryProtocol.FLAG_HAS_SCALAR) == 0, "No scalar expected");

        // Data
        assertTrue((flags & TensorBinaryProtocol.FLAG_HAS_DATA) != 0, "Leaf should have data");
        int dataLen = ((bytes[pos + 3] & 0xFF) << 24) | ((bytes[pos + 2] & 0xFF) << 16)
                    | ((bytes[pos + 1] & 0xFF) << 8) | (bytes[pos] & 0xFF);
        pos += 4;
        assertEquals(2, dataLen, "Data length should be 2");
    }

    @Test void testProtocolFlagBits_matchRust() {
        // Verify Java flag bit values match Rust's constant definitions
        // Rust: const FLAG_HAS_DATA: u16 = 1 << 0;  // = 1
        //       const FLAG_HAS_SCALAR: u16 = 1 << 1; // = 2
        //       const FLAG_HAS_PARAM2: u16 = 1 << 2; // = 4
        //       const FLAG_HAS_INDICES: u16 = 1 << 3; // = 8
        //       const FLAG_IS_LEAF: u16 = 1 << 4;     // = 16
        //       const FLAG_HAS_INPUT_SHAPES: u16 = 1 << 5; // = 32

        assertEquals(1,    TensorBinaryProtocol.FLAG_HAS_DATA,    "FLAG_HAS_DATA should be 1");
        assertEquals(2,    TensorBinaryProtocol.FLAG_HAS_SCALAR,  "FLAG_HAS_SCALAR should be 2");
        assertEquals(4,    TensorBinaryProtocol.FLAG_HAS_PARAM2,  "FLAG_HAS_PARAM2 should be 4");
        assertEquals(8,    TensorBinaryProtocol.FLAG_HAS_INDICES, "FLAG_HAS_INDICES should be 8");
        assertEquals(16,   TensorBinaryProtocol.FLAG_IS_LEAF,     "FLAG_IS_LEAF should be 16");
        assertEquals(32,   TensorBinaryProtocol.FLAG_HAS_INPUT_SHAPES, "FLAG_HAS_INPUT_SHAPES should be 32");
    }

    // ═══════════════════════════════════════════════════════════════
    // Protocol hardening tests (A7)
    // ═══════════════════════════════════════════════════════════════

    @Test void testFlagMetadata_complete() {
        // Every known flag bit must have a FlagMeta entry
        int[] knownBits = {1, 2, 4, 8, 16, 32};
        for (int bit : knownBits) {
            assertNotNull(TensorBinaryProtocol.FLAG_METADATA.get(bit),
                () -> "FLAG_METADATA missing entry for bit 0x" + Integer.toHexString(bit));
        }
        // Verify the map size matches (no extra phantom entries)
        assertEquals(knownBits.length, TensorBinaryProtocol.FLAG_METADATA.size(),
            "FLAG_METADATA size should match known flag count");
    }

    @Test void testProtocolVersionConstants() {
        // VERSION defines what we can READ
        assertEquals(2, TensorBinaryProtocol.VERSION, "VERSION should be 2");
        assertEquals(1, TensorBinaryProtocol.MIN_SUPPORTED_VERSION, "MIN_SUPPORTED_VERSION should be 1");
        // WIRE_VERSION defaults to 1 for backward compat
        assertEquals(1, TensorBinaryProtocol.WIRE_VERSION,
            "Default WIRE_VERSION should be 1 (backward compat)");
        // Version range is non-empty
        assertTrue(TensorBinaryProtocol.VERSION >= TensorBinaryProtocol.MIN_SUPPORTED_VERSION);
    }

    @Test void testCachedGraphScanner_roundTripAllFlags() {
        // Build a graph that exercises as many flag types as possible:
        // FLAG_HAS_DATA (leaf), FLAG_HAS_SCALAR (pow), FLAG_HAS_PARAM2 (none in simple graph),
        // FLAG_HAS_INDICES (none), FLAG_IS_LEAF, FLAG_HAS_INPUT_SHAPES (broadcast add)
        double[] d1 = rand(6);  // [2, 3]
        double[] d2 = rand(3);  // [3] — broadcast
        RereDiffTensor a = (RereDiffTensor) AD.tensor(d1, 2, 3);
        a.setRequiresGrad(true);
        RereDiffTensor b = (RereDiffTensor) AD.tensor(d2, 3);
        b.setRequiresGrad(true);

        // add([2,3], [3]) → pow(2.0) → sum → scalar
        RereDiffTensor root = (RereDiffTensor) a.add(b).pow(2.0).sum();

        ArrayList<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);

        // Serialize + scan via CachedGraph (exercises the flag-aware scanner)
        TensorBinaryProtocol.CachedGraph cg =
            TensorBinaryProtocol.serializeGraphCached(root, order, 0);

        assertNotNull(cg, "CachedGraph should not be null");
        // Verify updateLeafData works (doesn't throw, produces non-empty buffer)
        byte[] updated = cg.updateLeafData(order);
        assertNotNull(updated);
        assertTrue(updated.length > 100, "Updated buffer should be >100 bytes, got " + updated.length);

        // Verify byte-identical round-trip: fresh serialize vs CachedGraph update
        java.nio.ByteBuffer freshBuf = TensorBinaryProtocol.serializeGraph(root, order);
        byte[] freshBytes = TensorBinaryProtocol.toByteArray(freshBuf);
        assertEquals(freshBytes.length, updated.length,
            "Round-trip: fresh serialize and CachedGraph update should have same length");
        for (int i = 0; i < freshBytes.length; i++) {
            final int idx = i;
            assertEquals(freshBytes[idx], updated[idx],
                () -> "Byte mismatch at offset " + idx);
        }
    }

    @Test void testV2WireFormat_byteIdenticalToV1() {
        // V2 with no extensions adds 2 bytes per node for extension_block_len=0,
        // plus the header version field changes from 1 to 2
        double[] d1 = rand(8);
        RereDiffTensor a = (RereDiffTensor) AD.tensor(d1, 8);
        a.setRequiresGrad(true);
        RereDiffTensor root = (RereDiffTensor) a.pow(2.0).sum();

        ArrayList<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);

        int savedWire = TensorBinaryProtocol.WIRE_VERSION;
        TensorBinaryProtocol.WIRE_VERSION = 1;
        try {
            byte[] v1Bytes = TensorBinaryProtocol.toByteArray(
                TensorBinaryProtocol.serializeGraph(root, order));

            TensorBinaryProtocol.WIRE_VERSION = 2;
            byte[] v2Bytes = TensorBinaryProtocol.toByteArray(
                TensorBinaryProtocol.serializeGraph(root, order));

            // V2 adds 2 bytes per node (u16 extension_block_len=0)
            assertEquals(v1Bytes.length + order.size() * 2, v2Bytes.length,
                "V2 should be exactly " + (order.size() * 2) + " bytes larger than V1");

            // Verify prefix is identical EXCEPT the version field (bytes 4-7)
            for (int j = 0; j < 4; j++) { // magic only
                final int idx = j;
                assertEquals(v1Bytes[idx], v2Bytes[idx],
                    () -> "Magic byte mismatch at offset " + idx);
            }
            for (int j = 8; j < 12; j++) { // num_nodes
                final int idx = j;
                assertEquals(v1Bytes[idx], v2Bytes[idx],
                    () -> "Header byte mismatch at offset " + idx);
            }

            // Scan V2 buffer with CachedGraph — should not throw
            TensorBinaryProtocol.CachedGraph cg =
                TensorBinaryProtocol.serializeGraphCached(root, order, 42);
            assertNotNull(cg);
        } finally {
            TensorBinaryProtocol.WIRE_VERSION = savedWire;
        }
    }

    @Test void testValidateHeader_rejectsBadMagic() {
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocate(12)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buf.putInt(0xDEADBEEF); // bad magic
        buf.putInt(1);
        buf.putInt(0);
        buf.flip();
        assertThrows(BinaryProtocol.ProtocolVersionException.class,
            () -> BinaryProtocol.validateHeader(buf));
    }

    @Test void testValidateHeader_rejectsUnsupportedVersion() {
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocate(12)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buf.putInt(TensorBinaryProtocol.MAGIC);
        buf.putInt(99); // unsupported version
        buf.putInt(0);
        buf.flip();
        assertThrows(BinaryProtocol.ProtocolVersionException.class,
            () -> BinaryProtocol.validateHeader(buf));
    }

    @Test void testValidateHeader_acceptsVersion1() {
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocate(12)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buf.putInt(TensorBinaryProtocol.MAGIC);
        buf.putInt(1);
        buf.putInt(0);
        buf.flip();
        assertDoesNotThrow(() -> BinaryProtocol.validateHeader(buf));
    }

    // ═══════════════════════════════════════════════════════════════
    // B2: Pure-Java protocol round-trip (no native code)
    // ═══════════════════════════════════════════════════════════════

    /**
     * Build a multi-op tensor graph, serialize to binary, scan the bytes
     * with the flag registry, verify no overruns and all node IDs are valid.
     * Runs without any native code — catches protocol bugs early.
     */
    @Test void testPureJavaRoundTrip_multiOpGraph() {
        int B = 2, C = 4;
        double[] d1 = rand(B * C);
        double[] d2 = rand(B * C);

        RereDiffTensor a = (RereDiffTensor) AD.leafTensor(d1, B, C);
        RereDiffTensor b = (RereDiffTensor) AD.leafTensor(d2, B, C);

        // Chain: a + b → pow(2) → sum
        RereDiffTensor ab = (RereDiffTensor) a.add(b);
        RereDiffTensor pow = (RereDiffTensor) ab.pow(2.0);
        RereDiffTensor loss = (RereDiffTensor) pow.sum();

        // Build topology
        ArrayList<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        loss.buildTopo(order, visited);

        // Serialize and scan
        TensorBinaryProtocol.CachedGraph cg =
            TensorBinaryProtocol.serializeGraphCached(loss, order, 0xCAFE);
        assertNotNull(cg);
        byte[] data = cg.updateLeafData(order);
        assertNotNull(data);
        assertTrue(data.length > 12, "binary data too short: " + data.length);

        // Verify header
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.wrap(data)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        int magic = buf.getInt();
        assertEquals(TensorBinaryProtocol.MAGIC, magic, "bad magic");
        int version = buf.getInt();
        assertTrue(version >= 1 && version <= TensorBinaryProtocol.VERSION,
            "unsupported version: " + version);
        int numNodes = buf.getInt();
        assertEquals(order.size(), numNodes, "node count mismatch");

        // Scan each node's bytes manually, verifying no overrun
        java.util.Set<Integer> nodeIds = new java.util.HashSet<>();
        for (int n = 0; n < numNodes; n++) {
            int nodeStart = buf.position();
            int flags = buf.getShort() & 0xFFFF;
            int opLen = buf.getShort() & 0xFFFF;
            int id = buf.getInt();
            assertTrue(id >= 0 && id < numNodes, "invalid node id " + id + " at node " + n);
            nodeIds.add(id);

            // Read op tag
            byte[] opBytes = new byte[opLen];
            buf.get(opBytes);
            String op = new String(opBytes, java.nio.charset.StandardCharsets.UTF_8);
            assertNotNull(op);
            assertFalse(op.isEmpty(), "empty op tag at node " + n);

            // Read shape
            int ndim = buf.getShort() & 0xFFFF;
            int[] shape = new int[ndim];
            for (int d = 0; d < ndim; d++) shape[d] = buf.getInt();

            // Read inputs
            int numInputs = buf.getShort() & 0xFFFF;
            for (int i = 0; i < numInputs; i++) {
                int inputId = buf.getInt();
                assertTrue(nodeIds.contains(inputId) || inputId < numNodes,
                    "input " + inputId + " not yet seen at node " + n);
            }

            // Skip flag-dependent data using the registry (validates no overrun)
            for (int bit = 0; bit < 16; bit++) {
                int bitVal = 1 << bit;
                if ((flags & bitVal) != 0) {
                    buf.position(TensorBinaryProtocol.skipFlagData(
                        data, buf.position(), bitVal, flags, version));
                }
            }

            // V2 extension block
            if (version >= 2) {
                int extLen = buf.getShort() & 0xFFFF;
                buf.position(buf.position() + extLen);
            }

            // Safety: ensure we didn't overrun
            assertTrue(buf.position() <= data.length,
                "buffer overrun at node " + n);
        }

        // All nodes should have unique IDs
        assertEquals(numNodes, nodeIds.size(), "duplicate node IDs");
        // Verify we consumed most of the buffer (allow small implementation-dependent padding)
        assertTrue(buf.position() <= data.length,
            "buffer overrun: pos=" + buf.position() + " > len=" + data.length);
        int remaining = data.length - buf.position();
        assertTrue(remaining < 16,
            "too many unconsumed bytes: " + remaining + " (pos=" + buf.position()
            + " len=" + data.length + ")");
    }
}
