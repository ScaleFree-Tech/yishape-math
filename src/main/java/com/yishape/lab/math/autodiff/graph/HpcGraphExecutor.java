package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.graph.binary.BinaryProtocol;
import com.yishape.lab.math.autodiff.graph.binary.TensorBinaryProtocol;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import com.yishape.lab.math.compute.hpc.HpcAutodiff;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.util.YishapeLogger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

/**
 * Bridges HPC graph execution into the autodiff impl package, where it has
 * package-private access to {@link RereDiffVector#accGrad}.
 */
public final class HpcGraphExecutor {

    private static final YishapeLogger log = YishapeLogger.getLogger(HpcGraphExecutor.class);

    /** Graphs with more nodes than this threshold should use binary protocol, not JSON. */
    public static final int BINARY_THRESHOLD = 500;

    /**
     * Ops supported in tensor-native HPC execution.
     * Reference: GraphOpSchema.Hpc.SUPPORTED.
     */
    private static final HashSet<String> TENSOR_SUPPORTED_OPS = new HashSet<>(GraphOpSchema.Hpc.SUPPORTED);

    // --- HPC failure cooldown (per-thread, to prevent cross-thread pollution) ---
    private static final int COOLDOWN_THRESHOLD = 3;
    private static final int COOLDOWN_STEPS = 100;
    private static final ThreadLocal<int[]> HPC_COOLDOWN =
        ThreadLocal.withInitial(() -> new int[]{0, 0}); // [0]=failures, [1]=remaining

    /** Tracks which unsupported ops have already been reported to stderr, to suppress duplicates. */
    private static final java.util.Set<String> REPORTED_UNSUPPORTED_OPS = java.util.concurrent.ConcurrentHashMap.newKeySet();

    // --- Graph structure hash cache (per-thread to prevent cross-thread races) ---
    // Previously static volatile — thread A could overwrite thread B's cached skeleton
    // mid-execution, causing NaN gradients or incorrect results.
    private static final ThreadLocal<HpcCacheEntry> HPC_CACHE =
        ThreadLocal.withInitial(HpcCacheEntry::new);

    /** Per-thread cache entry: structure hash + serialized graph skeleton. */
    static final class HpcCacheEntry {
        int structureHash = 0;
        com.yishape.lab.math.autodiff.graph.binary.TensorBinaryProtocol.CachedGraph cachedGraph = null;

        void invalidate() {
            structureHash = 0;
            cachedGraph = null;
        }
    }

    // Tensor thread locals
    private static final ThreadLocal<ArrayList<RereDiffTensor>> HPC_TENSOR_TOPO =
        ThreadLocal.withInitial(ArrayList::new);
    private static final ThreadLocal<HashSet<RereDiffTensor>> HPC_TENSOR_VISITED =
        ThreadLocal.withInitial(HashSet::new);

    private HpcGraphExecutor() {
    }

    /**
     * Attempt HPC graph execution for a vector-based graph.
     * Delegates to the tensor-native path via {@link RereDiffVector#tensor}.
     */
    public static double tryExecute(RereDiffVector root) {
        return tryExecute(root.tensor);
    }

    /**
     * Attempt HPC graph execution for a tensor-based computation graph.
     * Exports via {@link TensorGraphExporter} and delegates to the HPC bridge.
     *
     * @param root the tensor graph root node
     * @return loss value, or {@link Double#NaN} if HPC is unavailable or fails
     */
    public static double tryExecute(RereDiffTensor root) {
        if (!com.yishape.lab.math.compute.hpc.HpcConfig.allowAttempts()) return Double.NaN;

        // HPC cooldown: skip after repeated consecutive failures (per-thread)
        int[] cd = HPC_COOLDOWN.get();
        if (cd[1] > 0) {
            cd[1]--;
            return Double.NaN;
        }

        ArrayList<RereDiffTensor> order = HPC_TENSOR_TOPO.get();
        order.clear();
        HashSet<RereDiffTensor> visited = HPC_TENSOR_VISITED.get();
        visited.clear();
        root.buildTopo(order, visited);

        // Phase 3.4: Auto-sync FloatDiffTensor FP32 master weights → FP64 value
        // before HPC serialization, so native execution reads the latest weights.
        com.yishape.lab.math.autodiff.impl.FloatDiffTensor.syncFloatLeaves(order);

        // Validate graph structure (per-thread cached: skip when topology unchanged).
        HpcCacheEntry cacheEntry = HPC_CACHE.get();
        int structureHash = ExportShapeValidator.computeStructureHash(order);
        if (structureHash != cacheEntry.structureHash) {
            ExportShapeValidator.Result validation = ExportShapeValidator.validate(order);
            if (validation.hasErrors()) {
                log.warn("HPC graph validation failed — falling back to CPU:\n{}", validation);
                return Double.NaN;
            }
            cacheEntry.structureHash = structureHash; // only cache on success
        }

        // Check unsupported ops
        for (RereDiffTensor v : order) {
            if (v.opTag() != null && !TENSOR_SUPPORTED_OPS.contains(v.opTag())) {
                String msg = String.format("unsupported op='%s' (graph has %d nodes)",
                    v.opTag(), order.size());
                if (REPORTED_UNSUPPORTED_OPS.add(v.opTag())) {
                    if (log.isDebugEnabled()) {
                        int leafCount = 0;
                        for (RereDiffTensor n : order) {
                            if (n.isLeaf()) leafCount++;
                        }
                        log.debug("HPC tensor graph fallback: {}", msg);
                    }
                }
                return NativeStrictMode.failOrNaN("HPC", msg);
            }
        }

        // Collect leaves
        ArrayList<RereDiffTensor> leaves = new ArrayList<>();
        for (RereDiffTensor v : order) {
            if (v.isLeaf()) {
                leaves.add(v);
            }
        }

        // Try binary path first (use cached graph when topology unchanged)
        double binaryResult;
        var cached = cacheEntry.cachedGraph;
        if (cached != null && structureHash == cached.structureHash()) {
            binaryResult = tryExecuteTensorBinaryIncremental(order, leaves, cached, cacheEntry);
        } else {
            binaryResult = tryExecuteTensorBinaryFull(root, order, leaves, structureHash, cacheEntry);
        }
        if (!Double.isNaN(binaryResult)) {
            HPC_COOLDOWN.get()[0] = 0;
            detachGraphAfterNativeExecution(order);
            com.yishape.lab.math.autodiff.impl.FloatDiffTensor.syncDoubleLeaves(order);
            return binaryResult;
        }
        // In strict mode, binary path failure is a bug — surface it immediately
        if (NativeStrictMode.isStrict()) {
            throw new NativeStrictMode.NativeExecutionException("HPC",
                "Binary protocol execution returned NaN for op='" + root.opTag()
                + "' — likely unsupported op, protocol mismatch, or Rust bug. "
                + "Run without -Dyishape.strictNative to fall back to JSON.");
        }
        if (log.isDebugEnabled()) log.debug("Binary path returned NaN, falling back to JSON (nodes={})", order.size());

        // JSON fallback
        String json = TensorGraphExporter.toJson(root);
        if (json == null) {
            log.debug("HPC tensor graph fallback: JSON export failed for {} nodes", order.size());
            return NativeStrictMode.failOrNaN("HPC", "JSON export failed (nodes=%d)", order.size());
        }
        if (order.size() > BINARY_THRESHOLD) {
            log.warn("HPC graph has {} nodes (>{}); JSON path should be avoided — "
                + "binary protocol is preferred. Check why binary path failed.", order.size(), BINARY_THRESHOLD);
        }
        double[][] result = HpcAutodiff.tryExecute(json);
        if (result == null || result.length < 2 || result[0] == null) {
            log.debug("HPC tensor graph fallback: Rust execution returned null or invalid result (nodes={})",
                order.size());
            trackHpcFailure();
            return NativeStrictMode.failOrNaN("HPC",
                "Rust execution returned null/invalid (nodes=%d) — likely faer error or unsupported op",
                order.size());
        }

        // NOTE: Do NOT derive batchSize from root.scalarParam.
        // scalarParam is overloaded: exponent (pow/powSum/powMean), divisor n (mean/div),
        // activation alpha, etc. Treating it as batchSize causes incorrect loss/grad scaling.
        // Each HPC op must produce self-contained, correctly-scaled results.
        double loss = result[0][0];
        // Validate gradient count and lengths — any mismatch means the HPC graph
        // is inconsistent with the Java graph; fall back to CPU to avoid silent errors.
        if (result.length - 1 != leaves.size()) {
            log.warn("HPC tensor gradient count mismatch: got {} gradients for {} leaves — falling back to CPU",
                    result.length - 1, leaves.size());
            trackHpcFailure();
            return NativeStrictMode.failOrNaN("HPC",
                "gradient count mismatch: got %d gradients for %d leaves",
                result.length - 1, leaves.size());
        }
        for (int i = 0; i < leaves.size(); i++) {
            if (result[i + 1] == null) {
                log.warn("HPC tensor gradient[{}] is null — falling back to CPU", i);
                trackHpcFailure();
                return Double.NaN;
            }
            int gradLen = result[i + 1].length;
            int leafLen = Math.toIntExact(leaves.get(i).totalSize());
            if (gradLen != leafLen) {
                log.warn("HPC tensor gradient length mismatch at leaf {}: got {} expected {} — falling back to CPU",
                        i, gradLen, leafLen);
                trackHpcFailure();
                return Double.NaN;
            }
            leaves.get(i).accGrad(result[i + 1]);
        }
        HPC_COOLDOWN.get()[0] = 0; // reset cooldown on success
        detachGraphAfterNativeExecution(order);
        com.yishape.lab.math.autodiff.impl.FloatDiffTensor.syncDoubleLeaves(order);
        return loss;
    }

    /**
     * Detach graph references after successful native execution.
     * Clears backwardFn and inputs on non-leaf nodes to prevent
     * double gradient accumulation if {@code backward()} is called again.
     * Mirrors the cleanup in {@code RereDiffTensor.backwardImpl()} (lines 298-305).
     */
    private static void detachGraphAfterNativeExecution(ArrayList<RereDiffTensor> order) {
        for (int i = order.size() - 1; i >= 0; i--) {
            RereDiffTensor v = order.get(i);
            if (!v.isLeaf()) {
                v.inputs = null;
                v.backwardFn = null;
                v.symbolicBackwardFn = null;
            }
        }
    }

    /** Track HPC failure for cooldown (per-thread). */
    private static void trackHpcFailure() {
        int[] cd = HPC_COOLDOWN.get();
        int failures = ++cd[0];
        if (failures >= COOLDOWN_THRESHOLD) {
            cd[1] = COOLDOWN_STEPS;
        }
    }

    /**
     * Reset the HPC cooldown for the current thread.
     * Call this before test suites to ensure HPC failures from a prior test
     * class don't poison subsequent HPC tests. Idempotent and safe to call
     * when HPC is not in use.
     */
    public static void resetCooldown() {
        HPC_COOLDOWN.get()[0] = 0;
        HPC_COOLDOWN.get()[1] = 0;
    }

    /** Full serialization + cache skeleton for subsequent incremental updates. */
    private static double tryExecuteTensorBinaryFull(RereDiffTensor root,
            ArrayList<RereDiffTensor> order, ArrayList<RereDiffTensor> leaves,
            int structureHash, HpcCacheEntry cacheEntry) {
        try {
            // Cache skeleton + full data for first step
            cacheEntry.cachedGraph = TensorBinaryProtocol.serializeGraphCached(root, order, structureHash);
            byte[] data = cacheEntry.cachedGraph.updateLeafData(order);
            byte[] resultBytes = HpcOptionalRuntime.tryExecuteGraphBinary(data);
            if (resultBytes == null || resultBytes.length == 0) {
                log.debug("HPC binary full execution returned null/empty for {} nodes", order.size());
                trackHpcFailure();
                cacheEntry.invalidate();
                return Double.NaN;
            }

            ByteBuffer resultBuf = ByteBuffer.wrap(resultBytes).order(ByteOrder.LITTLE_ENDIAN);
            var parsed = BinaryProtocol.deserializeResult(resultBuf);
            double loss = parsed.loss();
            if (Double.isNaN(loss)) {
                trackHpcFailure();
                cacheEntry.invalidate();
                return Double.NaN;
            }

            List<double[]> grads = parsed.grads();
            if (grads.size() != leaves.size()) {
                trackHpcFailure();
                cacheEntry.invalidate();
                return Double.NaN;
            }

            for (int i = 0; i < grads.size(); i++) {
                double[] g = grads.get(i);
                long leafSize = leaves.get(i).totalSize();
                if (g.length != leafSize) {
                    log.warn("HPC binary gradient length mismatch at leaf {}: got {} expected {} — falling back to CPU",
                        i, g.length, leafSize);
                    trackHpcFailure();
                    cacheEntry.invalidate();
                    return Double.NaN;
                }
                leaves.get(i).accGrad(g);
            }
            return loss;
        } catch (Exception e) {
            if (log.isDebugEnabled()) log.debug("HPC graph execution failed", e);
            trackHpcFailure();
            cacheEntry.invalidate();
            return Double.NaN;
        }
    }

    /** Incremental: clone cached skeleton, overwrite leaf data, send. */
    private static double tryExecuteTensorBinaryIncremental(
            ArrayList<RereDiffTensor> order, ArrayList<RereDiffTensor> leaves,
            com.yishape.lab.math.autodiff.graph.binary.TensorBinaryProtocol.CachedGraph cached,
            HpcCacheEntry cacheEntry) {
        try {
            byte[] data = cached.updateLeafData(order);
            byte[] resultBytes = HpcOptionalRuntime.tryExecuteGraphBinary(data);
            if (resultBytes == null || resultBytes.length == 0) {
                log.debug("HPC binary incremental execution returned null/empty for {} nodes", order.size());
                trackHpcFailure();
                cacheEntry.invalidate();
                return Double.NaN;
            }

            ByteBuffer resultBuf = ByteBuffer.wrap(resultBytes).order(ByteOrder.LITTLE_ENDIAN);
            var parsed = BinaryProtocol.deserializeResult(resultBuf);
            double loss = parsed.loss();
            if (Double.isNaN(loss)) {
                trackHpcFailure();
                cacheEntry.invalidate();
                return Double.NaN;
            }

            List<double[]> grads = parsed.grads();
            if (grads.size() != leaves.size()) {
                trackHpcFailure();
                cacheEntry.invalidate();
                return Double.NaN;
            }

            for (int i = 0; i < grads.size(); i++) {
                double[] g = grads.get(i);
                long leafSize = leaves.get(i).totalSize();
                if (g.length != leafSize) {
                    log.warn("HPC binary incremental gradient length mismatch at leaf {}: got {} expected {} — falling back to CPU",
                        i, g.length, leafSize);
                    trackHpcFailure();
                    cacheEntry.invalidate();
                    return Double.NaN;
                }
                leaves.get(i).accGrad(g);
            }
            return loss;
        } catch (Exception e) {
            if (log.isDebugEnabled()) log.debug("HPC incremental graph execution failed", e);
            trackHpcFailure();
            cacheEntry.invalidate();
            return Double.NaN;
        }
    }

}
