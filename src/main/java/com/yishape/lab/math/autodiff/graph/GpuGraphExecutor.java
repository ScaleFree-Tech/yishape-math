package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.graph.binary.BinaryProtocol;
import com.yishape.lab.math.autodiff.graph.binary.TensorBinaryProtocol;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;
import com.yishape.lab.math.compute.gpu.GpuConfig;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.util.YishapeLogger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

/**
 * GPU graph-level execution. Exports the autodiff graph to JSON,
 * executes on GPU via wgpu, and applies gradients to leaf nodes.
 * Mirrors {@link HpcGraphExecutor} but delegates to GPU runtime.
 */
public final class GpuGraphExecutor {

    private static final YishapeLogger log = YishapeLogger.getLogger(GpuGraphExecutor.class);

    /** Graphs with more nodes than this threshold should use binary protocol, not JSON. */
    public static final int BINARY_THRESHOLD = 500;

    /**
     * Minimum total element count for GPU execution. Graphs with fewer total elements
     * skip GPU to avoid kernel launch overhead dominating compute time.
     * Configurable via {@code -Dyishape.gpu.minElements=N}.
     */
    public static final int MIN_ELEMENTS = Integer.getInteger("yishape.gpu.minElements", 1000);

    /**
     * Enables verbose graph-level debug output (ops, shapes, params per node).
     * Set via {@code -Dyishape.gpu.verbose=true}.
     */
    private static final boolean VERBOSE = Boolean.getBoolean("yishape.gpu.verbose");

    // --- GPU failure cooldown (per-thread, prevents cross-thread pollution) ---
    private static final int COOLDOWN_THRESHOLD = 3;
    private static final int COOLDOWN_STEPS = 100;
    private static final ThreadLocal<int[]> GPU_COOLDOWN =
        ThreadLocal.withInitial(() -> new int[]{0, 0}); // [0]=failures, [1]=remaining

    /** Tracks which unsupported ops have already been reported to stderr, to suppress duplicates. */
    private static final java.util.Set<String> REPORTED_UNSUPPORTED_OPS = java.util.concurrent.ConcurrentHashMap.newKeySet();

    /**
     * Ops supported in tensor-native GPU execution.
     * Reference: GraphOpSchema.Gpu.SUPPORTED.
     * NOTE: Ops NOT yet implemented in Rust GPU: reciprocal, scatter, narrow,
     * batchNorm (1D; only batchNorm2d is supported). Adding them here before
     * Rust supports them causes "unsupported forward op: <op>" worker errors
     * and NaN GPU results.
     * rsubScalar and rdivScalar were added to Rust GPU (2026-06).
     * expand, permute and slice were recently added (2026-06-16) — see Rust
     * gpu_worker graph.rs forward_dispatch/backward_dispatch for implementations.
     */
    private static final HashSet<String> TENSOR_SUPPORTED_OPS = new HashSet<>(GraphOpSchema.Gpu.SUPPORTED);

    // --- GPU binary skeleton cache (per-thread, same pattern as HpcGraphExecutor) ---
    private static final ThreadLocal<GpuCacheEntry> GPU_CACHE =
        ThreadLocal.withInitial(GpuCacheEntry::new);

    static final class GpuCacheEntry {
        int structureHash = 0;
        TensorBinaryProtocol.CachedGraph cachedGraph = null;

        void invalidate() {
            structureHash = 0;
            cachedGraph = null;
        }
    }

    // Tensor thread locals
    private static final ThreadLocal<ArrayList<RereDiffTensor>> TENSOR_TOPO =
        ThreadLocal.withInitial(ArrayList::new);
    private static final ThreadLocal<HashSet<RereDiffTensor>> TENSOR_VISITED =
        ThreadLocal.withInitial(HashSet::new);

    private GpuGraphExecutor() {}

    /**
     * Attempt GPU graph execution for a vector-based graph.
     * Delegates to the tensor-native path via {@link RereDiffVector#tensor}.
     */
    public static double tryExecute(RereDiffVector root) {
        return tryExecute(root.tensor);
    }

    /**
     * Attempt GPU graph execution for a tensor-based computation graph.
     * Exports via {@link TensorGraphExporter} and delegates to the GPU bridge.
     *
     * @param root the tensor graph root node
     * @return loss value, or {@link Double#NaN} if GPU is unavailable or fails
     */
    public static double tryExecute(RereDiffTensor root) {
        if (log.isDebugEnabled()) {
            log.debug("tryExecute called for op={} shape={}", root.opTag(), java.util.Arrays.toString(root.shape()));
        }
        if (!GpuConfig.allowAttempts()) { return Double.NaN; }
        if (!GpuOptionalRuntime.isGpuAvailable()) { return Double.NaN; }

        // GPU cooldown: skip after repeated consecutive failures (per-thread)
        int[] cd = GPU_COOLDOWN.get();
        if (cd[1] > 0) {
            cd[1]--;
            return Double.NaN;
        }

        ArrayList<RereDiffTensor> order = TENSOR_TOPO.get();
        order.clear();
        HashSet<RereDiffTensor> visited = TENSOR_VISITED.get();
        visited.clear();
        root.buildTopo(order, visited);

        // Phase 3.4: Auto-sync FloatDiffTensor FP32 master weights → FP64 value
        // before GPU serialization, so native execution reads the latest weights.
        com.yishape.lab.math.autodiff.impl.FloatDiffTensor.syncFloatLeaves(order);

        // Smart threshold: skip GPU for tiny graphs where kernel launch overhead
        // would dominate compute time. Controlled by -Dyishape.gpu.minElements.
        long totalElements = 0;
        for (RereDiffTensor v : order) {
            totalElements += v.totalSize();
        }
        if (totalElements < MIN_ELEMENTS) {
            if (log.isDebugEnabled()) {
                log.debug("GPU skipped: graph has {} total elements (<{} threshold)",
                    totalElements, MIN_ELEMENTS);
            }
            return Double.NaN;
        }

        // Shape validation (per-thread cached: skip when topology unchanged).
        GpuCacheEntry cacheEntry = GPU_CACHE.get();
        int structureHash = ExportShapeValidator.computeStructureHash(order);
        if (structureHash != cacheEntry.structureHash) {
            ExportShapeValidator.Result validation = ExportShapeValidator.validate(order);
            if (validation.hasErrors()) {
                log.error("GPU graph export BLOCKED: shape validation failed:\n{}", validation);
                return Double.NaN;
            }
            if (validation.hasWarnings() && VERBOSE) {
                log.debug("GPU shape validation warning: {}", validation);
            }
            cacheEntry.structureHash = structureHash;
        }
        if (VERBOSE && log.isDebugEnabled()) {
            java.util.LinkedHashSet<String> seenOps = new java.util.LinkedHashSet<>();
            for (RereDiffTensor v : order) {
                if (v.opTag() != null) seenOps.add(v.opTag());
            }
            log.debug("GPU graph ops: {} nodes={}", seenOps, order.size());
            for (RereDiffTensor v : order) {
                if (v.totalSize() == 0) {
                    log.debug("GPU zero-size node: op='{}' shape={} exportShape={} scalar={} isLeaf={}",
                        v.opTag(), java.util.Arrays.toString(v.shape()),
                        java.util.Arrays.toString(v.exportShape()), v.scalarParam(), v.isLeaf());
                }
                if ("conv2d".equals(v.opTag())) {
                    long bits = Double.doubleToRawLongBits(v.scalarParam());
                    int kh = (int)((bits >> 16) & 0xFF);
                    int kw = (int)((bits >> 8) & 0xFF);
                    int stride = (int)(bits & 0xFF);
                    long bits2 = Double.doubleToRawLongBits(v.scalarParam2());
                    int pad = (int)((bits2 >> 16) & 0xFFFF);
                    int outCh = (int)(bits2 & 0xFFFF);
                    log.debug("GPU conv2d params: kh={} kw={} stride={} pad={} outCh={} shape={} exportShape={}",
                        kh, kw, stride, pad, outCh, java.util.Arrays.toString(v.shape()),
                        java.util.Arrays.toString(v.exportShape()));
                }
                if ("linear".equals(v.opTag())) {
                    log.debug("GPU linear shape={}", java.util.Arrays.toString(v.shape()));
                }
                if ("mha".equals(v.opTag())) {
                    long bits = Double.doubleToRawLongBits(v.scalarParam());
                    int nh = (int)((bits >> 32) & 0xFFFF);
                    int nkvh = (int)((bits >> 16) & 0xFFFF);
                    int dm = (int)(bits & 0xFFFF);
                    long bits2 = Double.doubleToRawLongBits(v.scalarParam2());
                    int sl = (int)((bits2 >> 32) & 0xFFFF);
                    boolean causal = (bits2 & 0x2) != 0;
                    boolean hasBias = (bits2 & 0x1) != 0;
                    log.debug("GPU mha params: shape={} exportShape={} numHeads={} numKVHeads={} dModel={} seqLen={} causal={} hasBias={}",
                        java.util.Arrays.toString(v.shape()), java.util.Arrays.toString(v.exportShape()),
                        nh, nkvh, dm, sl, causal, hasBias);
                }
                if ("maxpool2d".equals(v.opTag())) {
                    long bits = Double.doubleToRawLongBits(v.scalarParam());
                    int kh = (int)((bits >> 16) & 0xFF);
                    int kw = (int)((bits >> 8) & 0xFF);
                    int stride = (int)(bits & 0xFF);
                    long bits2 = Double.doubleToRawLongBits(v.scalarParam2());
                    int pad = (int)((bits2 >> 16) & 0xFFFF);
                    // C is from shape[1], not packed in param2 (param2 only carries padding)
                    log.debug("GPU maxpool2d params: kh={} kw={} stride={} pad={} shape={} exportShape={}",
                        kh, kw, stride, pad, java.util.Arrays.toString(v.shape()),
                        java.util.Arrays.toString(v.exportShape()));
                }
            }
        }
        // Guard: reject graphs with zero-size nodes (would cause wgpu crash)
        for (RereDiffTensor v : order) {
            if (v.totalSize() == 0 && !v.isLeaf()) {
                log.warn("GPU skipped: zero-size node op='{}' shape={} — would crash wgpu",
                    v.opTag(), java.util.Arrays.toString(v.shape()));
                return Double.NaN;
            }
        }
        for (RereDiffTensor v : order) {
            if (v.opTag() != null && !TENSOR_SUPPORTED_OPS.contains(v.opTag())) {
                String msg = String.format("unsupported op='%s' (graph has %d nodes)",
                    v.opTag(), order.size());
                if (REPORTED_UNSUPPORTED_OPS.add(v.opTag())) {
                    log.warn("GPU tensor graph fallback: {}", msg);
                }
                return NativeStrictMode.failOrNaN("GPU", msg);
            }
        }

        // Collect leaves (needed for both binary cache dispatch and JSON fallback)
        ArrayList<RereDiffTensor> leaves = new ArrayList<>();
        for (RereDiffTensor v : order) {
            if (v.isLeaf()) leaves.add(v);
        }

        // Try binary path first (use cached graph when topology unchanged)
        TensorBinaryProtocol.CachedGraph cached = cacheEntry.cachedGraph;
        double binaryResult;
        if (cached != null && cacheEntry.structureHash == cached.structureHash()) {
            binaryResult = tryExecuteTensorBinaryIncremental(order, leaves, cached, cacheEntry);
        } else {
            binaryResult = tryExecuteTensorBinaryFull(root, order, leaves, cacheEntry);
        }
        if (!Double.isNaN(binaryResult)) {
            GPU_COOLDOWN.get()[0] = 0; // reset on success
            // Build cache skeleton AFTER successful execution but BEFORE detach,
            // so the graph structure is still intact for serialization.
            if (cacheEntry.cachedGraph == null) {
                try {
                    cacheEntry.cachedGraph = TensorBinaryProtocol.serializeGraphCached(
                        root, order, cacheEntry.structureHash);
                } catch (Exception e) {
                    if (log.isDebugEnabled()) log.debug("GPU cache build failed (non-fatal)", e);
                }
            }
            detachGraphAfterNativeExecution(order);
            com.yishape.lab.math.autodiff.impl.FloatDiffTensor.syncDoubleLeaves(order);
            return binaryResult;
        }
        // In strict mode, binary path failure is a bug — surface it immediately
        if (NativeStrictMode.isStrict()) {
            throw new NativeStrictMode.NativeExecutionException("GPU",
                "Binary protocol execution returned NaN for op='" + root.opTag()
                + "' — likely unsupported op, protocol mismatch, or Rust bug. "
                + "Run without -Dyishape.strictNative to fall back to JSON.");
        }
        log.debug("GPU binary path returned NaN for op={}, trying JSON fallback", root.opTag());

        // NOTE: The isolated worker guard that blocked JSON fallback has been REMOVED.
        // Previously, when an isolated worker existed and binary path failed, the guard
        // blocked JSON fallback to prevent wgpu 29.0.3 Storage::remove abort from
        // crashing the JVM. In practice this made GPU execution impossible whenever
        // the binary path encountered ANY error (unsupported op, execution bug, etc.)
        // — not just worker crashes. The JSON path uses YishapeGpu.getOrCreateContext()
        // (different code path from gpu_worker), and the isolated subprocess already
        // protects against Rust panics. HPC path never had this guard and is more robust.
        // Set -Dyishape.gpu.blockJsonFallback=true to restore old blocking behavior.
        if (Boolean.getBoolean("yishape.gpu.blockJsonFallback")) {
            log.warn("GPU binary path failed, blockJsonFallback=true blocks JSON fallback. "
                + "Remove -Dyishape.gpu.blockJsonFallback to permit in-process fallback.");
            return Double.NaN;
        }

        if (VERBOSE && log.isDebugEnabled()) {
            log.debug("GPU binary path returned NaN for op={}, trying JSON fallback", root.opTag());
        }

        // Export and execute (JSON fallback, leaves already collected above)
        String json = TensorGraphExporter.toJson(root);
        if (VERBOSE && log.isDebugEnabled()) {
            log.debug("GPU toJson returned {} chars", json != null ? json.length() : 0);
        }
        if (json == null) {
            log.debug("GPU tensor graph fallback: JSON export failed");
            return NativeStrictMode.failOrNaN("GPU", "JSON export failed for op='%s'", root.opTag());
        }
        if (order.size() > BINARY_THRESHOLD) {
            log.warn("GPU graph has {} nodes (>{}); JSON path should be avoided — "
                + "binary protocol is preferred. Check why binary path failed.", order.size(), BINARY_THRESHOLD);
        }

        String resultJson = GpuOptionalRuntime.tryExecuteGraph(json);
        if (resultJson == null) {
            if (VERBOSE && log.isDebugEnabled()) {
                log.debug("GPU Rust execution returned null, nodes={} leaves={}", order.size(), leaves.size());
            }
            trackGpuFailure();
            return NativeStrictMode.failOrNaN("GPU",
                "Rust execution returned null (nodes=%d, leaves=%d) — likely wgpu crash or unsupported op",
                order.size(), leaves.size());
        }

        try {
            // NOTE: Do NOT derive batchSize from root.scalarParam.
            // scalarParam is overloaded — it stores op-specific parameters:
            //   exponent for pow/powSum, divisor n for mean/div, alpha for activations.
            // Treating it as batchSize would incorrectly divide loss/grads (e.g. powSum
            // with scalarParam=2 gives halved results). Each GPU op must be self-contained.
            double loss = applyTensorGradientsFromJson(root, leaves, resultJson);
            if (!Double.isNaN(loss)) {
                GPU_COOLDOWN.get()[0] = 0; // reset cooldown on success
                detachGraphAfterNativeExecution(order);
                com.yishape.lab.math.autodiff.impl.FloatDiffTensor.syncDoubleLeaves(order);
            }
            return loss;
        } catch (Exception e) {
            trackGpuFailure();
            log.debug("GPU graph execution failed", e);
            return NativeStrictMode.failOrNaN("GPU",
                "JSON result parsing failed: %s", e.getMessage());
        }
    }

    /** Parse JSON result and apply gradients to tensor leaves. */
    private static double applyTensorGradientsFromJson(RereDiffTensor root,
            ArrayList<RereDiffTensor> leaves, String resultJson) {
        double loss = extractDoubleField(resultJson, "loss");
        if (Double.isNaN(loss)) return Double.NaN;

        int gradStart = resultJson.indexOf("\"grads\"");
        if (gradStart < 0) gradStart = resultJson.indexOf("\"gradients\"");
        if (gradStart < 0) return Double.NaN;
        int arrStart = resultJson.indexOf('[', gradStart);
        if (arrStart < 0) return Double.NaN;
        int arrEnd = findMatchingBracket(resultJson, arrStart);
        if (arrEnd < 0) return Double.NaN;

        int pos = arrStart + 1;
        int leafIdx = 0;
        while (pos < arrEnd && leafIdx < leaves.size()) {
            while (pos < arrEnd && (resultJson.charAt(pos) == ',' || Character.isWhitespace(resultJson.charAt(pos)))) {
                pos++;
            }
            if (pos >= arrEnd || resultJson.charAt(pos) != '[') break;
            int innerEnd = findMatchingBracket(resultJson, pos);
            if (innerEnd < 0) break;

            String inner = resultJson.substring(pos + 1, innerEnd);
            String[] rawTokens = inner.split(",");
            // Count valid (non-empty) tokens to handle trailing commas or sparse arrays
            int validCount = 0;
            for (String t : rawTokens) {
                if (!t.trim().isEmpty()) validCount++;
            }
            double[] gradData = new double[validCount];
            int idx = 0;
            for (String t : rawTokens) {
                String trimmed = t.trim();
                if (trimmed.isEmpty()) continue;
                gradData[idx++] = Double.parseDouble(trimmed);
            }

            // C23: verify gradient length matches leaf tensor size BEFORE accGrad
            long leafSize = leaves.get(leafIdx).totalSize();
            if (gradData.length != leafSize) {
                log.warn("GPU tensor gradient length mismatch at leaf {}: got {} expected {} — falling back to CPU",
                    leafIdx, gradData.length, leafSize);
                trackGpuFailure();
                return Double.NaN;
            }
            leaves.get(leafIdx).accGrad(gradData);
            leafIdx++;
            pos = innerEnd + 1;
        }

        if (leafIdx < leaves.size()) {
            log.warn("GPU tensor returned fewer gradients ({}) than leaves ({}) — falling back to CPU",
                leafIdx, leaves.size());
            trackGpuFailure();
            return Double.NaN;
        }
        return loss;
    }

    /**
     * Detach graph references after successful native execution.
     * Clears backwardFn and inputs on intermediate nodes to prevent
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

    /** Track a GPU failure: increment counter, enter cooldown if threshold reached. */
    private static void trackGpuFailure() {
        int[] cd = GPU_COOLDOWN.get();
        int failures = ++cd[0];
        if (failures >= COOLDOWN_THRESHOLD) {
            cd[1] = COOLDOWN_STEPS;
            if (log.isDebugEnabled()) {
                log.debug("GPU cooldown: {} consecutive failures, cooling for {} steps",
                    failures, COOLDOWN_STEPS);
            }
        }
    }

    /**
     * Reset the GPU cooldown for the current thread.
     * Call this before test suites to ensure GPU failures from a prior test
     * class don't poison subsequent GPU tests. Idempotent and safe to call
     * when GPU is not in use.
     */
    public static void resetCooldown() {
        GPU_COOLDOWN.get()[0] = 0;
        GPU_COOLDOWN.get()[1] = 0;
    }

    // ── Binary execution with skeleton cache (same pattern as HpcGraphExecutor) ──

    /**
     * Full serialization: serialize fresh via {@link TensorBinaryProtocol#serializeGraph}.
     * Cache skeleton is built AFTER successful GPU execution (not here) to avoid
     * side-effects from {@code serializeGraphCached} during the critical path.
     */
    private static double tryExecuteTensorBinaryFull(RereDiffTensor root,
            ArrayList<RereDiffTensor> order, ArrayList<RereDiffTensor> leaves,
            GpuCacheEntry cacheEntry) {
        try {
            java.nio.ByteBuffer buf = TensorBinaryProtocol.serializeGraph(root, order);
            byte[] data = new byte[buf.remaining()];
            buf.get(data);
            return executeBinaryAndApplyGrads(order, leaves, data, cacheEntry);
        } catch (Exception e) {
            if (log.isDebugEnabled()) log.debug("GPU full binary execution failed", e);
            trackGpuFailure();
            cacheEntry.invalidate();
            return Double.NaN;
        }
    }

    /** Incremental: clone cached skeleton, overwrite leaf data, execute. */
    private static double tryExecuteTensorBinaryIncremental(
            ArrayList<RereDiffTensor> order, ArrayList<RereDiffTensor> leaves,
            TensorBinaryProtocol.CachedGraph cached, GpuCacheEntry cacheEntry) {
        try {
            byte[] data = cached.updateLeafData(order);
            return executeBinaryAndApplyGrads(order, leaves, data, cacheEntry);
        } catch (Exception e) {
            if (log.isDebugEnabled()) log.debug("GPU incremental binary execution failed", e);
            trackGpuFailure();
            cacheEntry.invalidate();
            return Double.NaN;
        }
    }

    /** Send binary data to GPU, parse result, apply gradients to leaves. */
    private static double executeBinaryAndApplyGrads(
            ArrayList<RereDiffTensor> order, ArrayList<RereDiffTensor> leaves,
            byte[] data, GpuCacheEntry cacheEntry) {
        if (log.isDebugEnabled()) {
            log.debug("GPU binary serialized {} nodes, {} bytes", order.size(), data.length);
        }

        byte[] resultBytes = GpuOptionalRuntime.tryExecuteGraphBinary(data);
        if (resultBytes == null || resultBytes.length == 0) {
            log.debug("GPU binary execution returned null/empty result");
            cacheEntry.invalidate();
            return Double.NaN;
        }
        if (log.isDebugEnabled()) {
            log.debug("GPU binary execution returned {} bytes", resultBytes.length);
        }

        java.nio.ByteBuffer resultBuf = java.nio.ByteBuffer.wrap(resultBytes)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN);

        var parsed = com.yishape.lab.math.autodiff.graph.binary.BinaryProtocol.deserializeResult(resultBuf);
        double loss = parsed.loss();
        if (Double.isNaN(loss)) {
            cacheEntry.invalidate();
            return Double.NaN;
        }

        // NOTE: no batchSize normalization. root.scalarParam stores op-specific
        // parameters (exponent, divisor, alpha), NOT batch size. Each GPU op must
        // produce correct loss/grads independently.
        java.util.List<double[]> grads = parsed.grads();
        if (grads.size() != leaves.size()) {
            log.warn("GPU binary grad count mismatch: got {} leaves, expected {}",
                grads.size(), leaves.size());
            cacheEntry.invalidate();
            return Double.NaN;
        }

        if (log.isDebugEnabled()) {
            log.debug("GPU binary result: loss={} numGrads={} orderSize={}",
                loss, grads.size(), order.size());
        }

        for (int i = 0; i < grads.size(); i++) {
            double[] g = grads.get(i);
            // C23: verify gradient length matches leaf tensor size
            long leafSize = leaves.get(i).totalSize();
            if (g.length != leafSize) {
                log.warn("GPU binary gradient length mismatch at leaf {}: got {} expected {} — falling back to CPU",
                    i, g.length, leafSize);
                trackGpuFailure();
                cacheEntry.invalidate();
                return Double.NaN;
            }
            leaves.get(i).accGrad(g);
        }
        return loss;
    }

    private static double extractDoubleField(String json, String field) {
        String key = "\"" + field + "\"";
        int idx = json.indexOf(key);
        if (idx < 0) return Double.NaN;
        idx = json.indexOf(':', idx);
        if (idx < 0) return Double.NaN;
        int start = idx + 1;
        while (start < json.length() && Character.isWhitespace(json.charAt(start))) start++;
        int end = start;
        while (end < json.length() && json.charAt(end) != ',' && json.charAt(end) != '}') end++;
        try {
            return Double.parseDouble(json.substring(start, end).trim());
        } catch (NumberFormatException e) {
            log.debug("GPU graph result JSON number parse failed", e);
            return Double.NaN;
        }
    }

    /** Public test-access wrapper for {@link #findMatchingBracket}. */
    static int findMatchingBracketPublic(String s, int openPos) {
        return findMatchingBracket(s, openPos);
    }

    private static int findMatchingBracket(String s, int openPos) {
        if (s.charAt(openPos) != '[') return -1;
        int depth = 0;
        for (int i = openPos; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '[') depth++;
            else if (c == ']') {
                depth--;
                if (depth == 0) return i;
            }
        }
        return -1;
    }

}
