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

    private static final HashSet<String> SUPPORTED_OPS = new HashSet<>(Arrays.asList(
        "add", "sub", "mul", "div",
        "addScalar", "subScalar", "mulScalar", "divScalar", "rsubScalar", "rdivScalar",
        "neg", "pow", "exp", "log", "sin", "cos", "tan",
        "sigmoid", "tanh", "relu", "abs", "sqrt", "square", "dropout",
        "sum", "mean", "dot", "matmul",
        "gelu", "softmax", "logSoftmax", "silu", "mish",
        "leakyRelu", "elu", "selu", "softplus", "hardtanh", "clamp",
        "softmaxCrossEntropy",
        "normalize",
        "layerNorm",
        "broadcast", "transpose", "reshape", "flatten",
        "squeeze", "unsqueeze",
        "gather",
        "leaf", "constant",
        // Fused elementwise + reduction ops (from AD pattern fusion)
        "absSum", "absMean", "reluSum", "reluMean", "logSum", "logMean",
        "sigmoidSum", "sigmoidMean", "tanhSum", "tanhMean",
        "siluSum", "siluMean", "mishSum", "mishMean",
        "expSum", "expMean", "squareSum", "squareMean",
        "mulSum", "mulMean", "powSum", "powMean",
        // DL CustomOp layers (graphOpTag-based)
        "linear", "conv2d", "maxpool2d", "batchNorm2d", "embedding", "mha", "lstmStep",
        "selectiveScan", "selectiveScan2", "depthwiseConv1d",
        "scaledDotProductAttention",
        "softmaxCrossEntropy",
        // Concatenation
        "cat"
    ));

    /** Ops supported in tensor-native GPU execution (superset). */
    private static final HashSet<String> TENSOR_SUPPORTED_OPS = new HashSet<>(SUPPORTED_OPS);
    static {
        TENSOR_SUPPORTED_OPS.addAll(Arrays.asList(
            "permute", "expand", "reciprocal", "rsub", "rdiv",
            "gather", "scatter", "select", "slice", "narrow",
            "cat"
        ));
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
        if (!GpuConfig.allowAttempts()) return Double.NaN;
        if (!GpuOptionalRuntime.isGpuAvailable()) return Double.NaN;

        ArrayList<RereDiffTensor> order = TENSOR_TOPO.get();
        order.clear();
        HashSet<RereDiffTensor> visited = TENSOR_VISITED.get();
        visited.clear();
        root.buildTopo(order, visited);

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

        // Check ops first (common to both binary and JSON paths)
        // TEMPORARY: print all unique ops in the graph for debugging
        java.util.LinkedHashSet<String> seenOps = new java.util.LinkedHashSet<>();
        for (RereDiffTensor v : order) {
            if (v.opTag() != null) seenOps.add(v.opTag());
        }
        System.err.println("[GPU-OPS] Graph ops: " + seenOps);
        for (RereDiffTensor v : order) {
            if (v.opTag() != null && !TENSOR_SUPPORTED_OPS.contains(v.opTag())) {
                int leafCount = 0;
                for (RereDiffTensor n : order) {
                    if (n.isLeaf()) leafCount++;
                }
                log.warn("GPU tensor graph fallback: unsupported op='{}', graph has {} nodes ({} leaves)",
                    v.opTag(), order.size(), leafCount);
                System.err.println("[GPU-UNSUPPORTED] '" + v.opTag() + "' graph=" + order.size() + " leaves=" + leafCount);
                return Double.NaN;
            }
        }

        // Try binary path first
        // TEMPORARY: force JSON path to isolate binary protocol issues
        double binaryResult = Double.NaN; // tryExecuteTensorBinary(root, order);
        // if (!Double.isNaN(binaryResult)) return binaryResult;

        // Collect leaves
        ArrayList<RereDiffTensor> leaves = new ArrayList<>();
        for (RereDiffTensor v : order) {
            if (v.isLeaf()) {
                leaves.add(v);
            }
        }

        // Export and execute
        String json = TensorGraphExporter.toJson(root);
        if (json == null) {
            log.debug("GPU tensor graph fallback: JSON export failed");
            return Double.NaN;
        }
        if (order.size() > BINARY_THRESHOLD) {
            log.warn("GPU graph has {} nodes (>{}); JSON path should be avoided — "
                + "binary protocol is preferred. Check why binary path failed.", order.size(), BINARY_THRESHOLD);
        }

        String resultJson = GpuOptionalRuntime.tryExecuteGraph(json);
        if (resultJson == null) {
            log.debug("GPU tensor graph fallback: Rust execution returned null");
            return Double.NaN;
        }

        try {
            // NOTE: Do NOT derive batchSize from root.scalarParam.
            // scalarParam is overloaded — it stores op-specific parameters:
            //   exponent for pow/powSum, divisor n for mean/div, alpha for activations.
            // Treating it as batchSize would incorrectly divide loss/grads (e.g. powSum
            // with scalarParam=2 gives halved results). Each GPU op must be self-contained.
            return applyTensorGradientsFromJson(root, leaves, resultJson);
        } catch (Exception e) {
            log.debug("GPU graph execution failed", e);
            return Double.NaN;
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
            String[] tokens = inner.split(",");
            double[] gradData = new double[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                gradData[i] = Double.parseDouble(tokens[i].trim());
            }

            leaves.get(leafIdx).accGrad(gradData);
            leafIdx++;
            pos = innerEnd + 1;
        }

        if (leafIdx < leaves.size()) {
            log.warn("GPU tensor returned fewer gradients ({}) than leaves ({})", leafIdx, leaves.size());
        }
        return loss;
    }

    /** Binary graph execution for tensors using YSGP protocol. */
    private static double tryExecuteTensorBinary(RereDiffTensor root,
            ArrayList<RereDiffTensor> order) {
        try {
            java.nio.ByteBuffer buf = TensorBinaryProtocol.serializeGraph(root, order);
            byte[] data = new byte[buf.remaining()];
            buf.get(data);

            byte[] resultBytes = GpuOptionalRuntime.tryExecuteGraphBinary(data);
            if (resultBytes == null || resultBytes.length == 0) return Double.NaN;

            java.nio.ByteBuffer resultBuf = java.nio.ByteBuffer.wrap(resultBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN);

            var parsed = com.yishape.lab.math.autodiff.graph.binary.BinaryProtocol.deserializeResult(resultBuf);
            double loss = parsed.loss();
            if (Double.isNaN(loss)) return Double.NaN;

            // Collect leaves — NOTE: no batchSize normalization.
            // root.scalarParam stores op-specific parameters (exponent, divisor, alpha),
            // NOT batch size. Each GPU op must produce correct loss/grads independently.
            java.util.ArrayList<RereDiffTensor> leaves = new java.util.ArrayList<>();
            for (RereDiffTensor v : order) { if (v.isLeaf()) leaves.add(v); }
            java.util.List<double[]> grads = parsed.grads();
            if (grads.size() != leaves.size()) return Double.NaN;

            for (int i = 0; i < grads.size(); i++) {
                double[] g = grads.get(i);
                leaves.get(i).accGrad(g);
            }
            return loss;
        } catch (Exception e) {
            log.debug("GPU graph execution failed", e);
            return Double.NaN;
        }
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
