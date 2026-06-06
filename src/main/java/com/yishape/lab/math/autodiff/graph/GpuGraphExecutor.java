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
        "softmaxCrossEntropy"
    ));

    /** Ops supported in tensor-native GPU execution (superset). */
    private static final HashSet<String> TENSOR_SUPPORTED_OPS = new HashSet<>(SUPPORTED_OPS);
    static {
        TENSOR_SUPPORTED_OPS.addAll(Arrays.asList(
            "permute", "expand", "reciprocal", "rsub", "rdiv",
            "gather", "scatter", "select", "slice", "narrow"
        ));
    }

    // Vector thread locals
    private static final ThreadLocal<ArrayList<RereDiffVector>> TOPO_LIST =
        ThreadLocal.withInitial(ArrayList::new);
    private static final ThreadLocal<HashSet<RereDiffVector>> TOPO_SET =
        ThreadLocal.withInitial(HashSet::new);

    // Tensor thread locals
    private static final ThreadLocal<ArrayList<RereDiffTensor>> TENSOR_TOPO =
        ThreadLocal.withInitial(ArrayList::new);
    private static final ThreadLocal<HashSet<RereDiffTensor>> TENSOR_VISITED =
        ThreadLocal.withInitial(HashSet::new);

    private GpuGraphExecutor() {}

    /**
     * Attempt GPU graph execution. Returns loss value or NaN if unavailable.
     */
    public static double tryExecute(RereDiffVector root) {
        if (!GpuConfig.allowAttempts()) return Double.NaN;
        if (!GpuOptionalRuntime.isGpuAvailable()) return Double.NaN;

        ArrayList<RereDiffVector> order = TOPO_LIST.get();
        order.clear();
        HashSet<RereDiffVector> visited = TOPO_SET.get();
        visited.clear();
        root.buildTopo(order, visited);

        // Try binary path first (faster, no subnormal double corruption)
        double binaryResult = tryExecuteBinary(root, order);
        if (!Double.isNaN(binaryResult)) return binaryResult;

        // Fall back to JSON path (reuse order from binary attempt)
        // Check all ops are supported
        for (RereDiffVector v : order) {
            if (v.opTag != null && !SUPPORTED_OPS.contains(v.opTag)) {
                if (log.isDebugEnabled()) {
                    int leafCount = 0;
                    for (RereDiffVector n : order) {
                        if (n.isLeaf) leafCount++;
                    }
                    log.debug("GPU graph fallback: unsupported op='{}', graph has {} nodes ({} leaves)",
                        v.opTag, order.size(), leafCount);
                }
                return Double.NaN;
            }
        }

        // Export graph to JSON
        String json = GraphExporter.toJson(root);
        if (json == null) {
            log.debug("GPU graph fallback: JSON export failed");
            return Double.NaN;
        }

        // Execute on GPU
        String resultJson = GpuOptionalRuntime.tryExecuteGraph(json);
        if (resultJson == null) {
            log.debug("GPU graph fallback: Rust execution returned null");
            return Double.NaN;
        }

        // Parse result and apply gradients to leaves
        try {
            double batchSize = !Double.isNaN(root.scalarParam) ? root.scalarParam : 1.0;
            return applyGradientsfromJson(root, order, resultJson, batchSize);
        } catch (Exception e) {
            log.debug("GPU graph execution failed", e);
            return Double.NaN;
        }
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

        // Try binary path first
        double binaryResult = tryExecuteTensorBinary(root, order);
        if (!Double.isNaN(binaryResult)) return binaryResult;

        // Check ops
        for (RereDiffTensor v : order) {
            if (v.opTag != null && !TENSOR_SUPPORTED_OPS.contains(v.opTag)) {
                if (log.isDebugEnabled()) {
                    int leafCount = 0;
                    for (RereDiffTensor n : order) {
                        if (n.isLeaf) leafCount++;
                    }
                    log.debug("GPU tensor graph fallback: unsupported op='{}', graph has {} nodes ({} leaves)",
                        v.opTag, order.size(), leafCount);
                }
                return Double.NaN;
            }
        }

        // Collect leaves
        ArrayList<RereDiffTensor> leaves = new ArrayList<>();
        for (RereDiffTensor v : order) {
            if (v.isLeaf) {
                leaves.add(v);
            }
        }

        // Export and execute
        String json = TensorGraphExporter.toJson(root);
        if (json == null) {
            log.debug("GPU tensor graph fallback: JSON export failed");
            return Double.NaN;
        }

        String resultJson = GpuOptionalRuntime.tryExecuteGraph(json);
        if (resultJson == null) {
            log.debug("GPU tensor graph fallback: Rust execution returned null");
            return Double.NaN;
        }

        try {
            double batchSize = !Double.isNaN(root.scalarParam) ? root.scalarParam : 1.0;
            return applyTensorGradientsFromJson(root, leaves, resultJson, batchSize);
        } catch (Exception e) {
            log.debug("GPU graph execution failed", e);
            return Double.NaN;
        }
    }

    /** Parse JSON result and apply gradients to tensor leaves. */
    private static double applyTensorGradientsFromJson(RereDiffTensor root,
            ArrayList<RereDiffTensor> leaves, String resultJson, double batchSize) {
        double loss = extractDoubleField(resultJson, "loss");
        if (Double.isNaN(loss)) return Double.NaN;
        if (batchSize > 1.0) loss /= batchSize;

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

            if (batchSize > 1.0) {
                for (int i = 0; i < gradData.length; i++) gradData[i] /= batchSize;
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

            // Collect leaves
            java.util.ArrayList<RereDiffTensor> leaves = new java.util.ArrayList<>();
            for (RereDiffTensor v : order) { if (v.isLeaf) leaves.add(v); }
            java.util.List<double[]> grads = parsed.grads();
            if (grads.size() != leaves.size()) return Double.NaN;

            double batchSize = !Double.isNaN(root.scalarParam) ? root.scalarParam : 1.0;
            if (batchSize > 1.0) loss /= batchSize;

            for (int i = 0; i < grads.size(); i++) {
                double[] g = grads.get(i);
                if (batchSize > 1.0) { for (int j = 0; j < g.length; j++) g[j] /= batchSize; }
                leaves.get(i).accGrad(g);
            }
            return loss;
        } catch (Exception e) {
            log.debug("GPU graph execution failed", e);
            return Double.NaN;
        }
    }

    /**
     * Parse GPU execution result JSON and apply gradients to leaf nodes.
     * Expected format: {"loss":double, "gradients":[[...], [...], ...]}
     * Simple parser without external JSON library dependency.
     */
    private static double applyGradientsfromJson(RereDiffVector root,
            ArrayList<RereDiffVector> order, String resultJson, double batchSize) {
        double loss = extractDoubleField(resultJson, "loss");
        if (Double.isNaN(loss)) return Double.NaN;
        if (batchSize > 1.0) loss /= batchSize;

        // Extract gradients array — GPU returns "grads", HPC returns "gradients"
        int gradStart = resultJson.indexOf("\"grads\"");
        if (gradStart < 0) gradStart = resultJson.indexOf("\"gradients\"");
        if (gradStart < 0) return Double.NaN;
        int arrStart = resultJson.indexOf('[', gradStart);
        if (arrStart < 0) return Double.NaN;
        int arrEnd = findMatchingBracket(resultJson, arrStart);
        if (arrEnd < 0) return Double.NaN;

        // Collect leaf nodes
        ArrayList<RereDiffVector> leaves = new ArrayList<>();
        for (RereDiffVector v : order) {
            if (v.isLeaf) {
                leaves.add(v);
            }
        }

        // Diagnostic: print leaf ordering
        boolean diag = Boolean.getBoolean("yishape.grad.diag");
        if (diag) {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("[GpuLeafOrder] %d leaves in topo order:\n", leaves.size()));
            for (int i = 0; i < leaves.size(); i++) {
                RereDiffVector leaf = leaves.get(i);
                String tag = leaf.opTag != null ? leaf.opTag : "null";
                sb.append(String.format("  leaves[%d] opTag=%s size=%d isLeaf=%b\n",
                    i, tag, leaf.value.size(), leaf.isLeaf));
            }
            System.out.print(sb);
        }

        // Parse each gradient array and apply to corresponding leaf
        int pos = arrStart + 1;
        int leafIdx = 0;
        while (pos < arrEnd && leafIdx < leaves.size()) {
            // Skip whitespace and commas
            while (pos < arrEnd && (resultJson.charAt(pos) == ',' || Character.isWhitespace(resultJson.charAt(pos)))) {
                pos++;
            }
            if (pos >= arrEnd || resultJson.charAt(pos) != '[') break;
            int innerEnd = findMatchingBracket(resultJson, pos);
            if (innerEnd < 0) break;

            // Parse double values from inner array
            String inner = resultJson.substring(pos + 1, innerEnd);
            String[] tokens = inner.split(",");
            double[] gradData = new double[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                gradData[i] = Double.parseDouble(tokens[i].trim());
            }

            // Apply gradient to leaf
            int gradLen = gradData.length;
            int leafLen = leaves.get(leafIdx).value.size();
            if (gradLen != leafLen) {
                log.warn("GPU gradient length mismatch at leaf {}: got {} expected {}",
                        leafIdx, gradLen, leafLen);
            }
            if (diag) {
                double nrm = 0;
                for (double v : gradData) nrm += v * v;
                System.out.printf("[GpuGradMap] leaf[%d] size=%d gradLen=%d gradL2=%.6e%n",
                    leafIdx, leaves.get(leafIdx).value.size(), gradLen, Math.sqrt(nrm));
            }
            if (batchSize > 1.0) {
                for (int i = 0; i < gradData.length; i++) gradData[i] /= batchSize;
            }
            leaves.get(leafIdx).accGrad(IDoubleVector.of(gradData));
            leafIdx++;
            pos = innerEnd + 1;
        }

        if (leafIdx < leaves.size()) {
            log.warn("GPU returned fewer gradients ({}) than leaves ({})", leafIdx, leaves.size());
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

    /** Binary graph execution using YSGP protocol. Returns loss or NaN on failure. */
    private static double tryExecuteBinary(RereDiffVector root, ArrayList<RereDiffVector> order) {
        try {
            java.nio.ByteBuffer buf = com.yishape.lab.math.autodiff.graph.binary.BinaryProtocol.serializeGraph(root, order);
            byte[] data = new byte[buf.remaining()];
            buf.get(data);
            
            byte[] resultBytes = GpuOptionalRuntime.tryExecuteGraphBinary(data);
            if (resultBytes == null || resultBytes.length == 0) return Double.NaN;
            
            java.nio.ByteBuffer resultBuf = java.nio.ByteBuffer.wrap(resultBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            
            var parsed = com.yishape.lab.math.autodiff.graph.binary.BinaryProtocol.deserializeResult(resultBuf);
            double loss = parsed.loss();
            if (Double.isNaN(loss)) return Double.NaN;
            
            // Apply gradients to leaves
            java.util.ArrayList<RereDiffVector> leaves = new java.util.ArrayList<>();
            for (RereDiffVector v : order) { if (v.isLeaf) leaves.add(v); }
            java.util.List<double[]> grads = parsed.grads();
            if (grads.size() != leaves.size()) return Double.NaN;
            
            double batchSize = !Double.isNaN(root.scalarParam) ? root.scalarParam : 1.0;
            if (batchSize > 1.0) loss /= batchSize;
            
            for (int i = 0; i < grads.size(); i++) {
                double[] g = grads.get(i);
                if (batchSize > 1.0) { for (int j = 0; j < g.length; j++) g[j] /= batchSize; }
                leaves.get(i).accGrad(IDoubleVector.of(g));
            }
            return loss;
        } catch (Exception e) {
            log.debug("GPU graph execution failed", e);
            return Double.NaN;
        }
    }
}
