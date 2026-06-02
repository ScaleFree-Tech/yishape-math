package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;
import com.yishape.lab.math.compute.gpu.GpuConfig;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.util.YishapeLogger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

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
        "sigmoid", "tanh", "relu", "abs", "sqrt", "square",
        "sum", "mean", "dot", "matmul",
        "gelu", "softmax", "logSoftmax", "silu", "mish",
        "leakyRelu", "elu", "selu", "softplus", "hardtanh", "clamp",
        "softmaxCrossEntropy",
        "layerNorm", "batchNorm",
        "broadcast", "transpose", "reshape", "flatten",
        "leaf", "constant",
        // Fused elementwise + reduction ops (from AD pattern fusion)
        "absSum", "absMean", "reluSum", "reluMean", "logSum", "logMean",
        "sigmoidSum", "sigmoidMean", "tanhSum", "tanhMean",
        "siluSum", "siluMean", "mishSum", "mishMean",
        "expSum", "expMean", "squareSum", "squareMean",
        "mulSum", "mulMean", "powSum", "powMean"
    ));

    private static final ThreadLocal<ArrayList<RereDiffVector>> TOPO_LIST =
        ThreadLocal.withInitial(ArrayList::new);
    private static final ThreadLocal<HashSet<RereDiffVector>> TOPO_SET =
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

        // Check all ops are supported
        for (RereDiffVector v : order) {
            if (v.opTag != null && !SUPPORTED_OPS.contains(v.opTag)) {
                return Double.NaN;
            }
        }

        // Export graph to JSON
        String json = GraphExporter.toJson(root);
        if (json == null) return Double.NaN;

        // Execute on GPU
        String resultJson = GpuOptionalRuntime.tryExecuteGraph(json);
        if (resultJson == null) return Double.NaN;

        // Parse result and apply gradients to leaves
        try {
            return applyGradientsfromJson(root, order, resultJson);
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    /**
     * Parse GPU execution result JSON and apply gradients to leaf nodes.
     * Expected format: {"loss":double, "gradients":[[...], [...], ...]}
     * Simple parser without external JSON library dependency.
     */
    private static double applyGradientsfromJson(RereDiffVector root,
            ArrayList<RereDiffVector> order, String resultJson) {
        // Extract loss value
        double loss = extractDoubleField(resultJson, "loss");
        if (Double.isNaN(loss)) return Double.NaN;

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
