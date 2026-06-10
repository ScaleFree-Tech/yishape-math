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

    private static final HashSet<String> SUPPORTED_OPS = new HashSet<>(Arrays.asList(
        "add", "sub", "mul", "div",
        "addScalar", "subScalar", "mulScalar", "divScalar", "rsubScalar", "rdivScalar",
        "neg", "pow", "exp", "log", "sin", "cos", "tan",
        "sigmoid", "tanh", "relu", "abs", "sqrt", "square", "dropout",
        "sum", "mean", "squareSum", "squareMean",
        "absSum", "absMean", "reluSum", "reluMean", "logSum", "logMean",
        "sigmoidSum", "sigmoidMean", "tanhSum", "tanhMean",
        "siluSum", "siluMean", "mishSum", "mishMean",
        "expSum", "expMean", "powSum", "powMean", "mulSum", "mulMean",
        "logSumExp",
        "dot", "broadcast", "matmul", "transpose", "reshape", "flatten",
        "squeeze", "unsqueeze",
        "softmaxCrossEntropy",
        "gelu", "softmax", "logSoftmax", "leakyRelu", "elu", "selu",
        "silu", "mish", "softplus", "hardtanh", "clamp", "layerNorm",
        "mmul", // RereDiffTensor.mmul() uses "mmul" (not "matmul")
        "leaf", "constant",
        // DL CustomOp layers (graphOpTag-based)
        "linear", "conv2d", "maxpool2d", "avgpool2d", "batchNorm2d", "embedding", "mha", "lstmStep",
        "selectiveScan", "selectiveScan2", "depthwiseConv1d"
    ));

    /** Ops supported in tensor-native HPC execution (superset). */
    private static final HashSet<String> TENSOR_SUPPORTED_OPS = new HashSet<>(SUPPORTED_OPS);
    static {
        // Additional tensor-native operations also available in Rust HPC
        TENSOR_SUPPORTED_OPS.addAll(Arrays.asList(
            "permute", "expand", "reciprocal", "rsub", "rdiv",
            "gather", "scatter", "select", "slice", "narrow",
            "cat", "contiguous",
            "batchNorm", "groupNorm"
        ));
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

        ArrayList<RereDiffTensor> order = HPC_TENSOR_TOPO.get();
        order.clear();
        HashSet<RereDiffTensor> visited = HPC_TENSOR_VISITED.get();
        visited.clear();
        root.buildTopo(order, visited);

        // Validate graph structure before execution:
        // root-is-scalar, non-leaf root, in-bounds input references.
        ExportShapeValidator.Result validation = ExportShapeValidator.validate(order);
        if (validation.hasErrors()) {
            System.err.println("[HPC-VALIDATE-FAIL] " + validation.toString().replace("\n", "\n[HPC-VALIDATE-FAIL] "));
            log.warn("HPC graph validation failed — falling back to CPU:\n{}", validation);
            return Double.NaN;
        }

        // Check unsupported ops
        for (RereDiffTensor v : order) {
            if (v.opTag() != null && !TENSOR_SUPPORTED_OPS.contains(v.opTag())) {
                System.err.println("[HPC-UNSUPPORTED-OP] op='" + v.opTag() + "' nodes=" + order.size());
                if (log.isDebugEnabled()) {
                    int leafCount = 0;
                    for (RereDiffTensor n : order) {
                        if (n.isLeaf()) leafCount++;
                    }
                    log.debug("HPC tensor graph fallback: unsupported op='{}', graph has {} nodes ({} leaves)",
                        v.opTag(), order.size(), leafCount);
                }
                return Double.NaN;
            }
        }

        // Collect leaves
        ArrayList<RereDiffTensor> leaves = new ArrayList<>();
        for (RereDiffTensor v : order) {
            if (v.isLeaf()) {
                leaves.add(v);
            }
        }

        // Try binary path first
        double binaryResult = tryExecuteTensorBinary(root, order, leaves);
        if (!Double.isNaN(binaryResult)) return binaryResult;
        if (log.isDebugEnabled()) log.debug("Binary path returned NaN, falling back to JSON (nodes={})", order.size());

        // JSON fallback
        String json = TensorGraphExporter.toJson(root);
        if (json == null) {
            System.err.println("[HPC-JSON-FAIL] TensorGraphExporter.toJson returned null for " + order.size() + " nodes");
            log.debug("HPC tensor graph fallback: JSON export failed");
            return Double.NaN;
        }
        if (order.size() > BINARY_THRESHOLD) {
            log.warn("HPC graph has {} nodes (>{}); JSON path should be avoided — "
                + "binary protocol is preferred. Check why binary path failed.", order.size(), BINARY_THRESHOLD);
        }
        double[][] result = HpcAutodiff.tryExecute(json);
        if (result == null || result.length < 2 || result[0] == null) {
            System.err.println("[HPC-EXEC-FAIL] HpcAutodiff.tryExecute returned " + (result == null ? "null" : "invalid: len=" + result.length) + " for " + order.size() + " nodes, json=" + json.length() + " chars");
            log.debug("HPC tensor graph fallback: Rust execution returned null or invalid result");
            return Double.NaN;
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
            return Double.NaN;
        }
        for (int i = 0; i < leaves.size(); i++) {
            if (result[i + 1] == null) {
                log.warn("HPC tensor gradient[{}] is null — falling back to CPU", i);
                return Double.NaN;
            }
            int gradLen = result[i + 1].length;
            int leafLen = Math.toIntExact(leaves.get(i).totalSize());
            if (gradLen != leafLen) {
                log.warn("HPC tensor gradient length mismatch at leaf {}: got {} expected {} — falling back to CPU",
                        i, gradLen, leafLen);
                return Double.NaN;
            }
            leaves.get(i).accGrad(result[i + 1]);
        }
        return loss;
    }

    /** Binary graph execution for tensors using YSGP protocol. */
    private static double tryExecuteTensorBinary(RereDiffTensor root,
            ArrayList<RereDiffTensor> order, ArrayList<RereDiffTensor> leaves) {
        try {
            ByteBuffer buf = TensorBinaryProtocol.serializeGraph(root, order);
            byte[] data = new byte[buf.remaining()];
            buf.get(data);
            byte[] resultBytes = HpcOptionalRuntime.tryExecuteGraphBinary(data);
            if (resultBytes == null || resultBytes.length == 0) {
                System.err.println("[HPC-BINARY-FAIL] HpcOptionalRuntime.tryExecuteGraphBinary returned " + (resultBytes == null ? "null" : "empty") + " for " + order.size() + " nodes, " + data.length + " bytes");
                return Double.NaN;
            }

            ByteBuffer resultBuf = ByteBuffer.wrap(resultBytes).order(ByteOrder.LITTLE_ENDIAN);

            var parsed = BinaryProtocol.deserializeResult(resultBuf);
            double loss = parsed.loss();
            if (Double.isNaN(loss)) return Double.NaN;

            // No batchSize normalization — scalarParam stores op parameters, not batch size.
            // Each HPC op computes correctly-scaled loss and grads independently.
            List<double[]> grads = parsed.grads();
            if (grads.size() != leaves.size()) return Double.NaN;

            for (int i = 0; i < grads.size(); i++) {
                double[] g = grads.get(i);
                leaves.get(i).accGrad(g);
            }
            return loss;
        } catch (Exception e) {
            if (log.isDebugEnabled()) log.debug("HPC graph execution failed", e);
            return Double.NaN;
        }
    }

}
