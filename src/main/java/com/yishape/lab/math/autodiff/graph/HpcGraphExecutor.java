package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.graph.binary.BinaryProtocol;
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
        "dot", "broadcast", "matmul", "transpose", "reshape", "flatten",
        "squeeze", "unsqueeze",
        "softmaxCrossEntropy",
        "gelu", "softmax", "logSoftmax", "leakyRelu", "elu", "selu",
        "silu", "mish", "softplus", "hardtanh", "clamp", "layerNorm",
        "leaf", "constant",
        // DL CustomOp layers (graphOpTag-based)
        "linear", "conv2d", "maxpool2d", "batchNorm2d", "embedding", "mha", "lstmStep",
        "selectiveScan", "selectiveScan2", "depthwiseConv1d"
    ));

    private static final ThreadLocal<ArrayList<RereDiffVector>> HPC_TOPO_LIST =
        ThreadLocal.withInitial(ArrayList::new);
    private static final ThreadLocal<HashSet<RereDiffVector>> HPC_TOPO_SET =
        ThreadLocal.withInitial(HashSet::new);
    private static final ThreadLocal<ArrayList<RereDiffVector>> HPC_LEAVES =
        ThreadLocal.withInitial(ArrayList::new);

    private HpcGraphExecutor() {
    }

    /**
     * Attempt HPC graph execution. On success, applies gradients to all leaf nodes
     * and returns the loss value.
     *
     * @param root the graph root node
     * @return loss value, or {@link Double#NaN} if HPC is unavailable or fails
     */
    public static double tryExecute(RereDiffVector root) {
        if (!com.yishape.lab.math.compute.hpc.HpcConfig.allowAttempts()) return Double.NaN;
        ArrayList<RereDiffVector> order = HPC_TOPO_LIST.get();
        order.clear();
        HashSet<RereDiffVector> visited = HPC_TOPO_SET.get();
        visited.clear();
        root.buildTopo(order, visited);

        // Early check: if any node has an op tag not supported by Rust, skip HPC
        for (RereDiffVector v : order) {
            if (v.opTag != null && !SUPPORTED_OPS.contains(v.opTag)) {
                if (log.isDebugEnabled()) {
                    int leafCount = 0;
                    for (RereDiffVector n : order) {
                        if (n.isLeaf) leafCount++;
                    }
                    log.debug("HPC graph fallback: unsupported op='{}', graph has {} nodes ({} leaves)",
                        v.opTag, order.size(), leafCount);
                }
                return Double.NaN;
            }
        }

        // Try binary path first (faster, exact scalar params)
        double binaryResult = tryExecuteBinary(root, order);
        if (!Double.isNaN(binaryResult)) return binaryResult;

        ArrayList<RereDiffVector> leaves = HPC_LEAVES.get();
        leaves.clear();
        for (RereDiffVector v : order) {
            if (v.isLeaf) {
                leaves.add(v);
            }
        }

        // Diagnostic: print leaf ordering
        boolean diag = Boolean.getBoolean("yishape.grad.diag");
        if (diag) {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("[HpcLeafOrder] %d leaves in topo order:\n", leaves.size()));
            for (int i = 0; i < leaves.size(); i++) {
                RereDiffVector leaf = leaves.get(i);
                String tag = leaf.opTag != null ? leaf.opTag : "null";
                sb.append(String.format("  leaves[%d] opTag=%s size=%d isLeaf=%b\n",
                    i, tag, leaf.value.size(), leaf.isLeaf));
            }
            System.out.print(sb);
        }

        // JSON fallback path
        String json = GraphExporter.toJson(root);
        if (json == null) {
            log.debug("HPC graph fallback: JSON export failed");
            return Double.NaN;
        }
        double[][] result = HpcAutodiff.tryExecute(json);
        if (result == null || result.length < 2 || result[0] == null) {
            log.debug("HPC graph fallback: Rust execution returned null or invalid result");
            return Double.NaN;
        }

        double batchSize = !Double.isNaN(root.scalarParam) ? root.scalarParam : 1.0;
        double loss = result[0][0];
        // HPC forward already returns MEAN loss, no need to divide again
        int numGrads = Math.min(result.length - 1, leaves.size());
        if (result.length - 1 != leaves.size()) {
            log.warn("HPC gradient count mismatch: got {} gradients for {} leaves",
                    result.length - 1, leaves.size());
        }
        // Diagnostic: print leaf details and gradient norms
        if (diag) {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("[HpcGradDiag] loss=%.6f leaves=%d grads=%d\n", loss, leaves.size(), result.length - 1));
            for (int i = 0; i < leaves.size(); i++) {
                RereDiffVector leaf = leaves.get(i);
                String tag = leaf.opTag != null ? leaf.opTag : (leaf.isLeaf ? "leaf" : "unknown");
                int leafSize = leaf.value.size();
                double gradNorm = 0;
                int gradLen = 0;
                if (i < numGrads && result[i + 1] != null) {
                    gradLen = result[i + 1].length;
                    for (double v : result[i + 1]) gradNorm += v * v;
                    gradNorm = Math.sqrt(gradNorm);
                }
                sb.append(String.format("  leaf[%d] op=%s size=%d gradLen=%d gradL2=%.6e\n",
                        i, tag, leafSize, gradLen, gradNorm));
            }
            System.out.print(sb);
        }
        for (int i = 0; i < numGrads; i++) {
            if (result[i + 1] != null) {
                int gradLen = result[i + 1].length;
                int leafLen = leaves.get(i).value.size();
                if (gradLen != leafLen) {
                    log.warn("HPC gradient length mismatch at leaf {}: got {} expected {}",
                            i, gradLen, leafLen);
                }
                if (batchSize > 1.0) {
                    for (int j = 0; j < gradLen; j++) result[i + 1][j] /= batchSize;
                }
                leaves.get(i).accGrad(IDoubleVector.of(result[i + 1]));
            }
        }
        return loss;
    }

    /** Binary graph execution using YSGP protocol. Returns loss or NaN on failure. */
    private static double tryExecuteBinary(RereDiffVector root, ArrayList<RereDiffVector> order) {
        try {
            ByteBuffer buf = BinaryProtocol.serializeGraph(root, order);
            byte[] data = new byte[buf.remaining()];
            buf.get(data);

            byte[] resultBytes = HpcOptionalRuntime.tryExecuteGraphBinary(data);
            if (resultBytes == null || resultBytes.length == 0) return Double.NaN;

            ByteBuffer resultBuf = ByteBuffer.wrap(resultBytes).order(ByteOrder.LITTLE_ENDIAN);

            var parsed = BinaryProtocol.deserializeResult(resultBuf);
            double loss = parsed.loss();
            if (Double.isNaN(loss)) return Double.NaN;

            // Apply gradients to leaves
            ArrayList<RereDiffVector> leaves = new ArrayList<>();
            for (RereDiffVector v : order) { if (v.isLeaf) leaves.add(v); }
            List<double[]> grads = parsed.grads();
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
            return Double.NaN;
        }
    }
}
