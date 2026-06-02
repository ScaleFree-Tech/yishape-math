package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

import com.yishape.lab.math.compute.hpc.HpcAutodiff;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.util.YishapeLogger;

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
        "sigmoid", "tanh", "relu", "abs", "sqrt", "square",
        "sum", "mean", "squareSum", "squareMean",
        "absSum", "absMean", "reluSum", "reluMean", "logSum", "logMean",
        "sigmoidSum", "sigmoidMean", "tanhSum", "tanhMean",
        "siluSum", "siluMean", "mishSum", "mishMean",
        "expSum", "expMean", "powSum", "powMean", "mulSum", "mulMean",
        "dot", "broadcast", "matmul", "transpose", "reshape", "flatten",
        "softmaxCrossEntropy",
        "gelu", "softmax", "logSoftmax", "leakyRelu", "elu", "selu",
        "silu", "mish", "softplus", "hardtanh", "clamp", "layerNorm", "batchNorm",
        "leaf", "constant"
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
        ArrayList<RereDiffVector> order = HPC_TOPO_LIST.get();
        order.clear();
        HashSet<RereDiffVector> visited = HPC_TOPO_SET.get();
        visited.clear();
        root.buildTopo(order, visited);

        // Early check: if any node has an op tag not supported by Rust, skip HPC
        for (RereDiffVector v : order) {
            if (v.opTag != null && !SUPPORTED_OPS.contains(v.opTag)) {
                return Double.NaN;
            }
        }

        ArrayList<RereDiffVector> leaves = HPC_LEAVES.get();
        leaves.clear();
        for (RereDiffVector v : order) {
            if (v.isLeaf) {
                leaves.add(v);
            }
        }

        String json = GraphExporter.toJson(root);
        double[][] result = HpcAutodiff.tryExecute(json);
        if (result == null || result.length < 2 || result[0] == null) {
            return Double.NaN;
        }

        double loss = result[0][0];
        int numGrads = Math.min(result.length - 1, leaves.size());
        if (result.length - 1 != leaves.size()) {
            log.warn("HPC gradient count mismatch: got {} gradients for {} leaves",
                    result.length - 1, leaves.size());
        }
        for (int i = 0; i < numGrads; i++) {
            if (result[i + 1] != null) {
                int gradLen = result[i + 1].length;
                int leafLen = leaves.get(i).value.size();
                if (gradLen != leafLen) {
                    log.warn("HPC gradient length mismatch at leaf {}: got {} expected {}",
                            i, gradLen, leafLen);
                }
                leaves.get(i).accGrad(IDoubleVector.of(result[i + 1]));
            }
        }
        return loss;
    }
}
