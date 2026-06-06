package com.yishape.lab.math.autodiff.graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;

/**
 * Computation-graph inspection and rewrite utilities for tensor-based AD graphs.
 * 张量计算图分析与图优化工具。
 *
 * <p>Mirrors {@link GraphOptimizer} but operates on {@link RereDiffTensor} nodes,
 * which naturally carry N-D shape information.
 */
public class TensorGraphOptimizer {

    private TensorGraphOptimizer() {}

    /** Graph-level constant folding: eliminates identity operations (addScalar(0), mulScalar(1), etc.). */
    public static IDiffTensor optimize(IDiffTensor root) {
        if (!(root instanceof RereDiffTensor r)) {
            return root;
        }
        return foldConstants(r);
    }

    private static RereDiffTensor foldConstants(RereDiffTensor root) {
        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);

        Map<RereDiffTensor, RereDiffTensor> replacements = new HashMap<>();

        for (RereDiffTensor node : order) {
            if (node.isLeaf) {
                replacements.put(node, node);
                continue;
            }

            String tag = node.opTag;
            double sp = node.scalarParam;
            int nInputs = node.inputs != null ? node.inputs.size() : 0;

            // addScalar(0) → pass-through input
            if ("addScalar".equals(tag) && sp == 0.0 && nInputs == 1) {
                RereDiffTensor rep = replacements.getOrDefault(node.inputs.get(0), node.inputs.get(0));
                replacements.put(node, rep);
                continue;
            }
            // subScalar(0) → pass-through input
            if ("subScalar".equals(tag) && sp == 0.0 && nInputs == 1) {
                RereDiffTensor rep = replacements.getOrDefault(node.inputs.get(0), node.inputs.get(0));
                replacements.put(node, rep);
                continue;
            }
            // mulScalar(1) → pass-through input
            if ("mulScalar".equals(tag) && sp == 1.0 && nInputs == 1) {
                RereDiffTensor rep = replacements.getOrDefault(node.inputs.get(0), node.inputs.get(0));
                replacements.put(node, rep);
                continue;
            }
            // divScalar(1) → pass-through input
            if ("divScalar".equals(tag) && sp == 1.0 && nInputs == 1) {
                RereDiffTensor rep = replacements.getOrDefault(node.inputs.get(0), node.inputs.get(0));
                replacements.put(node, rep);
                continue;
            }
            // mulScalar(0) → zeros constant (detach from input)
            if ("mulScalar".equals(tag) && sp == 0.0 && nInputs == 1) {
                int totalSize = Math.toIntExact(node.totalSize());
                RereDiffTensor zeros = new RereDiffTensor(new double[totalSize], node.shape());
                zeros.opTag = "constant";
                zeros.isLeaf = true;
                replacements.put(node, zeros);
                continue;
            }
            // pow(1) → pass-through input
            if ("pow".equals(tag) && sp == 1.0 && nInputs == 1) {
                RereDiffTensor rep = replacements.getOrDefault(node.inputs.get(0), node.inputs.get(0));
                replacements.put(node, rep);
                continue;
            }

            // Rewrite inputs to use replacements (transitive folding)
            boolean changed = false;
            for (int i = 0; i < nInputs; i++) {
                RereDiffTensor replaced = replacements.get(node.inputs.get(i));
                if (replaced != null && replaced != node.inputs.get(i)) {
                    if (!changed) {
                        node.inputs = new ArrayList<>(node.inputs);
                        changed = true;
                    }
                    node.inputs.set(i, replaced);
                }
            }
            replacements.put(node, node);
        }

        return replacements.getOrDefault(root, root);
    }

    public static int countNodes(IDiffTensor root) {
        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        ((RereDiffTensor) root).buildTopo(order, visited);
        return order.size();
    }

    public static int countLeaves(IDiffTensor root) {
        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        ((RereDiffTensor) root).buildTopo(order, visited);
        int count = 0;
        for (RereDiffTensor v : order) {
            if (v.isLeaf) count++;
        }
        return count;
    }

    public static GraphStats stats(IDiffTensor root) {
        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        ((RereDiffTensor) root).buildTopo(order, visited);
        int leaves = 0, nonLeaf = 0;
        for (RereDiffTensor v : order) {
            if (v.isLeaf) leaves++;
            else nonLeaf++;
        }
        return new GraphStats(order.size(), leaves, nonLeaf);
    }

    /** Summary statistics of a tensor autodiff graph. / 张量计算图统计摘要。 */
    public record GraphStats(int totalNodes, int leafNodes, int nonLeafNodes) {
        @Override
        public String toString() {
            return String.format("TensorGraph: %d nodes (%d leaves, %d ops)", totalNodes, leafNodes, nonLeafNodes);
        }
    }
}
