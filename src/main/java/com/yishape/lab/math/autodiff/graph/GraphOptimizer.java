package com.yishape.lab.math.autodiff.graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;

/**
 * Computation-graph inspection and rewrite utilities.
 * 计算图分析与图优化工具。
 */
public class GraphOptimizer {

    private GraphOptimizer() {}

    /** Graph-level constant folding: eliminates identity operations (addScalar(0), mulScalar(1), etc.). */
    public static IDiffVector optimize(IDiffVector root) {
        if (!(root instanceof RereDiffVector r)) {
            return root;
        }
        return foldConstants(r);
    }

    private static RereDiffVector foldConstants(RereDiffVector root) {
        List<RereDiffVector> order = new ArrayList<>();
        Set<RereDiffVector> visited = new HashSet<>();
        root.buildTopo(order, visited);

        Map<RereDiffVector, RereDiffVector> replacements = new HashMap<>();

        for (RereDiffVector node : order) {
            if (node.isLeaf) {
                replacements.put(node, node);
                continue;
            }

            String tag = node.opTag;
            double sp = node.scalarParam;
            int nInputs = node.inputs.size();

            // addScalar(0) → pass-through input
            if ("addScalar".equals(tag) && sp == 0.0 && nInputs == 1) {
                RereDiffVector rep = replacements.getOrDefault(node.inputs.get(0), node.inputs.get(0));
                replacements.put(node, rep);
                continue;
            }
            // subScalar(0) → pass-through input
            if ("subScalar".equals(tag) && sp == 0.0 && nInputs == 1) {
                RereDiffVector rep = replacements.getOrDefault(node.inputs.get(0), node.inputs.get(0));
                replacements.put(node, rep);
                continue;
            }
            // mulScalar(1) → pass-through input
            if ("mulScalar".equals(tag) && sp == 1.0 && nInputs == 1) {
                RereDiffVector rep = replacements.getOrDefault(node.inputs.get(0), node.inputs.get(0));
                replacements.put(node, rep);
                continue;
            }
            // divScalar(1) → pass-through input
            if ("divScalar".equals(tag) && sp == 1.0 && nInputs == 1) {
                RereDiffVector rep = replacements.getOrDefault(node.inputs.get(0), node.inputs.get(0));
                replacements.put(node, rep);
                continue;
            }
            // mulScalar(0) → zeros constant (detach from input)
            if ("mulScalar".equals(tag) && sp == 0.0 && nInputs == 1) {
                RereDiffVector zeros = new RereDiffVector(IDoubleVector.zeros(node.value.size()));
                zeros.opTag = "constant";
                zeros.isLeaf = true;
                replacements.put(node, zeros);
                continue;
            }
            // pow(1) → pass-through input
            if ("pow".equals(tag) && sp == 1.0 && nInputs == 1) {
                RereDiffVector rep = replacements.getOrDefault(node.inputs.get(0), node.inputs.get(0));
                replacements.put(node, rep);
                continue;
            }

            // Rewrite inputs to use replacements (transitive folding)
            boolean changed = false;
            for (int i = 0; i < nInputs; i++) {
                RereDiffVector replaced = replacements.get(node.inputs.get(i));
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

    public static int countNodes(IDiffVector root) {
        List<RereDiffVector> order = new ArrayList<>();
        Set<RereDiffVector> visited = new HashSet<>();
        ((RereDiffVector) root).buildTopo(order, visited);
        return order.size();
    }

    public static int countLeaves(IDiffVector root) {
        List<RereDiffVector> order = new ArrayList<>();
        Set<RereDiffVector> visited = new HashSet<>();
        ((RereDiffVector) root).buildTopo(order, visited);
        int count = 0;
        for (RereDiffVector v : order) {
            if (v.isLeaf) count++;
        }
        return count;
    }

    public static GraphStats stats(IDiffVector root) {
        List<RereDiffVector> order = new ArrayList<>();
        Set<RereDiffVector> visited = new HashSet<>();
        ((RereDiffVector) root).buildTopo(order, visited);
        int leaves = 0, nonLeaf = 0, fusibleChains = 0;
        for (RereDiffVector v : order) {
            if (v.isLeaf) leaves++;
            else nonLeaf++;
        }
        return new GraphStats(order.size(), leaves, nonLeaf, fusibleChains);
    }

    /** Summary statistics of a vector autodiff graph. / 向量计算图统计摘要。 */
    public record GraphStats(int totalNodes, int leafNodes, int nonLeafNodes, int fusibleChains) {
        @Override
        public String toString() {
            return String.format("Graph: %d nodes (%d leaves, %d ops)", totalNodes, leafNodes, nonLeafNodes);
        }
    }
}
