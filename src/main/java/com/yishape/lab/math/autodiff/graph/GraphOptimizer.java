package com.yishape.lab.math.autodiff.graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;

/**
 * Computation-graph inspection and rewrite utilities.
 * Operates on the tensor graph via {@link RereDiffVector#tensor}.
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
        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.tensor.buildTopo(order, visited);

        Map<RereDiffTensor, RereDiffTensor> replacements = new HashMap<>();

        for (RereDiffTensor node : order) {
            if (node.isLeaf) {
                replacements.put(node, node);
                continue;
            }

            String tag = node.opTag;
            double sp = node.scalarParam;
            int nInputs = node.inputs.size();

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
                RereDiffTensor zeros = new RereDiffTensor(new double[(int) node.value.totalSize()], node.shape());
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

        RereDiffTensor newRoot = replacements.getOrDefault(root.tensor, root.tensor);
        return new RereDiffVector(newRoot);
    }

    public static int countNodes(IDiffVector root) {
        if (!(root instanceof RereDiffVector r)) return 1;
        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        r.tensor.buildTopo(order, visited);
        return order.size();
    }

    public static int countLeaves(IDiffVector root) {
        if (!(root instanceof RereDiffVector r)) return 1;
        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        r.tensor.buildTopo(order, visited);
        int count = 0;
        for (RereDiffTensor t : order) {
            if (t.isLeaf) count++;
        }
        return count;
    }

    public static GraphStats stats(IDiffVector root) {
        if (!(root instanceof RereDiffVector r)) {
            return new GraphStats(1, 1, 0, 0);
        }
        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        r.tensor.buildTopo(order, visited);
        int leaves = 0, nonLeaf = 0, fusibleChains = 0;
        for (RereDiffTensor t : order) {
            if (t.isLeaf) leaves++;
            else nonLeaf++;
        }
        return new GraphStats(order.size(), leaves, nonLeaf, fusibleChains);
    }

    /** Summary statistics of an autodiff graph. */
    public record GraphStats(int totalNodes, int leafNodes, int nonLeafNodes, int fusibleChains) {
        @Override
        public String toString() {
            return String.format("Graph: %d nodes (%d leaves, %d ops)", totalNodes, leafNodes, nonLeafNodes);
        }
    }
}
