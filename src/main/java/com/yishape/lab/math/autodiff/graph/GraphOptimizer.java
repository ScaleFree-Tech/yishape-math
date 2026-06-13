package com.yishape.lab.math.autodiff.graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import com.yishape.lab.util.YishapeLogger;

/**
 * Computation-graph inspection and rewrite utilities.
 * Operates on the tensor graph via {@link RereDiffVector#tensor}.
 */
public class GraphOptimizer {

    private GraphOptimizer() {}

    private static final YishapeLogger log = YishapeLogger.getLogger(GraphOptimizer.class);

    /** GPU→SIMD→SISD fallback computer for constant folding evaluation. */
    private static final DoubleVectorComputer COMPUTER = new DoubleVectorComputer();

    private static final Set<String> FOLDABLE_OPS = Set.of(
        "add", "sub", "mul", "div",
        "addScalar", "subScalar", "mulScalar", "divScalar",
        "neg", "exp", "log", "sqrt", "abs", "sin", "cos", "tan",
        "sigmoid", "tanh", "relu", "gelu", "square",
        "pow", "sum", "mean"
    );

    /**
     * Optimizes a topologically sorted tensor graph in-place for HPC/GPU export.
     * Includes constant folding (evaluate all-constant subgraphs eagerly) and
     * common subexpression elimination (deduplicate identical nodes).
     *
     * @param order topologically sorted node list, mutated in-place
     * @return number of nodes eliminated
     */
    public static int optimize(List<RereDiffTensor> order) {
        int before = order.size();
        foldConstantsInOrder(order);
        int afterFold = order.size();
        int cseRemoved = eliminateCommonSubexpressions(order);
        int total = before - order.size();
        if (total > 0 && log.isDebugEnabled()) {
            log.debug("Graph optimization: {}->{} nodes ({} folded, {} CSE)",
                before, order.size(), before - afterFold, cseRemoved);
        }
        return total;
    }

    /**
     * Constant folding for tensor graph: evaluate ops whose inputs are all leaf/constant.
     * Forward scan allows newly-folded constants to propagate to consumers in the same pass.
     */
    private static void foldConstantsInOrder(List<RereDiffTensor> order) {
        for (int i = 0; i < order.size(); i++) {
            RereDiffTensor node = order.get(i);
            if (node.isLeaf() || node.opTag() == null || node.inputs() == null || node.inputs().isEmpty())
                continue;
            if (!FOLDABLE_OPS.contains(node.opTag())) continue;

            boolean allConst = true;
            for (RereDiffTensor inp : node.inputs()) {
                if (!inp.isLeaf() && !"constant".equals(inp.opTag())) { allConst = false; break; }
            }
            if (!allConst) continue;

            double[] result = evaluateConstantOp(node);
            if (result == null) continue;

            node.setValue(new RereDoubleTensor(result, node.shape()));
            node.setIsLeaf(true);
            node.setOpTag("constant");
            node.setInputs(List.of());
            node.setBackwardFn(null);
            node.setRequiresGrad(false);
            node.setGradData(null);
        }
    }

    /**
     * CSE: eliminate duplicate nodes with identical opTag, inputs, and scalar params.
     */
    private static int eliminateCommonSubexpressions(List<RereDiffTensor> order) {
        Map<Long, RereDiffTensor> seen = new HashMap<>();
        int eliminated = 0;
        Iterator<RereDiffTensor> it = order.iterator();
        while (it.hasNext()) {
            RereDiffTensor node = it.next();
            if (node.opTag() == null || "leaf".equals(node.opTag())) continue;
            if (node.inputs() == null || node.inputs().isEmpty()) continue;

            long key = cseKey(node);
            RereDiffTensor existing = seen.putIfAbsent(key, node);
            if (existing != null) {
                for (RereDiffTensor n : order) {
                    if (n.inputs() != null) {
                        List<RereDiffTensor> inList = n.inputs();
                        for (int j = 0; j < inList.size(); j++) {
                            if (inList.get(j) == node) {
                                try {
                                    inList.set(j, existing);
                                } catch (UnsupportedOperationException e) {
                                    // C25: immutable list (e.g. List.of()) — replace with mutable copy
                                    List<RereDiffTensor> mutable = new ArrayList<>(inList);
                                    mutable.set(j, existing);
                                    n.setInputs(mutable);
                                }
                            }
                        }
                    }
                }
                it.remove();
                eliminated++;
            }
        }
        return eliminated;
    }

    private static long cseKey(RereDiffTensor node) {
        long h = node.opTag().hashCode();
        if (node.inputs() != null) {
            for (RereDiffTensor inp : node.inputs()) h = h * 31 + System.identityHashCode(inp);
        }
        h = h * 31 + Double.hashCode(Double.isNaN(node.scalarParam()) ? 0 : node.scalarParam());
        h = h * 31 + Double.hashCode(Double.isNaN(node.scalarParam2()) ? 0 : node.scalarParam2());
        h = h * 31 + java.util.Arrays.hashCode(node.shape()); // C24: shape prevents merging differently-shaped nodes
        return h;
    }

    private static double[] evaluateConstantOp(RereDiffTensor node) {
        String tag = node.opTag();
        if (node.inputs().size() == 1) {
            double[] a = node.inputs().get(0).value().getStorageData();
            return evalUnary(tag, a, a.length, node.scalarParam());
        }
        if (node.inputs().size() == 2) {
            double[] a = node.inputs().get(0).value().getStorageData();
            double[] b = node.inputs().get(1).value().getStorageData();
            if (a.length != b.length) return null;
            return evalBinary(tag, a, b, a.length);
        }
        return null;
    }

    /** Evaluate a unary constant op via the GPU→HPC→SIMD→SISD fallback chain. */
    private static double[] evalUnary(String tag, double[] a, int n, double p) {
        switch (tag) {
            case "neg":     return COMPUTER.negate(a);
            case "exp":     return COMPUTER.universalOperate(a, UniversalOperation.EXP, 0);
            case "log":     return COMPUTER.universalOperate(a, UniversalOperation.LOG, 0);
            case "sqrt":    return COMPUTER.universalOperate(a, UniversalOperation.SQRT, 0);
            case "abs":     return COMPUTER.universalOperate(a, UniversalOperation.ABS, 0);
            case "sin":     return COMPUTER.universalOperate(a, UniversalOperation.SIN, 0);
            case "cos":     return COMPUTER.universalOperate(a, UniversalOperation.COS, 0);
            case "tan":     return COMPUTER.universalOperate(a, UniversalOperation.TAN, 0);
            case "sigmoid": return COMPUTER.universalOperate(a, UniversalOperation.SIGMOID, 0);
            case "tanh":    return COMPUTER.universalOperate(a, UniversalOperation.TANH, 0);
            case "relu":    return COMPUTER.universalOperate(a, UniversalOperation.RELU, 0);
            case "gelu":    return COMPUTER.universalOperate(a, UniversalOperation.GELU, 0);
            case "square":  return COMPUTER.binaryOperate(a, a, BinaryOperation.MULTIPLY);
            case "pow":     if (Double.isNaN(p)) return null;
                            return COMPUTER.universalOperate(a, UniversalOperation.POW, p);
            case "sum":     return new double[]{COMPUTER.reduceOperate(a, ReduceOperation.SUM)};
            case "mean":    return new double[]{COMPUTER.reduceOperate(a, ReduceOperation.MEAN)};
            default:        return null;
        }
    }

    /** Evaluate a binary constant op via the GPU→HPC→SIMD→SISD fallback chain. */
    private static double[] evalBinary(String tag, double[] a, double[] b, int n) {
        switch (tag) {
            case "add": return COMPUTER.binaryOperate(a, b, BinaryOperation.ADD);
            case "sub": return COMPUTER.binaryOperate(a, b, BinaryOperation.SUBTRACT);
            case "mul": return COMPUTER.binaryOperate(a, b, BinaryOperation.MULTIPLY);
            case "div": return COMPUTER.binaryOperate(a, b, BinaryOperation.DIVIDE);
            default:   return null;
        }
    }

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
            if (node.isLeaf()) {
                replacements.put(node, node);
                continue;
            }

            String tag = node.opTag();
            double sp = node.scalarParam();
            List<RereDiffTensor> inputs = node.inputs();
            int nInputs = inputs != null ? inputs.size() : 0;

            // addScalar(0) → pass-through input
            if ("addScalar".equals(tag) && sp == 0.0 && nInputs == 1) {
                RereDiffTensor rep = replacements.getOrDefault(inputs.get(0), inputs.get(0));
                replacements.put(node, rep);
                continue;
            }
            // subScalar(0) → pass-through input
            if ("subScalar".equals(tag) && sp == 0.0 && nInputs == 1) {
                RereDiffTensor rep = replacements.getOrDefault(inputs.get(0), inputs.get(0));
                replacements.put(node, rep);
                continue;
            }
            // mulScalar(1) → pass-through input
            if ("mulScalar".equals(tag) && sp == 1.0 && nInputs == 1) {
                RereDiffTensor rep = replacements.getOrDefault(inputs.get(0), inputs.get(0));
                replacements.put(node, rep);
                continue;
            }
            // divScalar(1) → pass-through input
            if ("divScalar".equals(tag) && sp == 1.0 && nInputs == 1) {
                RereDiffTensor rep = replacements.getOrDefault(inputs.get(0), inputs.get(0));
                replacements.put(node, rep);
                continue;
            }
            // mulScalar(0) → zeros constant (detach from input)
            if ("mulScalar".equals(tag) && sp == 0.0 && nInputs == 1) {
                RereDiffTensor zeros = new RereDiffTensor(new double[(int) node.value().totalSize()], node.shape());
                zeros.setOpTag("constant");
                zeros.setIsLeaf(true);
                replacements.put(node, zeros);
                continue;
            }
            // pow(1) → pass-through input
            if ("pow".equals(tag) && sp == 1.0 && nInputs == 1) {
                RereDiffTensor rep = replacements.getOrDefault(inputs.get(0), inputs.get(0));
                replacements.put(node, rep);
                continue;
            }

            // Rewrite inputs to use replacements (transitive folding)
            boolean changed = false;
            for (int i = 0; i < nInputs; i++) {
                RereDiffTensor replaced = replacements.get(inputs.get(i));
                if (replaced != null && replaced != inputs.get(i)) {
                    if (!changed) {
                        node.setInputs(new ArrayList<>(inputs));
                        changed = true;
                    }
                    node.inputs().set(i, replaced);
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
            if (t.isLeaf()) count++;
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
            if (t.isLeaf()) leaves++;
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
