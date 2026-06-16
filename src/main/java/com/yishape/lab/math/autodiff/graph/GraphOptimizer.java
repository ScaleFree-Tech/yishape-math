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
import com.yishape.lab.math.compute.gpu.GpuActivation;
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
        "rsubScalar", "rdivScalar", "reciprocal",
        "neg", "exp", "log", "sqrt", "abs", "sin", "cos", "tan",
        "sigmoid", "tanh", "relu", "gelu", "square", "sign",
        "silu", "mish", "softplus", "elu", "leakyRelu", "selu",
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
     * Uses a replacements-map approach (same pattern as {@link #foldConstants(RereDiffVector)})
     * instead of in-place mutation, so consumer rewiring is handled correctly.
     *
     * <p>Forward scan allows newly-folded constants to propagate to consumers in the same pass:
     * a node whose inputs were just folded earlier in this pass will resolve through the
     * replacements map and be folded too.</p>
     */
    static void foldConstantsInOrder(List<RereDiffTensor> order) {
        Map<RereDiffTensor, RereDiffTensor> replacements = new HashMap<>();

        for (int i = 0; i < order.size(); i++) {
            RereDiffTensor node = order.get(i);
            if (node.isLeaf() || node.opTag() == null || node.inputs() == null || node.inputs().isEmpty())
                continue;
            if (!FOLDABLE_OPS.contains(node.opTag())) continue;

            // Check: all inputs must be constant leaves. Only leaves with
            // opTag="constant" (AD.constant() or previously-folded) are safe.
            // Leaves with null opTag COULD be trainable variables — conservatively
            // skip folding since requiresGrad defaults to true for both variables
            // and AD.constant() (the latter needs requiresGrad for tape-of-tape).
            boolean allConst = true;
            for (RereDiffTensor inp : node.inputs()) {
                RereDiffTensor resolved = replacements.getOrDefault(inp, inp);
                if (!resolved.isLeaf()) {
                    allConst = false;
                    break;
                }
                String tag = resolved.opTag();
                if (!"constant".equals(tag)) {
                    allConst = false;
                    break;
                }
            }
            if (!allConst) continue;

            double[] result = evaluateConstantOp(node, replacements);
            if (result == null) continue;

            // Create new constant node — scalarParam/scalarParam2 default to NaN
            RereDiffTensor constant = new RereDiffTensor(result, node.shape());
            constant.setOpTag("constant");
            constant.setIsLeaf(true);
            constant.setRequiresGrad(false);
            replacements.put(node, constant);
        }

        // Rewire consumers and remove folded nodes
        if (!replacements.isEmpty()) {
            for (RereDiffTensor node : order) {
                List<RereDiffTensor> inList = node.inputs();
                if (inList != null && !inList.isEmpty()) {
                    boolean changed = false;
                    for (int j = 0; j < inList.size(); j++) {
                        RereDiffTensor rep = replacements.get(inList.get(j));
                        if (rep != null) {
                            if (!changed) {
                                inList = new ArrayList<>(inList);
                                changed = true;
                            }
                            inList.set(j, rep);
                        }
                    }
                    if (changed) node.setInputs(inList);
                }
            }
            order.removeIf(replacements::containsKey);
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

    /**
     * Evaluate a constant graph node by resolving all inputs through the replacements map.
     * Uses {@link RereDoubleTensor#toDoubleArray()} (not {@code getStorageData()}) to
     * correctly handle non-contiguous tensor views (slice, permute, narrow, etc.).
     */
    private static double[] evaluateConstantOp(RereDiffTensor node,
            Map<RereDiffTensor, RereDiffTensor> replacements) {
        String tag = node.opTag();
        if (node.inputs().size() == 1) {
            RereDiffTensor inp = replacements.getOrDefault(node.inputs().get(0), node.inputs().get(0));
            double[] a = inp.value().toDoubleArray();
            return evalUnary(tag, a, a.length, node.scalarParam());
        }
        if (node.inputs().size() == 2) {
            RereDiffTensor inp0 = replacements.getOrDefault(node.inputs().get(0), node.inputs().get(0));
            RereDiffTensor inp1 = replacements.getOrDefault(node.inputs().get(1), node.inputs().get(1));
            double[] a = inp0.value().toDoubleArray();
            double[] b = inp1.value().toDoubleArray();
            if (a.length != b.length) return null;
            return evalBinary(tag, a, b, a.length);
        }
        return null;
    }

    /** Evaluate a unary/scalar constant op via the GPU→HPC→SIMD→SISD fallback chain. */
    private static double[] evalUnary(String tag, double[] a, int n, double p) {
        switch (tag) {
            // Scalar broadcast ops
            case "addScalar":   return COMPUTER.binaryOperate(a, p, BinaryOperation.ADD);
            case "subScalar":   return COMPUTER.binaryOperate(a, p, BinaryOperation.SUBTRACT);
            case "mulScalar":   return COMPUTER.binaryOperate(a, p, BinaryOperation.MULTIPLY);
            case "divScalar":   return COMPUTER.binaryOperate(a, p, BinaryOperation.DIVIDE);
            // rsubScalar: p - a[i] = -a[i] + p
            case "rsubScalar": {
                double[] neg = COMPUTER.negate(a);
                return COMPUTER.binaryOperate(neg, p, BinaryOperation.ADD);
            }
            // rdivScalar: p / a[i] = p * (1/a[i])
            case "rdivScalar": {
                double[] recip = COMPUTER.binaryOperate(
                    COMPUTER.fill(n, 1.0), a, BinaryOperation.DIVIDE);
                return COMPUTER.binaryOperate(recip, p, BinaryOperation.MULTIPLY);
            }
            // reciprocal: 1.0 / a[i]
            case "reciprocal":
                return COMPUTER.binaryOperate(
                    COMPUTER.fill(n, 1.0), a, BinaryOperation.DIVIDE);
            // Unary ops
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
            case "sign":    return COMPUTER.sign(a);
            case "silu": {
                // GPU→SISD: GpuActivation dispatches to WGSL shader when GPU is available
                double[] r = GpuActivation.trySilu(a);
                if (r != null) return r;
                // SISD fallback: constant folding is a one-time optimization pass
                double[] out = new double[n];
                for (int i = 0; i < n; i++) out[i] = a[i] / (1.0 + Math.exp(-a[i]));
                return out;
            }
            case "mish": {
                // SISD only: mish has no WGSL shader (GpuActivation op index 4 is skipped)
                double[] out = new double[n];
                for (int i = 0; i < n; i++) {
                    double sp = Math.log(1.0 + Math.exp(a[i]));
                    out[i] = a[i] * Math.tanh(sp);
                }
                return out;
            }
            case "softplus": {
                // beta from scalarParam; GPU shader uses default beta=1.0
                double beta = Double.isNaN(p) ? 1.0 : p;
                if (beta == 1.0) {
                    double[] r = GpuActivation.trySoftplus(a);
                    if (r != null) return r;
                }
                // SISD fallback (or non-default beta)
                double[] out = new double[n];
                for (int i = 0; i < n; i++) {
                    double bx = beta * a[i];
                    out[i] = bx > 20 ? a[i] : Math.log(1.0 + Math.exp(bx)) / beta;
                }
                return out;
            }
            case "elu": {
                // alpha from scalarParam; GPU shader uses default alpha=1.0
                double alpha = Double.isNaN(p) ? 1.0 : p;
                if (alpha == 1.0) {
                    double[] r = GpuActivation.tryElu(a);
                    if (r != null) return r;
                }
                // SISD fallback (or non-default alpha)
                double[] out = new double[n];
                for (int i = 0; i < n; i++)
                    out[i] = a[i] >= 0 ? a[i] : alpha * (Math.exp(a[i]) - 1);
                return out;
            }
            case "leakyRelu": {
                // alpha from scalarParam; GPU shader uses default alpha=0.01
                double alpha = Double.isNaN(p) ? 0.01 : p;
                if (alpha == 0.01) {
                    double[] r = GpuActivation.tryLeakyRelu(a);
                    if (r != null) return r;
                }
                // SISD fallback (or non-default alpha)
                double[] out = new double[n];
                for (int i = 0; i < n; i++)
                    out[i] = a[i] >= 0 ? a[i] : alpha * a[i];
                return out;
            }
            case "selu": {
                // GPU→SISD: selu has fixed alpha=1.673..., scale=1.050... (WGSL shader matches)
                double[] r = GpuActivation.trySelu(a);
                if (r != null) return r;
                // SISD fallback
                double alpha = 1.6732632423543772, scale = 1.0507009873554804;
                double[] out = new double[n];
                for (int i = 0; i < n; i++)
                    out[i] = scale * (a[i] >= 0 ? a[i] : alpha * (Math.exp(a[i]) - 1));
                return out;
            }
            case "pow":     if (Double.isNaN(p)) return null;
                            return COMPUTER.universalOperate(a, UniversalOperation.POW, p);
            case "sum":     // Only fold flat sum (scalarParam NaN). Axis-specific sum has non-NaN stride → skip.
                            if (!Double.isNaN(p)) return null;
                            return new double[]{COMPUTER.reduceOperate(a, ReduceOperation.SUM)};
            case "mean":    if (!Double.isNaN(p)) return null;
                            return new double[]{COMPUTER.reduceOperate(a, ReduceOperation.MEAN)};
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

        // Phase 0: Constant subgraph folding — evaluate all-constant ops eagerly.
        // This runs before identity elimination to simplify the graph first
        // (e.g., pow(sum(x), 2) where x is constant becomes a single constant node).
        foldConstantsInOrder(order);

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
