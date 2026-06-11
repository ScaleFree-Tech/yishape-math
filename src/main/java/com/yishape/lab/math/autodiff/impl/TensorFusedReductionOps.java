package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Fused element-wise + reduction operator chain for tensors.
 *
 * <p>Chains multiple unary/scalar ops on a single flat buffer, followed by an N-D
 * reduction terminator (softmax, sum, mean over a specific dimension or all elements).
 * Creates a single fused {@link RereDiffTensor} node whose backward applies the
 * adjoint chain in reverse order (reduction backward → element-wise backward chain).
 *
 * <p>Built via {@code new TensorFusedReductionOps(tensor).relu().exp().sum(dim)}.
 * Mirrors {@link FusedReductionOps} but operates on {@link RereDiffTensor} with
 * N-D shape awareness.
 *
 * <p>Integration point: use {@link com.yishape.lab.math.autodiff.AD#fuseReduceTensor(IDiffTensor)}
 * to obtain an instance.
 */
public final class TensorFusedReductionOps {

    private static final DoubleVectorComputer COMPUTER = new DoubleVectorComputer();

    private final RereDiffTensor root;
    private final int totalSize;
    private final int[] originalShape;
    private final List<double[]> opInputs = new ArrayList<>();
    private final List<DoubleBinaryOperator> opBackwards = new ArrayList<>();
    private final List<String> opTags = new ArrayList<>();
    private double[] cur;

    public TensorFusedReductionOps(RereDiffTensor input) {
        this.root = input;
        this.totalSize = (int) input.value().totalSize();
        this.originalShape = input.shape();
        this.cur = input.value().toDoubleArray();
    }

    // ── Element-wise builder methods ──

    /** Register a unary op with a universal operation (SIMD-accelerated forward). */
    private TensorFusedReductionOps addOp(UniversalOperation uop,
                                           DoubleBinaryOperator backward, String tag) {
        double[] input = cur;
        double[] output = COMPUTER.universalOperate(input, uop, 0.0);
        opInputs.add(input);
        opBackwards.add(backward);
        opTags.add(tag);
        cur = output;
        return this;
    }

    /** Register a unary op with a custom lambda forward (no SIMD). */
    private TensorFusedReductionOps addOp(DoubleUnaryOperator forward,
                                           DoubleBinaryOperator backward, String tag) {
        double[] input = cur;
        double[] output = new double[totalSize];
        for (int i = 0; i < totalSize; i++) output[i] = forward.applyAsDouble(input[i]);
        opInputs.add(input);
        opBackwards.add(backward);
        opTags.add(tag);
        cur = output;
        return this;
    }

    // ---- Standard element-wise ops (SIMD-accelerated where universal op exists) ----

    public TensorFusedReductionOps exp()    { return addOp(UniversalOperation.EXP,  (g, x) -> g * Math.exp(x), "exp"); }
    public TensorFusedReductionOps log()    { return addOp(UniversalOperation.LOG,  (g, x) -> g / x, "log"); }
    public TensorFusedReductionOps sqrt()   { return addOp(UniversalOperation.SQRT, (g, x) -> g / (2.0 * Math.sqrt(x)), "sqrt"); }
    public TensorFusedReductionOps relu()   { return addOp(UniversalOperation.RELU, (g, x) -> x > 0 ? g : 0, "relu"); }
    public TensorFusedReductionOps sigmoid(){ return addOp(UniversalOperation.SIGMOID, (g, x) -> { double s = 1.0/(1.0+Math.exp(-x)); return g * s * (1-s); }, "sigmoid"); }
    public TensorFusedReductionOps tanh()   { return addOp(UniversalOperation.TANH, (g, x) -> { double t = Math.tanh(x); return g * (1.0 - t*t); }, "tanh"); }
    public TensorFusedReductionOps abs()    { return addOp(UniversalOperation.ABS,  (g, x) -> x >= 0 ? g : -g, "abs"); }
    public TensorFusedReductionOps sin()    { return addOp(UniversalOperation.SIN,  (g, x) -> g * Math.cos(x), "sin"); }
    public TensorFusedReductionOps cos()    { return addOp(UniversalOperation.COS,  (g, x) -> -g * Math.sin(x), "cos"); }
    public TensorFusedReductionOps gelu()   { return addOp(UniversalOperation.GELU, (g, x) -> {
            double c = Math.sqrt(2.0 / Math.PI);
            double t = Math.tanh(c * (x + 0.044715 * x * x * x));
            double dt = 1.0 - t * t;
            double dx = c * (1.0 + 3.0 * 0.044715 * x * x);
            return g * (0.5 * (1.0 + t) + 0.5 * x * dt * dx);
        }, "gelu"); }

    // ---- Scalar-fallback ops (no UniversalOperation) ----

    public TensorFusedReductionOps neg()    { return addOp(x -> -x,    (g, x) -> -g, "neg"); }
    public TensorFusedReductionOps square() { return addOp(x -> x*x, (g, x) -> g * 2.0 * x, "square"); }
    public TensorFusedReductionOps silu()   { return addOp(x -> x / (1.0 + Math.exp(-x)), (g, x) -> {
            double s = 1.0/(1.0+Math.exp(-x));
            return g * (s + x * s * (1.0 - s));
        }, "silu"); }
    public TensorFusedReductionOps mish()   { return addOp(x -> x * Math.tanh(Math.log1p(Math.exp(x))), (g, x) -> {
            double sp = Math.log1p(Math.exp(x));
            double t = Math.tanh(sp);
            double sig = 1.0/(1.0+Math.exp(-x));
            return g * (t + x * (1.0 - t*t) * sig);
        }, "mish"); }

    public TensorFusedReductionOps pow(double n)   {
        return addOp(x -> Math.pow(x, n),
                     (g, x) -> g * n * Math.pow(x, n - 1), "pow");
    }

    public TensorFusedReductionOps add(double s) {
        return addOp(x -> x + s, (g, x) -> g, "addScalar");
    }

    public TensorFusedReductionOps mul(double s) {
        return addOp(x -> x * s, (g, x) -> g * s, "mulScalar");
    }

    // ---- Extended unary ops (no UniversalOperation) ----

    public TensorFusedReductionOps leakyRelu(double alpha) {
        return addOp(x -> x > 0 ? x : alpha * x,
                     (g, x) -> x > 0 ? g : g * alpha, "leakyRelu");
    }

    public TensorFusedReductionOps elu(double alpha) {
        return addOp(x -> x > 0 ? x : alpha * (Math.exp(x) - 1),
                     (g, x) -> x > 0 ? g : g * alpha * Math.exp(x), "elu");
    }

    public TensorFusedReductionOps selu() {
        final double lambda = 1.0507009873554804934193349852946;
        final double alpha = 1.6732632423543772848170429916717;
        return addOp(x -> x > 0 ? lambda * x : lambda * alpha * (Math.exp(x) - 1),
                     (g, x) -> x > 0 ? g * lambda : g * lambda * alpha * Math.exp(x), "selu");
    }

    public TensorFusedReductionOps softplus() {
        return addOp(x -> x > 20 ? x : Math.log1p(Math.exp(x)),
                     (g, x) -> x > 20 ? g : g / (1.0 + Math.exp(-x)), "softplus");
    }

    public TensorFusedReductionOps hardtanh(double min, double max) {
        return addOp(x -> x < min ? min : (x > max ? max : x),
                     (g, x) -> (x < min || x > max) ? 0 : g, "hardtanh");
    }

    // ---- Scalar binary ops ----

    public TensorFusedReductionOps sub(double s) {
        return addOp(x -> x - s, (g, x) -> g, "subScalar");
    }

    public TensorFusedReductionOps div(double s) {
        return addOp(x -> x / s, (g, x) -> g / s, "divScalar");
    }

    // ── Reduction terminators ──

    /**
     * Reduce-sum over a specific dimension.
     * @param dim      dimension to reduce
     * @param keepdim  if true, keeps the reduced dimension as size 1
     */
    public IDiffTensor sum(int dim, boolean keepdim) {
        int[] outShape = reducedShape(originalShape, dim, keepdim);
        double[] result = reduceSum(cur, originalShape, dim);
        String chainTag = opTags.isEmpty() ? "sum" : String.join("_", opTags) + "_sum";
        return buildFusedNode(result, outShape, chainTag, "sum", dim, keepdim);
    }

    /** Reduce-sum over all elements (SIMD-accelerated). */
    public IDiffTensor sum() {
        double s = COMPUTER.reduceOperate(cur, ReduceOperation.SUM);
        String chainTag = opTags.isEmpty() ? "sum" : String.join("_", opTags) + "_sum";
        return buildFusedNode(new double[]{s}, new int[]{1}, chainTag, "sumAll", -1, false);
    }

    /**
     * Reduce-mean over a specific dimension.
     * @param dim      dimension to reduce
     * @param keepdim  if true, keeps the reduced dimension as size 1
     */
    public IDiffTensor mean(int dim, boolean keepdim) {
        int[] outShape = reducedShape(originalShape, dim, keepdim);
        double scale = 1.0 / originalShape[dim];
        double[] summed = reduceSum(cur, originalShape, dim);
        double[] result = COMPUTER.binaryOperate(summed, scale, com.yishape.lab.math.compute.ops.BinaryOperation.MULTIPLY);
        String chainTag = opTags.isEmpty() ? "mean" : String.join("_", opTags) + "_mean";
        return buildFusedNode(result, outShape, chainTag, "mean", dim, keepdim);
    }

    /** Reduce-mean over all elements (SIMD-accelerated). */
    public IDiffTensor mean() {
        double s = COMPUTER.reduceOperate(cur, ReduceOperation.SUM) / totalSize;
        String chainTag = opTags.isEmpty() ? "mean" : String.join("_", opTags) + "_mean";
        return buildFusedNode(new double[]{s}, new int[]{1}, chainTag, "meanAll", -1, false);
    }

    /**
     * Softmax along a specific dimension.
     */
    public IDiffTensor softmax(int dim) {
        int[] outShape = originalShape.clone();
        double[] result = softmaxForward(cur, originalShape, dim);
        String chainTag = opTags.isEmpty() ? "softmax" : String.join("_", opTags) + "_softmax";
        return buildFusedNode(result, outShape, chainTag, "softmax", dim, false);
    }

    /**
     * Log-softmax along a specific dimension.
     * Numerically stable: x - logSumExp(x) = log(softmax(x)).
     */
    public IDiffTensor logSoftmax(int dim) {
        int[] outShape = originalShape.clone();
        double[] result = logSoftmaxForward(cur, originalShape, dim);
        String chainTag = opTags.isEmpty() ? "logSoftmax" : String.join("_", opTags) + "_logSoftmax";
        return buildFusedNode(result, outShape, chainTag, "logSoftmax", dim, false);
    }

    // ── Internal: reduction forward ──

    static double[] reduceSum(double[] data, int[] shape, int dim) {
        int d = dim;
        int rank = shape.length;
        int outer = 1, reduce = shape[d], inner = 1;
        for (int i = 0; i < d; i++) outer *= shape[i];
        for (int i = d + 1; i < rank; i++) inner *= shape[i];

        double[] result = new double[outer * inner];
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double s = 0;
                for (int r = 0; r < reduce; r++) {
                    s += data[(o * reduce + r) * inner + i];
                }
                result[o * inner + i] = s;
            }
        }
        return result;
    }

    static int[] reducedShape(int[] shape, int dim, boolean keepdim) {
        int d = dim;
        if (keepdim) {
            int[] s = shape.clone();
            s[d] = 1;
            return s;
        }
        int[] s = new int[shape.length - 1];
        int pos = 0;
        for (int i = 0; i < shape.length; i++) {
            if (i != d) s[pos++] = shape[i];
        }
        return s;
    }

    static double[] softmaxForward(double[] data, int[] shape, int dim) {
        int d = dim;
        int rank = shape.length;
        int outer = 1, reduce = shape[d], inner = 1;
        for (int i = 0; i < d; i++) outer *= shape[i];
        for (int i = d + 1; i < rank; i++) inner *= shape[i];

        double[] result = new double[data.length];
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double maxVal = Double.NEGATIVE_INFINITY;
                for (int r = 0; r < reduce; r++) {
                    double v = data[(o * reduce + r) * inner + i];
                    if (v > maxVal) maxVal = v;
                }
                double sumExp = 0;
                for (int r = 0; r < reduce; r++) {
                    int idx = (o * reduce + r) * inner + i;
                    result[idx] = Math.exp(data[idx] - maxVal);
                    sumExp += result[idx];
                }
                for (int r = 0; r < reduce; r++) {
                    result[(o * reduce + r) * inner + i] /= sumExp;
                }
            }
        }
        return result;
    }

    static double[] logSoftmaxForward(double[] data, int[] shape, int dim) {
        int d = dim;
        int rank = shape.length;
        int outer = 1, reduce = shape[d], inner = 1;
        for (int i = 0; i < d; i++) outer *= shape[i];
        for (int i = d + 1; i < rank; i++) inner *= shape[i];

        double[] result = new double[data.length];
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                // Find max for numerical stability
                double maxVal = Double.NEGATIVE_INFINITY;
                for (int r = 0; r < reduce; r++) {
                    double v = data[(o * reduce + r) * inner + i];
                    if (v > maxVal) maxVal = v;
                }
                // Compute logSumExp = max + log(sum(exp(x - max)))
                double sumExp = 0;
                for (int r = 0; r < reduce; r++) {
                    sumExp += Math.exp(data[(o * reduce + r) * inner + i] - maxVal);
                }
                double lse = maxVal + Math.log(sumExp);
                // logSoftmax = x - logSumExp
                for (int r = 0; r < reduce; r++) {
                    result[(o * reduce + r) * inner + i] = data[(o * reduce + r) * inner + i] - lse;
                }
            }
        }
        return result;
    }

    // ── Internal: fused node construction ──

    private IDiffTensor buildFusedNode(double[] result, int[] outShape,
                                        String chainTag, String reduceTag,
                                        int reduceDim, boolean keepdim) {
        if (!root.requiresGrad()) {
            return new ConstantDiffTensor(new RereDoubleTensor(result, outShape));
        }

        int n = totalSize;
        double[] inData = cur; // element-wise output = input to reduction
        double[] softmaxOut = "softmax".equals(reduceTag) ? result : null;
        double[] logSoftmaxOut = "logSoftmax".equals(reduceTag) ? result : null;
        int[] origShape = originalShape;

        // Capture for backward
        List<double[]> savedInputs = new ArrayList<>(opInputs);
        List<DoubleBinaryOperator> savedBackwards = new ArrayList<>(opBackwards);

        Consumer<RereDiffTensor> bw = self -> {
            double[] gradBuf = AutodiffBufferPool.acquire(n);

            // Step 1: Reduction backward → broadcast grad to full shape
            switch (reduceTag) {
                case "sum" -> reductionBackwardSum(self.gradData(), gradBuf, origShape, reduceDim);
                case "sumAll" -> {
                    double g = self.gradData()[0];
                    Arrays.fill(gradBuf, 0, n, g);
                }
                case "mean" -> reductionBackwardMean(self.gradData(), gradBuf, origShape, reduceDim);
                case "meanAll" -> {
                    double g = self.gradData()[0] / n;
                    Arrays.fill(gradBuf, 0, n, g);
                }
                case "softmax" -> reductionBackwardSoftmax(self.gradData(), softmaxOut, gradBuf, origShape, reduceDim);
                case "logSoftmax" -> reductionBackwardLogSoftmax(self.gradData(), logSoftmaxOut, gradBuf, origShape, reduceDim);
            }

            // Step 2: Element-wise backward chain (reverse order)
            for (int k = savedBackwards.size() - 1; k >= 0; k--) {
                double[] in = savedInputs.get(k);
                DoubleBinaryOperator bwd = savedBackwards.get(k);
                for (int i = 0; i < n; i++) {
                    gradBuf[i] = bwd.applyAsDouble(gradBuf[i], in[i]);
                }
            }

            root.accGradFromPooled(gradBuf, n);
        };

        RereDiffTensor node = new RereDiffTensor(result, outShape, List.of(root), bw, chainTag);
        node.setExportShape(outShape);
        if (reduceDim >= 0) {
            node.setScalarParam(reduceDim);
            node.setScalarParam2(keepdim ? 1.0 : 0.0);
        }
        return node;
    }

    // ── Reduction backward helpers ──

    static void reductionBackwardSum(double[] grad, double[] buf, int[] shape, int dim) {
        int d = dim;
        int rank = shape.length;
        int outer = 1, reduce = shape[d], inner = 1;
        for (int i = 0; i < d; i++) outer *= shape[i];
        for (int i = d + 1; i < rank; i++) inner *= shape[i];

        for (int o = 0; o < outer; o++) {
            for (int r = 0; r < reduce; r++) {
                for (int i = 0; i < inner; i++) {
                    buf[(o * reduce + r) * inner + i] = grad[o * inner + i];
                }
            }
        }
    }

    static void reductionBackwardMean(double[] grad, double[] buf, int[] shape, int dim) {
        int d = dim;
        int rank = shape.length;
        int outer = 1, reduce = shape[d], inner = 1;
        for (int i = 0; i < d; i++) outer *= shape[i];
        for (int i = d + 1; i < rank; i++) inner *= shape[i];

        double scale = 1.0 / reduce;
        for (int o = 0; o < outer; o++) {
            for (int r = 0; r < reduce; r++) {
                for (int i = 0; i < inner; i++) {
                    buf[(o * reduce + r) * inner + i] = grad[o * inner + i] * scale;
                }
            }
        }
    }

    static void reductionBackwardSoftmax(double[] grad, double[] softmaxOut,
                                          double[] buf, int[] shape, int dim) {
        int d = dim;
        int rank = shape.length;
        int outer = 1, reduce = shape[d], inner = 1;
        for (int i = 0; i < d; i++) outer *= shape[i];
        for (int i = d + 1; i < rank; i++) inner *= shape[i];

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double dot = 0;
                for (int r = 0; r < reduce; r++) {
                    int idx = (o * reduce + r) * inner + i;
                    dot += grad[idx] * softmaxOut[idx];
                }
                for (int r = 0; r < reduce; r++) {
                    int idx = (o * reduce + r) * inner + i;
                    buf[idx] = softmaxOut[idx] * (grad[idx] - dot);
                }
            }
        }
    }

    /** Backward for logSoftmax: grad_input_i = grad_i - softmax_i * sum_j(grad_j). */
    static void reductionBackwardLogSoftmax(double[] grad, double[] logSoftmaxOut,
                                           double[] buf, int[] shape, int dim) {
        int d = dim;
        int rank = shape.length;
        int outer = 1, reduce = shape[d], inner = 1;
        for (int i = 0; i < d; i++) outer *= shape[i];
        for (int i = d + 1; i < rank; i++) inner *= shape[i];

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                // Compute softmax from logSoftmax: softmax[i] = exp(logSoftmax[i])
                double sumGrad = 0;
                for (int r = 0; r < reduce; r++) {
                    int idx = (o * reduce + r) * inner + i;
                    sumGrad += grad[idx];
                }
                for (int r = 0; r < reduce; r++) {
                    int idx = (o * reduce + r) * inner + i;
                    double sm = Math.exp(logSoftmaxOut[idx]); // softmax = exp(logSoftmax)
                    buf[idx] = grad[idx] - sm * sumGrad;
                }
            }
        }
    }

    // ── Accessors ──

    /** Returns the current intermediate data (after all element-wise ops). */
    public double[] intermediateData() { return cur; }
}
