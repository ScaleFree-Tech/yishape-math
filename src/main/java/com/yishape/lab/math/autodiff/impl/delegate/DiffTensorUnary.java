package com.yishape.lab.math.autodiff.impl.delegate;

import com.yishape.lab.math.autodiff.impl.DiffTensorUtil;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.impl.AutodiffBufferPool;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;

/**
 * Element-wise unary operations extracted from {@link RereDiffTensor}.
 * All methods are static, taking the tensor as first parameter.
 */
public final class DiffTensorUnary {
    private DiffTensorUnary() { /* utility class */ }

    // ==================== Simple delegations ====================

    public static IDiffTensor neg(RereDiffTensor tensor) { return unaryOp(tensor, x -> -x, (g, x) -> -g, "neg"); }
    public static IDiffTensor abs(RereDiffTensor tensor) { return unaryOp(tensor, Math::abs, (g, x) -> x >= 0 ? g : -g, "abs"); }
    private static final double SQRT_EPS = 1e-15;
    public static IDiffTensor sqrt(RereDiffTensor tensor) {
        return unaryOp(tensor, Math::sqrt,
            (g, x) -> g / (2.0 * Math.sqrt(Math.max(x, SQRT_EPS))),
            "sqrt");
    }
    public static IDiffTensor exp(RereDiffTensor tensor) { return unaryOpSelf(tensor, Math::exp, (g, y) -> g * y, "exp"); }
    private static final double LOG_EPS = 1e-15;
    public static IDiffTensor log(RereDiffTensor tensor) {
        return unaryOp(tensor,
            x -> Math.log(Math.max(x, LOG_EPS)),
            (g, x) -> g / Math.max(Math.abs(x), LOG_EPS),
            "log");
    }
    public static IDiffTensor sin(RereDiffTensor tensor) { return unaryOp(tensor, Math::sin, (g, x) -> g * Math.cos(x), "sin"); }
    public static IDiffTensor cos(RereDiffTensor tensor) { return unaryOp(tensor, Math::cos, (g, x) -> -g * Math.sin(x), "cos"); }
    public static IDiffTensor tan(RereDiffTensor tensor) { return unaryOp(tensor, Math::tan, (g, x) -> { double c = Math.cos(x); return g / (c * c); }, "tan"); }
    public static IDiffTensor sigmoid(RereDiffTensor tensor) { return unaryOpSelf(tensor, x -> 1.0 / (1.0 + Math.exp(-x)), (g, y) -> g * y * (1.0 - y), "sigmoid"); }
    public static IDiffTensor relu(RereDiffTensor tensor) { return unaryOp(tensor, x -> x > 0 ? x : 0, (g, x) -> x > 0 ? g : 0, "relu"); }
    public static IDiffTensor square(RereDiffTensor tensor) { return unaryOp(tensor, x -> x * x, (g, x) -> g * 2.0 * x, "square"); }
    public static IDiffTensor tanh(RereDiffTensor tensor) { return unaryOpSelf(tensor, Math::tanh, (g, y) -> g * (1.0 - y * y), "tanh"); }

    public static IDiffTensor silu(RereDiffTensor tensor) {
        return unaryOp(tensor, x -> x / (1.0 + Math.exp(-x)),
            (g, x) -> { double s = 1.0 / (1.0 + Math.exp(-x)); return g * (s + x * s * (1.0 - s)); }, "silu");
    }

    public static IDiffTensor gelu(RereDiffTensor tensor) {
        return unaryOp(tensor, x -> {
            double cdf = 0.5 * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
            return x * cdf;
        }, (g, x) -> {
            double x3 = x * x * x;
            double inner = Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x3);
            double tanhInner = Math.tanh(inner);
            double sech2 = 1.0 - tanhInner * tanhInner;
            double cdf = 0.5 * (1.0 + tanhInner);
            double pdf = 0.5 * Math.sqrt(2.0 / Math.PI) * (1.0 + 0.134145 * x * x) * sech2;
            return g * (cdf + x * pdf);
        }, "gelu");
    }

    public static IDiffTensor softplus(RereDiffTensor tensor, double beta) {
        return unaryOp(tensor, x -> { double bx = beta * x; return bx > 20 ? x : Math.log(1.0 + Math.exp(bx)) / beta; },
            (g, x) -> { double bx = beta * x; return bx > 20 ? g : g / (1.0 + Math.exp(-bx)); }, "softplus", beta);
    }

    public static IDiffTensor mish(RereDiffTensor tensor) {
        return unaryOp(tensor, x -> x * Math.tanh(Math.log(1.0 + Math.exp(x))),
            (g, x) -> {
                double sp = Math.log(1.0 + Math.exp(x));
                double th = Math.tanh(sp);
                double sig = 1.0 / (1.0 + Math.exp(-x));
                return g * (th + x * sig * (1.0 - th * th));
            }, "mish");
    }

    public static IDiffTensor elu(RereDiffTensor tensor, double alpha) {
        return unaryOp(tensor, x -> x >= 0 ? x : alpha * (Math.exp(x) - 1),
            (g, x) -> x >= 0 ? g : g * alpha * Math.exp(x), "elu", alpha);
    }

    public static IDiffTensor leakyRelu(RereDiffTensor tensor, double alpha) {
        return unaryOp(tensor, x -> x >= 0 ? x : alpha * x, (g, x) -> x >= 0 ? g : g * alpha, "leakyRelu", alpha);
    }

    public static IDiffTensor selu(RereDiffTensor tensor) {
        double alpha = 1.6732632423543772, scale = 1.0507009873554804;
        return unaryOp(tensor, x -> scale * (x >= 0 ? x : alpha * (Math.exp(x) - 1)),
            (g, x) -> x >= 0 ? g * scale : g * scale * alpha * Math.exp(x), "selu");
    }

    public static IDiffTensor hardtanh(RereDiffTensor tensor, double minVal, double maxVal) {
        return unaryOp(tensor, x -> Math.min(Math.max(x, minVal), maxVal),
            (g, x) -> (x > minVal && x < maxVal) ? g : 0, "hardtanh", minVal, maxVal);
    }

    // ==================== pow ====================

    public static IDiffTensor pow(RereDiffTensor tensor, double n) {
        if (!tensor.requiresGrad) {
            double[] out = (tensor.value instanceof RereDoubleTensor rdt)
                ? rdt.universalOp(UniversalOperation.POW, n)
                : null;
            if (out == null) {
                double[] data = tensor.value.toDoubleArray();
                out = new double[data.length];
                for (int i = 0; i < data.length; i++) out[i] = Math.pow(data[i], n);
            }
            return tensor.toNonDiff(new RereDoubleTensor(out, tensor.shape()));
        }
        int total = (int) tensor.value.totalSize();
        double[] out = (tensor.value instanceof RereDoubleTensor rdt)
            ? rdt.universalOp(UniversalOperation.POW, n)
            : null;
        if (out == null) {
            out = new double[total];
            for (int i = 0; i < total; i++) out[i] = Math.pow(tensor.value.linearGet(i), n);
        }
        // SISD backward with buffer pooling: AutodiffBufferPool.acquire() reuses
        // thread-local buffers across backward calls, amortizing allocation cost.
        // The per-element formula g * n * x^(n-1) is simple arithmetic; migrating
        // to vc.binaryOperate() would create fresh arrays on each backward call,
        // negating the pool's benefit. Forward already uses UniversalOperation.POW.
        java.util.function.Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int m = (int) input.value.totalSize();
            double[] inGrad = AutodiffBufferPool.acquire(m);
            for (int i = 0; i < total; i++) {
                double x = input.value.linearGet(i);
                inGrad[i] = (n == 0) ? 0 : self.grad[i] * n * Math.pow(x, n - 1);
            }
            input.accGradFromPooled(inGrad, m);
        };
        RereDiffTensor powResult = new RereDiffTensor(out, tensor.shape(), java.util.List.of(tensor), bw, "pow", n);
        RereDiffTensor xRef = tensor;
        double scalarN = n;
        powResult.symbolicBackwardFn = g -> {
            if (scalarN == 0.0) {
                return new IDiffTensor[]{
                    new RereDiffTensor(new double[(int)xRef.value.totalSize()], xRef.shape()).fill_(0)
                };
            } else if (scalarN == 1.0) {
                return new IDiffTensor[]{g};
            } else if (scalarN == 2.0) {
                return new IDiffTensor[]{g.mul(2.0).mul(xRef)};
            } else if (scalarN == 3.0) {
                return new IDiffTensor[]{g.mul(3.0).mul(xRef.pow(2))};
            } else {
                return new IDiffTensor[]{g.mul(scalarN).mul(xRef.pow(scalarN - 1))};
            }
        };
        return powResult;
    }

    // ==================== clamp ====================

    public static IDiffTensor clamp(RereDiffTensor tensor, double min, double max) {
        if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.clamp(min, max));
        int total = (int) tensor.value.totalSize();
        double[] out = new double[total];
        for (int i = 0; i < total; i++) {
            double x = tensor.value.linearGet(i);
            out[i] = x < min ? min : x > max ? max : x;
        }
        // SISD backward with buffer pooling: the conditional (x > min && x < max)
        // cannot be cleanly vectorized via DoubleVectorComputer. Pooling amortizes
        // allocation cost across backward calls. Forward already delegates to
        // tensor.value.clamp() which uses applyUnary().
        java.util.function.Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int m = (int) input.value.totalSize();
            double[] inGrad = AutodiffBufferPool.acquire(m);
            for (int i = 0; i < total; i++) {
                double x = input.value.linearGet(i);
                inGrad[i] = (x > min && x < max) ? self.grad[i] : 0;
            }
            input.accGradFromPooled(inGrad, m);
        };
        return new RereDiffTensor(out, tensor.shape(), java.util.List.of(tensor), bw, "clamp", min, max);
    }

    // ==================== dropout ====================

    public static IDiffTensor dropout(RereDiffTensor tensor, double p) {
        if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.clone());
        int total = (int) tensor.value.totalSize();
        long seed = RereDiffVector.DROPOUT_SEED_COUNTER.incrementAndGet();
        java.util.Random rng = new java.util.Random(seed);
        double[] mask = new double[total];
        double[] out = new double[total];
        double scale = 1.0 / (1.0 - p);
        for (int i = 0; i < total; i++) {
            mask[i] = rng.nextDouble() > p ? scale : 0.0;
            out[i] = tensor.value.linearGet(i) * mask[i];
        }
        // SISD backward with buffer pooling: dropout applies a mask to gradients.
        // The mask is pre-computed (random, per-sample) and cannot be vectorized
        // via DoubleVectorComputer. Pooling amortizes allocation cost.
        // SISD forward: mask[i] = (rng.nextDouble() > p) ? scale : 0.0 (random per
        // element, no parallel equivalent).
        java.util.function.Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int m = (int) input.value.totalSize();
            double[] inGrad = AutodiffBufferPool.acquire(m);
            for (int i = 0; i < m; i++) {
                inGrad[i] = self.grad[i] * mask[i];
            }
            input.accGradFromPooled(inGrad, m);
        };
        return new RereDiffTensor(out, tensor.shape(), java.util.List.of(tensor), bw, "dropout", p,
                Double.longBitsToDouble(seed));
    }

    // ==================== unaryOp helpers ====================

    static IDiffTensor unaryOp(RereDiffTensor tensor, java.util.function.DoubleUnaryOperator forward,
                               java.util.function.DoubleBinaryOperator backward,
                               String tag) {
        if (!tensor.requiresGrad) {
            com.yishape.lab.math.compute.ops.UniversalOperation uop = DiffTensorUtil.tagToUniversalOp(tag);
            if (uop != null && tensor.value instanceof RereDoubleTensor rdt) {
                double[] result = rdt.universalOp(uop, 0.0);
                return tensor.toNonDiff(new RereDoubleTensor(result, tensor.shape()));
            }
            double[] data = tensor.value.toDoubleArray();
            for (int i = 0; i < data.length; i++) data[i] = forward.applyAsDouble(data[i]);
            return tensor.toNonDiff(new RereDoubleTensor(data, tensor.shape()));
        }
        int n = (int) tensor.value.totalSize();
        com.yishape.lab.math.compute.ops.UniversalOperation uop = DiffTensorUtil.tagToUniversalOp(tag);
        double[] out;
        if (uop != null && tensor.value instanceof RereDoubleTensor rdt) {
            out = rdt.universalOp(uop, 0.0);
        } else {
            out = new double[n];
            for (int i = 0; i < n; i++) out[i] = forward.applyAsDouble(tensor.value.linearGet(i));
        }
        // SISD backward with buffer pooling: the backward lambda formulas vary widely
        // per op (neg→-g, abs→±g, sin→g*cos(x), etc.) and are routed through a common
        // DoubleBinaryOperator interface. Forward already uses UniversalOperation where
        // tagToUniversalOp maps the tag (exp, log, sqrt, relu, sigmoid, tanh, abs, sin,
        // cos, gelu, tan). Pooling amortizes allocation cost across diverse backward calls.
        java.util.function.Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            double[] inGrad = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                inGrad[i] = backward.applyAsDouble(self.grad[i], input.value.linearGet(i));
            }
            input.accGradFromPooled(inGrad, n);
        };
        RereDiffTensor result = new RereDiffTensor(out, tensor.shape(), java.util.List.of(tensor), bw, tag);
        RereDiffTensor xRef = tensor;
        double[] xData = tensor.value.toDoubleArray();
        result.symbolicBackwardFn = g -> new IDiffTensor[]{
            g.mul(DiffTensorUtil.symbolicUnaryFactor(tag, xRef, backward, xData, Double.NaN))
        };
        return result;
    }

    static IDiffTensor unaryOpSelf(RereDiffTensor tensor, java.util.function.DoubleUnaryOperator forward,
                                   java.util.function.DoubleBinaryOperator backwardUsingOutput,
                                   String tag) {
        if (!tensor.requiresGrad) {
            com.yishape.lab.math.compute.ops.UniversalOperation uop = DiffTensorUtil.tagToUniversalOp(tag);
            if (uop != null && tensor.value instanceof RereDoubleTensor rdt) {
                double[] result = rdt.universalOp(uop, 0.0);
                return tensor.toNonDiff(new RereDoubleTensor(result, tensor.shape()));
            }
            double[] data = tensor.value.toDoubleArray();
            for (int i = 0; i < data.length; i++) data[i] = forward.applyAsDouble(data[i]);
            return tensor.toNonDiff(new RereDoubleTensor(data, tensor.shape()));
        }
        int n = (int) tensor.value.totalSize();
        com.yishape.lab.math.compute.ops.UniversalOperation uop = DiffTensorUtil.tagToUniversalOp(tag);
        double[] out;
        if (uop != null && tensor.value instanceof RereDoubleTensor rdt) {
            out = rdt.universalOp(uop, 0.0);
        } else {
            out = new double[n];
            for (int i = 0; i < n; i++) out[i] = forward.applyAsDouble(tensor.value.linearGet(i));
        }
        // SISD backward with buffer pooling: same rationale as unaryOp above.
        // unaryOpSelf backends (exp, sigmoid, tanh) use g * y (output-based gradient)
        // where y = forward(x). Forward already uses UniversalOperation. Pooling
        // amortizes allocation cost across backward calls.
        java.util.function.Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            double[] inGrad = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                inGrad[i] = backwardUsingOutput.applyAsDouble(self.grad[i], out[i]);
            }
            input.accGradFromPooled(inGrad, n);
        };
        RereDiffTensor result = new RereDiffTensor(out, tensor.shape(), java.util.List.of(tensor), bw, tag);
        RereDiffTensor xRef = tensor;
        double[] xData = tensor.value.toDoubleArray();
        result.symbolicBackwardFn = g -> new IDiffTensor[]{
            g.mul(DiffTensorUtil.symbolicUnaryFactor(tag, xRef, backwardUsingOutput, xData, Double.NaN))
        };
        return result;
    }

    static IDiffTensor unaryOp(RereDiffTensor tensor, java.util.function.DoubleUnaryOperator forward,
                               java.util.function.DoubleBinaryOperator backward,
                               String tag, double scalarParam) {
        IDiffTensor result = unaryOp(tensor, forward, backward, tag);
        if (result instanceof RereDiffTensor rt) rt.scalarParam = scalarParam;
        return result;
    }

    static IDiffTensor unaryOp(RereDiffTensor tensor, java.util.function.DoubleUnaryOperator forward,
                               java.util.function.DoubleBinaryOperator backward,
                               String tag, double scalarParam, double scalarParam2) {
        IDiffTensor result = unaryOp(tensor, forward, backward, tag);
        if (result instanceof RereDiffTensor rt) { rt.scalarParam = scalarParam; rt.scalarParam2 = scalarParam2; }
        return result;
    }
}
