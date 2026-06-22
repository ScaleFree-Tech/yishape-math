package com.yishape.lab.math.autodiff;

import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.autodiff.impl.TangentDiffVector;

/**
 * Mixed-mode automatic differentiation: forward-over-reverse for Hessian-vector
 * products (HVP) and Jacobian-vector products (JVP).
 *
 * <p>Uses a combination of reverse-mode (tape-of-tape via {@link AD#grad})
 * and forward-mode (TangentDiffVector via {@link AD#tangent}) to compute
 * second-order quantities without materializing the full Hessian.</p>
 */
public final class MixedMode {

    private MixedMode() {}

    /**
     * Hessian-vector product: computes H @ v where H = ∇²f(x).
     *
     * <p>Algorithm (tape-of-tape):</p>
     * <ol>
     *   <li>Compute y = fn(x) in the primal graph</li>
     *   <li>Compute gradVec = AD.grad(y, x)[0] — tape-of-tape gradient</li>
     *   <li>Define g = gradVec · v (dot product) — scalar in the grad graph</li>
     *   <li>g.backward() propagates through the tape-of-tape to x</li>
     *   <li>x.getGradient() yields H @ v</li>
     * </ol>
     *
     * @param fn  scalar-valued function R^n → R, returning a scalar IDiffVector
     * @param x   input leaf variable
     * @param v   direction vector
     * @return H @ v as double[]
     */
    public static double[] hvp(Function<IDiffVector, IDiffVector> fn, IDiffVector x, IDoubleVector v) {
        return hvp(fn, x, v, AD.Backend.CPU);
    }

    /**
     * Hessian-vector product with a preferred native backend for the second-order
     * backward pass (Phase 4.3 Step 4).
     *
     * <p><b>Why only the second backward is native.</b> The first step —
     * {@link AD#grad} — builds the gradient graph by traversing each node's
     * {@code symbolicBackwardFn}. That is a CPU tape-of-tape construction with
     * no native execution, so the primal symbolic tape stays intact. Only the
     * final {@code g.backward()} (a first-order backward through the freshly
     * built gradient graph) is routed to the preferred backend:
     * <ol>
     *   <li>try the preferred native backend (GPU or HPC); on success its leaf
     *       gradients are populated directly,</li>
     *   <li>fall back to CPU {@code g.backward()} if native is unavailable or
     *       the gradient graph is not exportable (returns {@code false}).</li>
     * </ol>
     * Correctness is guaranteed by the CPU fallback; native is purely an
     * optimization. The default {@link #hvp} overload keeps {@code CPU} so
     * existing callers are unaffected.
     *
     * @param prefer  backend to try first for the second-order backward
     */
    public static double[] hvp(Function<IDiffVector, IDiffVector> fn, IDiffVector x,
                               IDoubleVector v, AD.Backend prefer) {
        IDiffVector y = fn.apply(x);
        IDiffVector[] grads = AD.grad(y, x);
        IDiffVector gradVec = grads[0];
        IDiffVector vConst = AD.constant(v);
        IDiffVector g = gradVec.dot(vConst);
        x.zeroGradient();
        backwardPrefer(g, prefer);
        return x.getGradient().getData();
    }

    /**
     * Runs a scalar root's backward through the preferred native backend,
     * falling back to CPU. Mirrors {@link AD#backwardWithNativeFallback} but
     * honours a {@code prefer} order and is safe for the gradient graphs built
     * by {@link AD#grad}: a native backend that cannot execute the graph
     * (unsupported op, or a grad-count/length mismatch on the tape-of-tape
     * gradient graph) returns {@code false} via {@link AD#tryGpuExecute} /
     * {@link AD#tryHpcExecute}, and we retry on the next tier — never a silent
     * wrong result. Correctness is guaranteed by the CPU fallback; native is
     * purely an optimization.
     *
     * <p><b>Observed behaviour for second-order graphs.</b> On the tape-of-tape
     * gradient graphs built by {@link AD#grad}, the GPU executor currently
     * reports a leaf-gradient count mismatch and returns {@code false}
     * (→ HPC → CPU), so the second-order backward in practice runs on HPC or
     * CPU. The result is therefore correct regardless of the {@code prefer}
     * hint. Backend-agnostic correctness (the result matches a numerical HVP
     * no matter which tier executed) is guarded by {@code MixedModeGpuTest}.
     */
    private static void backwardPrefer(IDiffVector g, AD.Backend prefer) {
        // CPU (default): plain CPU backward — preserves the historical hvp
        // contract; existing callers (HvpCorrectnessTest, MixedModeTest) are
        // unaffected and never touch the native path.
        if (prefer == AD.Backend.CPU) { g.backward(); return; }
        // GPU-prefer: GPU → HPC → CPU. HPC-prefer: HPC → CPU. Each native tier
        // returns false (and leaves grads untouched) when it cannot execute,
        // so a wrong native result can never reach the caller.
        if (prefer == AD.Backend.GPU && AD.tryGpuExecute(g)) return;
        if (AD.tryHpcExecute(g)) return;
        g.backward();
    }

    /**
     * Jacobian-vector product: computes J @ v where J = ∂f/∂x.
     * Uses forward-mode AD (TangentDiffVector).
     *
     * @param fn  vector-valued function R^n → R^m
     * @param x   input leaf variable
     * @param v   direction vector (length n)
     * @return J @ v as double[] (length m)
     */
    public static double[] jvp(Function<IDiffVector, IDiffVector> fn, IDiffVector x, IDoubleVector v) {
        IDiffVector tangentV = AD.vector(v);
        IDiffVector tangentX = AD.tangent(x, tangentV);
        IDiffVector result = fn.apply(tangentX);
        if (result instanceof TangentDiffVector tdv) {
            return tdv.getTangent().getData();
        }
        return result.getValue().getData();
    }

    /**
     * Full Hessian matrix computed via n HVP calls.
     * Only practical for small n (e.g., n &lt; 100).
     */
    public static IDoubleMatrix hessian(Function<IDiffVector, IDiffVector> fn, IDiffVector x) {
        return hessian(fn, x, AD.Backend.CPU);
    }

    /**
     * Full Hessian with a preferred native backend for each column's HVP.
     *
     * @see #hvp(Function, IDiffVector, IDoubleVector, AD.Backend)
     */
    public static IDoubleMatrix hessian(Function<IDiffVector, IDiffVector> fn, IDiffVector x,
                                        AD.Backend prefer) {
        int n = x.getValue().length();
        double[][] H = new double[n][n];
        for (int i = 0; i < n; i++) {
            double[] ei = new double[n];
            ei[i] = 1.0;
            double[] col = hvp(fn, x, IDoubleVector.of(ei), prefer);
            for (int j = 0; j < n; j++) {
                H[j][i] = col[j];
            }
        }
        return IDoubleMatrix.of(H);
    }

    /**
     * Vector-Jacobian product: computes J<sup>T</sup> @ g where J = ∂f/∂x.
     * Uses reverse-mode AD (VJP transform).
     *
     * <p>向量-雅可比积：计算 J<sup>T</sup> @ g，其中 J = ∂f/∂x。
     * 使用反向模式 AD（VJP 变换）。
     *
     * @param fn                vector-valued function R^n → R^m
     * @param x                 input leaf variable
     * @param upstreamGradient  upstream gradient vector (length m)
     * @return J<sup>T</sup> @ g as double[] (length n)
     */
    public static double[] vjp(Function<IDiffVector, IDiffVector> fn, IDiffVector x,
            IDoubleVector upstreamGradient) {
        var result = AD.vjp(fn, x);
        return result.vjpFn().apply(AD.vector(upstreamGradient)).getValue().getData();
    }

    /**
     * Full Jacobian matrix computed via n JVP calls.
     * Only practical for small n.
     */
    public static IDoubleMatrix jacobianFull(Function<IDiffVector, IDiffVector> fn, IDiffVector x) {
        int n = x.getValue().length();
        IDiffVector y = fn.apply(x);
        int m = y.getValue().length();
        double[][] J = new double[m][n];
        for (int i = 0; i < n; i++) {
            double[] ei = new double[n];
            ei[i] = 1.0;
            double[] col = jvp(fn, x, IDoubleVector.of(ei));
            for (int j = 0; j < m; j++) {
                J[j][i] = col[j];
            }
        }
        return IDoubleMatrix.of(J);
    }
}
