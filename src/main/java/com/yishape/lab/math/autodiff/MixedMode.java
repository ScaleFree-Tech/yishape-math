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
        IDiffVector y = fn.apply(x);
        IDiffVector[] grads = AD.grad(y, x);
        IDiffVector gradVec = grads[0];
        IDiffVector vConst = AD.constant(v);
        IDiffVector g = gradVec.dot(vConst);
        x.zeroGradient();
        g.backward();
        return x.getGradient().getData();
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
        int n = x.getValue().length();
        double[][] H = new double[n][n];
        for (int i = 0; i < n; i++) {
            double[] ei = new double[n];
            ei[i] = 1.0;
            double[] col = hvp(fn, x, IDoubleVector.of(ei));
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
