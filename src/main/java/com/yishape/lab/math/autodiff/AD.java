package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.vjp.BatchVjpResult;
import com.yishape.lab.math.autodiff.vjp.VjpFunction;
import com.yishape.lab.math.autodiff.vjp.VjpResult;
import com.yishape.lab.math.autodiff.vmap.VMap;
import com.yishape.lab.util.Messages;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IOnlineOptimizer;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.autodiff.impl.CheckpointVariable;
import com.yishape.lab.math.autodiff.impl.CustomDiffVector;
import com.yishape.lab.math.linalg.IFloatVector;
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;
import com.yishape.lab.math.linalg.complex.IComplexMatrix.IComplexVector;
import com.yishape.lab.math.autodiff.impl.FloatDiffTensor;
import com.yishape.lab.math.autodiff.impl.FloatDiffVector;
import com.yishape.lab.math.autodiff.impl.FusedMatrixOps;
import com.yishape.lab.math.autodiff.impl.FusedOps;
import com.yishape.lab.math.autodiff.impl.FusedReductionOps;
import com.yishape.lab.math.autodiff.impl.ODEDiffVector;
import com.yishape.lab.math.autodiff.graph.GraphExporter;
import com.yishape.lab.math.autodiff.graph.GraphOptimizer;
import com.yishape.lab.math.autodiff.graph.GpuGraphExecutor;
import com.yishape.lab.math.autodiff.graph.HpcGraphExecutor;
import com.yishape.lab.math.autodiff.graph.GraphRenderer;
import com.yishape.lab.math.autodiff.impl.RereDiffComplex;
import com.yishape.lab.math.autodiff.impl.RereDiffMatrix;
import com.yishape.lab.math.autodiff.impl.RereDiffSparseMatrix;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import com.yishape.lab.math.autodiff.impl.TangentDiffVector;
import com.yishape.lab.math.autodiff.impl.TracerDiffVector;
import com.yishape.lab.math.autodiff.vjp.VjpFunctionImpl;
import com.yishape.lab.math.autodiff.IDiffComplex;
import com.yishape.lab.math.autodiff.IDiffSparseMatrix;

/**
 * Automatic differentiation facade.
 * 自动微分门面类。
 *
 * <p>Provides factory methods for differentiable variables (vector, matrix, sparse, complex),
 * reverse-mode and forward-mode AD, Neural ODE integration, gradient checkpointing,
 * operator fusion, computation-graph tools, gradient checking, and optimizer integration.
 *
 * <p>提供可微变量（向量、矩阵、稀疏、复数）工厂、反向/正向模式自动微分、Neural ODE 积分、
 * 梯度检查点、算子融合、计算图工具、梯度校验以及与优化器的集成。</p>
 *
 * <p>Typical usage: {@code var loss = x.pow(2).sum(); loss.backward(); var grad = x.getGradient();}</p>
 *
 * @author lteb2
 * @see IDiffVector
 * @see IDiffMatrix
 */
public class AD {

    private AD() {
    }

    // ---- vector factories ----

    /** Creates a leaf differentiable vector from a raw array. / 从原始数组创建叶子可微向量。 */
    public static IDiffVector vector(double... data) {
        return new RereDiffVector(IDoubleVector.of(data));
    }

    /**
     * Reuses an existing leaf node by updating its data in-place.
     * Avoids recreating the computation graph node.
     * Only valid on leaf nodes created by {@link #vector(double[])}.
     */
    public static IDiffVector reuseNode(IDiffVector leaf, double[] newData) {
        if (!(leaf instanceof RereDiffVector rdv)) {
            throw new IllegalArgumentException(
                "reuseNode requires RereDiffVector, got: " + leaf.getClass().getSimpleName());
        }
        rdv.updateData(newData);
        return leaf;
    }

    public static IDiffVector vector(IDoubleVector data) {
        return new RereDiffVector(data.copy());
    }

    public static IDiffVector vector(double scalar) {
        return new RereDiffVector(IDoubleVector.of(scalar));
    }

    public static IDiffVector zeros(int size) {
        return new RereDiffVector(IDoubleVector.zeros(size));
    }

    public static IDiffVector ones(int size) {
        return new RereDiffVector(IDoubleVector.ones(size));
    }

    // ---- tensor factories ----

    public static IDiffTensor tensor(double[] data, int... shape) {
        return IDiffTensor.fromDiffVector(vector(data), shape);
    }

    public static IDiffTensor zerosTensor(int... shape) {
        long total = 1;
        for (int s : shape) total *= s;
        return IDiffTensor.fromDiffVector(zeros((int) total), shape);
    }

    public static IDiffTensor onesTensor(int... shape) {
        long total = 1;
        for (int s : shape) total *= s;
        return IDiffTensor.fromDiffVector(ones((int) total), shape);
    }

    public static IDiffTensor fullTensor(double value, int... shape) {
        long total = 1;
        for (int s : shape) total *= s;
        double[] data = new double[(int) total];
        java.util.Arrays.fill(data, value);
        return IDiffTensor.fromDiffVector(vector(data), shape);
    }

    public static IDiffVector arange(double start, double end, double step) {
        int n = (int) Math.ceil((end - start) / step);
        double[] data = new double[Math.max(n, 0)];
        for (int i = 0; i < data.length; i++) data[i] = start + i * step;
        return vector(data);
    }

    public static IDiffVector arange(double start, double end) {
        return arange(start, end, 1.0);
    }

    public static IDiffVector arange(int end) {
        return arange(0, end, 1.0);
    }

    public static IDiffTensor eye(int n) {
        double[] data = new double[n * n];
        for (int i = 0; i < n; i++) data[i * n + i] = 1.0;
        return tensor(data, n, n);
    }

    // ---- matrix factories ----

    public static IDiffMatrix matrix(double[][] data) {
        return new RereDiffMatrix(IDoubleMatrix.of(data));
    }

    public static IDiffMatrix matrix(IDoubleMatrix data) {
        return new RereDiffMatrix(data.copy());
    }

    public static IDiffMatrix matrixZeros(int rows, int cols) {
        return new RereDiffMatrix(IDoubleMatrix.zeros(rows, cols));
    }

    public static IDiffMatrix matrixOnes(int rows, int cols) {
        return new RereDiffMatrix(IDoubleMatrix.ones(rows, cols));
    }

    // ---- sparse factories ----

    public static IDiffSparseMatrix sparse(ISparseMatrix data) {
        return new RereDiffSparseMatrix(data);
    }

    // ---- complex factories ----

    public static IDiffComplex complex(IComplexVector data) {
        return new RereDiffComplex(data);
    }

    // ---- mixed-precision factories ----

    public static IDiffVector diffFloat(float[] data) {
        return new FloatDiffVector(data);
    }

    public static IDiffVector diffFloat(IFloatVector data) {
        return new FloatDiffVector(data);
    }

    public static IDiffTensor diffFloatTensor(float[] data, int... shape) {
        return new FloatDiffTensor(data, shape);
    }

    // ---- constant factories (for tape-of-tape) ----

    public static IDiffVector constant(IDoubleVector value) {
        return new RereDiffVector(value.copy());
    }

    public static IDiffVector constant(double scalar) {
        return new RereDiffVector(IDoubleVector.of(scalar));
    }

    // ---- tape-of-tape higher-order differentiation ----

    /**
     * Symbolic gradient of {@code output} w.r.t. {@code inputs} (returns differentiable nodes).
     * 对 {@code inputs} 求 {@code output} 的符号梯度（返回可微节点，用于高阶微分）。
     */
    public static IDiffVector[] grad(IDiffVector output, IDiffVector... inputs) {
        return RereDiffVector.grad(output, inputs);
    }

    // ---- JIT operator fusion ----

    /** Starts a fused element-wise op chain on vector {@code x}. / 在向量 {@code x} 上构建融合逐元素算子链。 */
    public static FusedOps fuse(IDiffVector x) {
        return new FusedOps(x);
    }

    /**
     * Starts a fused element-wise + reduction op chain on vector {@code x}.
     * The chain may include element-wise ops followed by a reduction terminator
     * (softmax, normalize, layerNorm, sum, mean). All ops execute in a single
     * forward/backward kernel.
     *
     * <pre>{@code
     *   IDiffVector y = AD.fuseReduce(x).exp().relu().softmax();
     * }</pre>
     */
    public static FusedReductionOps fuseReduce(IDiffVector x) {
        return new FusedReductionOps((RereDiffVector) x);
    }

    public static FusedMatrixOps fuseMatrix(IDiffMatrix x) {
        return new FusedMatrixOps(x);
    }

    // ---- auto-fusion ----

    /**
     * Tries to fuse element-wise ops inside {@code fn}; falls back to eager evaluation if not fusible.
     * 尝试融合 {@code fn} 内逐元素运算；不可融合时回退为逐算子求值。
     */
    public static IDiffVector elementwise(IDiffVector x, Function<IDiffVector, IDiffVector> fn) {
        if (!(x instanceof RereDiffVector rx)) {
            throw new IllegalArgumentException(
                "elementwise requires RereDiffVector, got: " + x.getClass().getSimpleName());
        }
        TracerDiffVector tracer = new TracerDiffVector(rx);
        try {
            fn.apply(tracer);
            if (tracer.isFusible()) {
                return tracer.buildFused();
            }
        } catch (RuntimeException e) {
            // non-fusible op detected, fall through to fallback
        }
        return fn.apply(x);
    }

    // ---- custom gradient registration ----

    /**
     * @deprecated Use {@link CustomOp} and {@link #op(CustomOp, IDiffVector...)} instead.
     *             CustomOp embeds the backward function and auto-manages lifecycle, preventing
     *             registry-based memory leaks.
     */
    @Deprecated
    public static void registerGradient(String name,
            java.util.function.Function<IDoubleVector, IDoubleVector[]> backwardFn) {
        CustomGradientRegistry.register(name, backwardFn);
    }

    /**
     * @deprecated Use {@link CustomOp} and {@link #op(CustomOp, IDiffVector...)} instead.
     */
    @Deprecated
    public static void unregisterGradient(String name) {
        CustomGradientRegistry.unregister(name);
    }

    /**
     * @deprecated Use {@link CustomOp} and {@link #op(CustomOp, IDiffVector...)} instead.
     *             CustomOp eliminates the need for manual nodeId management and forward-data caching.
     */
    @Deprecated
    public static IDiffVector custom(String name,
            java.util.function.Function<IDiffVector[], IDiffVector> forwardFn, IDiffVector... inputs) {
        return CustomDiffVector.create(name, forwardFn, inputs);
    }

    /**
     * Apply a {@link CustomOp} to the given inputs.
     * This is the preferred replacement for {@link #custom(String, java.util.function.Function, IDiffVector...)}.
     * No global registration, no memory leaks — the backward function is embedded in the op.
     */
    public static IDiffVector op(CustomOp op, IDiffVector... inputs) {
        return op.apply(inputs);
    }

    // ---- gradient checkpointing ----

    /**
     * Gradient checkpointing: stores forward value only; recomputes forward during backward to save memory.
     * 梯度检查点：仅保存前向值，反向时重算前向以节省显存。
     */
    public static IDiffVector checkpoint(Function<IDiffVector, IDiffVector> fn, IDiffVector x) {
        if (!(x instanceof RereDiffVector rx)) {
            throw new IllegalArgumentException(
                "checkpoint requires RereDiffVector input, got: " + x.getClass().getSimpleName());
        }
        IDiffVector result = fn.apply(x);
        if (!(result instanceof RereDiffVector out)) {
            throw new IllegalArgumentException(
                "checkpoint function must return RereDiffVector, got: " + result.getClass().getSimpleName());
        }
        return new CheckpointVariable(out.getValue().copy(), fn, rx);
    }

    // ---- neural ODE ----

    /**
     * Differentiable ODE integration (Neural ODE): dz/dt = dynamics(z), RK4 forward + adjoint backward.
     * 可微 ODE 积分（Neural ODE）：RK4 前向 + 伴随法反向。
     */
    public static IDiffVector odeint(Function<IDiffVector, IDiffVector> dynamics, IDiffVector z0,
            double t0, double t1, double dt) {
        return new ODEDiffVector(dynamics, z0, t0, t1, dt);
    }

    // ---- computation graph visualization ----

    /** Renders vector DAG as Graphviz DOT. / 将向量计算图渲染为 Graphviz DOT。 */
    public static String render(IDiffVector root) {
        if (!(root instanceof RereDiffVector rdv)) {
            throw new IllegalArgumentException(
                "render requires RereDiffVector, got: " + root.getClass().getSimpleName());
        }
        return GraphRenderer.renderVector(rdv);
    }

    public static String render(IDiffMatrix root) {
        if (!(root instanceof RereDiffMatrix rdm)) {
            throw new IllegalArgumentException(
                "render requires RereDiffMatrix, got: " + root.getClass().getSimpleName());
        }
        return GraphRenderer.renderMatrix(rdm);
    }

    // ---- graph optimization ----

    public static IDiffVector optimize(IDiffVector x) {
        return GraphOptimizer.optimize(x);
    }

    public static GraphOptimizer.GraphStats graphStats(IDiffVector x) {
        return GraphOptimizer.stats(x);
    }

    // ---- HPC bridge ----

    /** Exports computation graph as JSON for native HPC execution. / 导出 JSON 计算图供 HPC 执行。 */
    public static String exportGraph(IDiffVector root) {
        if (!(root instanceof RereDiffVector rdv)) {
            throw new IllegalArgumentException(
                "exportGraph requires RereDiffVector, got: " + root.getClass().getSimpleName());
        }
        return GraphExporter.toJson(rdv);
    }

    /**
     * Attempts HPC graph execution. On success, leaf gradients are populated
     * and the loss value is stored in the root node's value.
     *
     * @param root the computation graph root
     * @return true if HPC execution succeeded and gradients were applied
     */
    public static boolean tryHpcExecute(IDiffVector root) {
        if (!(root instanceof RereDiffVector rdv)) return false;
        return !Double.isNaN(HpcGraphExecutor.tryExecute(rdv));
    }

    /**
     * Attempts GPU graph execution. On success, leaf gradients are populated
     * and the loss value is stored in the root node's value.
     *
     * @param root the computation graph root
     * @return true if GPU execution succeeded and gradients were applied
     */
    public static boolean tryGpuExecute(IDiffVector root) {
        if (!(root instanceof RereDiffVector rdv)) return false;
        return !Double.isNaN(GpuGraphExecutor.tryExecute(rdv));
    }

    // ---- numerical gradient checker ----

    /** Returns true if analytical gradient matches central differences within {@code tolerance}. / 梯度校验是否通过。 */
    public static boolean checkGradient(Function<IDiffVector, IDiffVector> lossFn, IDiffVector x, double tolerance) {
        return checkGradientDetailed(lossFn, x, tolerance).passed();
    }

    public static GradientCheckResult checkGradientDetailed(Function<IDiffVector, IDiffVector> lossFn,
            IDiffVector x, double tolerance) {
        double epsilon = 1e-6;
        if (!(x instanceof RereDiffVector rx)) {
            throw new IllegalArgumentException(
                "checkGradient requires RereDiffVector input, got: " + x.getClass().getSimpleName());
        }
        IDoubleVector xVal = rx.getValue();
        int n = xVal.size();

        IDiffVector y = lossFn.apply(x);
        if (!(y instanceof RereDiffVector ry)) {
            throw new IllegalArgumentException(
                "checkGradient loss function must return RereDiffVector, got: " + y.getClass().getSimpleName());
        }
        ry.backward();
        IDoubleVector analytical = rx.getGradient();
        if (analytical == null) {
            return new GradientCheckResult(false, Double.NaN, Double.NaN, Double.NaN,
                    new int[0], new double[0], new double[0]);
        }
        double[] aGrad = analytical.getData();

        double[] nGrad = new double[n];
        double maxAbsError = 0;
        double maxRelError = 0;
        double sumAbsError = 0;
        int suspiciousCount = 0;
        int[] suspiciousTemp = new int[n];

        for (int i = 0; i < n; i++) {
            double orig = xVal.get(i);
            double[] xpData = xVal.getData().clone();
            xpData[i] = orig + epsilon;
            IDiffVector xp = new RereDiffVector(IDoubleVector.of(xpData));
            double fp = lossFn.apply(xp).getValue().get(0);

            double[] xmData = xVal.getData().clone();
            xmData[i] = orig - epsilon;
            IDiffVector xm = new RereDiffVector(IDoubleVector.of(xmData));
            double fm = lossFn.apply(xm).getValue().get(0);

            nGrad[i] = (fp - fm) / (2.0 * epsilon);
            double absErr = Math.abs(aGrad[i] - nGrad[i]);
            double relErr = absErr / (Math.max(Math.abs(aGrad[i]), Math.abs(nGrad[i])) + 1e-15);
            maxAbsError = Math.max(maxAbsError, absErr);
            maxRelError = Math.max(maxRelError, relErr);
            sumAbsError += absErr;

            if (relErr > tolerance || (Double.isNaN(aGrad[i]) != Double.isNaN(nGrad[i]))) {
                suspiciousTemp[suspiciousCount++] = i;
            }
        }

        int[] suspicious = java.util.Arrays.copyOf(suspiciousTemp, suspiciousCount);
        double meanAbsError = sumAbsError / n;
        boolean passed = maxRelError <= tolerance;

        return new GradientCheckResult(passed, maxAbsError, maxRelError, meanAbsError,
                suspicious, aGrad, nGrad);
    }

    // ---- forward-mode AD (Jacobian) ----

    /** Seeds forward-mode AD with primal and tangent direction. / 以前向值与切向量播种正向模式 AD。 */
    public static IDiffVector tangent(IDiffVector primal, IDiffVector tangent) {
        if (!(primal instanceof RereDiffVector rp)) {
            throw new IllegalArgumentException(
                "tangent requires RereDiffVector primal, got: " + primal.getClass().getSimpleName());
        }
        if (!(tangent instanceof RereDiffVector rt)) {
            throw new IllegalArgumentException(
                "tangent requires RereDiffVector tangent, got: " + tangent.getClass().getSimpleName());
        }
        return TangentDiffVector.seed(rp, rt.getValue());
    }

    /** Jacobian of {@code fn} at {@code x} via forward-mode AD (column = J·e_j). / 正向模式求 Jacobian。 */
    public static IDoubleMatrix jacobian(Function<IDiffVector, IDiffVector> fn, IDiffVector x) {
        if (!(x instanceof RereDiffVector rx)) {
            throw new IllegalArgumentException(
                "jacobian requires RereDiffVector input, got: " + x.getClass().getSimpleName());
        }
        int n = rx.getValue().size();
        IDiffVector y0 = fn.apply(rx);
        if (!(y0 instanceof RereDiffVector ry0)) {
            throw new IllegalArgumentException(
                "jacobian function must return RereDiffVector, got: " + y0.getClass().getSimpleName());
        }
        int m = ry0.getValue().size();
        double[][] jac = new double[m][n];
        for (int j = 0; j < n; j++) {
            IDoubleVector e = IDoubleVector.zeros(n);
            e.getData()[j] = 1.0;
            TangentDiffVector tv = TangentDiffVector.seed(rx, e);
            IDiffVector result = fn.apply(tv);
            IDoubleVector jvp = ((TangentDiffVector) result).getTangent();
            for (int i = 0; i < m; i++) {
                jac[i][j] = jvp.get(i);
            }
        }
        return IDoubleMatrix.of(jac);
    }

    // ---- optimizer integration ----

    /**
     * Minimizes scalar loss built from {@code lossBuilder} using {@code optimizer} and autodiff gradients.
     * 用自动微分梯度配合 {@code optimizer} 最小化 {@code lossBuilder} 定义的标量损失。
     */
    public static OptResult optimize(IVector initX, Function<IDiffVector, IDiffVector> lossBuilder,
            IOptimizer optimizer) {
        IObjectiveFunction objFun = x -> {
            IDiffVector var = new RereDiffVector((IDoubleVector) x);
            return lossBuilder.apply(var).getValue().get(0);
        };

        IGradientFunction grdFun = x -> {
            IDiffVector var = new RereDiffVector((IDoubleVector) x);
            IDiffVector loss = lossBuilder.apply(var);
            // GPU → HPC → CPU fallback chain
            if (!tryGpuExecute(loss)) {
                if (!tryHpcExecute(loss)) {
                    loss.backward();
                }
            }
            return var.getGradient();
        };

        return optimizer.optimize(initX, objFun, grdFun);
    }

    // ---- online learning integration ----

    public static <T> IOnlineOptimizer autogradOptimizer(IOnlineOptimizer base,
            BiFunction<IDiffVector, T, IDiffVector> lossBuilder) {
        return new IOnlineOptimizer() {
            private T lastSample;

            @Override public void initialize(IVector initialParams) { base.initialize(initialParams); }
            @Override public IVector step(IVector gradient) { return base.step(gradient); }
            @Override public IVector step(IVector gradient, double loss) { return base.step(gradient, loss); }

            @Override
            public <S> IVector step(S sample, BiFunction<IVector, S, Double> lossFunction) {
                @SuppressWarnings("unchecked")
                T t = (T) sample;
                lastSample = t;
                IVector params = base.getCurrentParams();
                IDiffVector var = new RereDiffVector((IDoubleVector) params);
                IDiffVector loss = lossBuilder.apply(var, t);
                // GPU → HPC → CPU fallback chain
                if (!tryGpuExecute(loss)) {
                    if (!tryHpcExecute(loss)) {
                        loss.backward();
                    }
                }
                return base.step(var.getGradient());
            }

            @Override public IVector getCurrentParams() { return base.getCurrentParams(); }
            @Override public void setCurrentParams(IVector params) { base.setCurrentParams(params); }
            @Override public double getCurrentLearningRate() { return base.getCurrentLearningRate(); }
            @Override public void setLearningRate(double learningRate) { base.setLearningRate(learningRate); }
            @Override public int getCurrentStep() { return base.getCurrentStep(); }
            @Override public void reset() { base.reset(); }
            @Override public boolean isInitialized() { return base.isInitialized(); }
            @Override public IOnlineOptimizer clone() { return base.clone(); }
        };
    }

    public static <T> IVector onlineLearn(IVector initParams, List<T> data,
            BiFunction<IDiffVector, T, IDiffVector> lossBuilder, IOnlineOptimizer optimizer, int epochs) {
        optimizer.initialize(initParams);
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (T sample : data) {
                IVector params = optimizer.getCurrentParams();
                IDiffVector var = new RereDiffVector((IDoubleVector) params);
                IDiffVector loss = lossBuilder.apply(var, sample);
                // GPU → HPC → CPU fallback chain
                if (!tryGpuExecute(loss)) {
                    if (!tryHpcExecute(loss)) {
                        loss.backward();
                    }
                }
                optimizer.step(var.getGradient(), loss.getValue().get(0));
            }
        }
        return optimizer.getCurrentParams();
    }

    // ---- VJP (Vector-Jacobian Product) ----

    /**
     * Computes the Vector-Jacobian Product transform of {@code fn} at {@code x}.
     *
     * <p>Returns both the forward output {@code y = fn(x)} and a reusable
     * {@link VjpFunction} that computes J<sup>T</sup> @ g for arbitrary
     * upstream gradients g without rebuilding the computation graph.
     *
     * <p>计算 {@code fn} 在 {@code x} 处的 VJP 变换。
     * 返回前向输出 y = fn(x) 和可重用的 VJP 算子。
     *
     * @param fn  vector-valued function R^n → R^m
     * @param x   input leaf variable
     * @return VjpResult containing y and the reusable VJP function
     */
    public static VjpResult vjp(Function<IDiffVector, IDiffVector> fn, IDiffVector x) {
        IDiffVector y = fn.apply(x);
        if (!(y instanceof RereDiffVector ry)) {
            throw new IllegalArgumentException(
                "vjp function must return RereDiffVector, got: " + y.getClass().getSimpleName());
        }
        if (!(x instanceof RereDiffVector rx)) {
            throw new IllegalArgumentException(
                "vjp requires RereDiffVector input, got: " + x.getClass().getSimpleName());
        }
        return new VjpResult(y, new VjpFunctionImpl(ry, rx));
    }

    /**
     * Batch VJP: computes VJP for each input in the list.
     *
     * <p>对一批输入计算每个样本的 VJP。比分别调用 N 次 vjp 更方便，
     * 并提供 sumGradients / meanGradients 便捷方法。
     *
     * @param fn  vector-valued function R^n → R^m
     * @param xs  list of input leaf variables
     * @return BatchVjpResult containing all forward outputs and VJP functions
     */
    public static BatchVjpResult batchVjp(Function<IDiffVector, IDiffVector> fn, List<? extends IDiffVector> xs) {
        int n = xs.size();
        if (n == 0) {
            throw new IllegalArgumentException("batchVjp requires non-empty input list");
        }
        IDiffVector[] ys = new IDiffVector[n];
        VjpFunction[] vjpFns = new VjpFunction[n];
        for (int i = 0; i < n; i++) {
            VjpResult result = vjp(fn, xs.get(i));
            ys[i] = result.y();
            vjpFns[i] = result.vjpFn();
        }
        return new BatchVjpResult(ys, vjpFns);
    }

    // ---- vmap (automatic batching) ----

    /**
     * Vectorized map: stacks inputs into a single flat vector, executes
     * {@code fn} once on a {@link BatchedDiffVector}, then unstacks the result.
     *
     * <p>Uses <strong>single-graph batched execution</strong>: all N samples share
     * one computation graph. Reductions ({@code sum()}, {@code mean()}) inside
     * {@code fn} are automatically applied per-sample over the inner dimension.
     * Element-wise operations are shape-agnostic and work unchanged on the
     * flattened representation.
     *
     * <p><strong>Supported operations inside {@code fn}:</strong>
     * <ul>
     *   <li>Element-wise: {@code add, sub, mul, div, pow, exp, log, sin, cos,
     *       tanh, sigmoid, relu, gelu, abs, sqrt, square}, and all activations</li>
     *   <li>Reductions: {@code sum(), mean()} (applied per-sample)</li>
     *   <li>{@code dropout(p)} (element-wise, seed-shared)</li>
     * </ul>
     *
     * <p><strong>Unsupported inside {@code fn}:</strong>
     * {@code slice, dot, broadcast, cat, softmax, layerNorm, batchNorm}
     * — these operations are incompatible with the batched flat representation.
     *
     * <p>对列表中每个样本独立应用 {@code fn}，使用单图批量执行。
     *
     * @param fn  function R^D → R^M operating on single elements (element-wise + sum/mean only)
     * @param xs  list of input vectors, all same length D
     * @return array of per-element outputs, each of length M
     */
    public static IDiffVector[] vmap(Function<IDiffVector, IDiffVector> fn, List<? extends IDiffVector> xs) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = xs.size();
        int d = xs.get(0).size();

        // Validate uniform dimension
        for (int i = 1; i < n; i++) {
            if (xs.get(i).size() != d) {
                throw new IllegalArgumentException(
                    Messages.get("vmap.dimension_mismatch", d, xs.get(i).size()));
            }
        }

        // Stack inputs into a single flat array
        double[][] raw = new double[n][];
        for (int i = 0; i < n; i++) {
            raw[i] = xs.get(i).getValue().getData();
        }
        double[] stacked = VMap.INSTANCE.stack(raw);

        // Single-graph batched execution
        IDiffVector batchedInput = vector(stacked);
        BatchedDiffVector bdv = new BatchedDiffVector(batchedInput, n, d);
        IDiffVector batchedResult = fn.apply(bdv);

        // Unstack results: split flat output into per-sample slices
        IDiffVector flat = (batchedResult instanceof BatchedDiffVector b)
            ? b.unwrap() : batchedResult;
        int outDim = flat.size() / n;
        IDiffVector[] ys = new IDiffVector[n];
        for (int i = 0; i < n; i++) {
            ys[i] = flat.slice(i * outDim, (i + 1) * outDim);
        }
        return ys;
    }

    /**
     * Convenience: apply fn to each element and sum the results.
     *
     * <p>Uses single-graph batched execution via {@link #vmap(Function, List)},
     * then sums across the batch dimension.
     *
     * <p>便捷方法：对每个样本应用 fn 并求和结果。
     */
    public static IDiffVector vmapSum(Function<IDiffVector, IDiffVector> fn, List<? extends IDiffVector> xs) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = xs.size();
        int d = xs.get(0).size();

        for (int i = 1; i < n; i++) {
            if (xs.get(i).size() != d) {
                throw new IllegalArgumentException(
                    Messages.get("vmap.dimension_mismatch", d, xs.get(i).size()));
            }
        }

        double[][] raw = new double[n][];
        for (int i = 0; i < n; i++) raw[i] = xs.get(i).getValue().getData();
        double[] stacked = VMap.INSTANCE.stack(raw);

        IDiffVector batchedInput = vector(stacked);
        BatchedDiffVector bdv = new BatchedDiffVector(batchedInput, n, d);
        IDiffVector result = fn.apply(bdv);

        // Sum across batch dimension
        IDiffVector flat = (result instanceof BatchedDiffVector b) ? b.unwrap() : result;
        int outDim = flat.size() / n;
        if (outDim == 1) {
            return flat.sum();
        }
        return flat.reshape(n, outDim).sum(0);
    }

    /**
     * Convenience: apply fn to each element and return the mean.
     *
     * <p>Uses single-graph batched execution via {@link #vmap(Function, List)},
     * then averages across the batch dimension.
     *
     * <p>便捷方法：对每个样本应用 fn 并返回均值。
     */
    public static IDiffVector vmapMean(Function<IDiffVector, IDiffVector> fn, List<? extends IDiffVector> xs) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = xs.size();
        int d = xs.get(0).size();

        for (int i = 1; i < n; i++) {
            if (xs.get(i).size() != d) {
                throw new IllegalArgumentException(
                    Messages.get("vmap.dimension_mismatch", d, xs.get(i).size()));
            }
        }

        double[][] raw = new double[n][];
        for (int i = 0; i < n; i++) raw[i] = xs.get(i).getValue().getData();
        double[] stacked = VMap.INSTANCE.stack(raw);

        IDiffVector batchedInput = vector(stacked);
        BatchedDiffVector bdv = new BatchedDiffVector(batchedInput, n, d);
        IDiffVector result = fn.apply(bdv);

        // Mean across batch dimension
        IDiffVector flat = (result instanceof BatchedDiffVector b) ? b.unwrap() : result;
        int outDim = flat.size() / n;
        if (outDim == 1) {
            return flat.mean();
        }
        return flat.reshape(n, outDim).sum(0).div(n);
    }

    // ---- IDiffTensor vmap overloads ----

    /**
     * IDiffTensor version of {@link #vmap(Function, List)}.
     *
     * <p>Stacks tensors along dim 0 into a single {@link BatchedDiffTensor},
     * executes {@code fn} once with single-graph batched execution, then
     * unstacks results along the batch dimension.
     *
     * <p>Same constraints as vector vmap: only element-wise operations and
     * {@code sum()/sum(dim,keepdim)/mean(dim,keepdim)} are supported inside
     * {@code fn}. Dimension-indexed operations have their dim shifted by +1.
     */
    public static IDiffTensor[] vmapT(Function<IDiffTensor, IDiffTensor> fn, List<? extends IDiffTensor> xs) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = xs.size();

        // Stack all tensors along dim 0 → [B, D1, D2, ...]
        IDiffTensor[] rest = new IDiffTensor[n - 1];
        for (int i = 1; i < n; i++) {
            rest[i - 1] = xs.get(i);
        }
        IDiffTensor batched = xs.get(0).stack(0, rest);

        // Single-graph batched execution
        BatchedDiffTensor bdt = new BatchedDiffTensor(batched);
        IDiffTensor result = fn.apply(bdt);

        // Unstack results along batch dim 0
        IDiffTensor flat = (result instanceof BatchedDiffTensor b) ? b.unwrap() : result;
        IDiffTensor[] ys = new IDiffTensor[n];
        for (int i = 0; i < n; i++) {
            ys[i] = flat.select(0, i);
        }
        return ys;
    }

    /**
     * IDiffTensor version of {@link #vmapSum(Function, List)}.
     *
     * <p>Uses single-graph batched execution, then sums across the batch
     * dimension (dim 0).
     */
    public static IDiffTensor vmapSumT(Function<IDiffTensor, IDiffTensor> fn, List<? extends IDiffTensor> xs) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = xs.size();

        IDiffTensor[] rest = new IDiffTensor[n - 1];
        for (int i = 1; i < n; i++) {
            rest[i - 1] = xs.get(i);
        }
        IDiffTensor batched = xs.get(0).stack(0, rest);

        BatchedDiffTensor bdt = new BatchedDiffTensor(batched);
        IDiffTensor result = fn.apply(bdt);

        IDiffTensor flat = (result instanceof BatchedDiffTensor b) ? b.unwrap() : result;
        return flat.sum(0, false);
    }

    /**
     * IDiffTensor version of {@link #vmapMean(Function, List)}.
     *
     * <p>Uses single-graph batched execution, then averages across the batch
     * dimension (dim 0).
     */
    public static IDiffTensor vmapMeanT(Function<IDiffTensor, IDiffTensor> fn, List<? extends IDiffTensor> xs) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = xs.size();

        IDiffTensor[] rest = new IDiffTensor[n - 1];
        for (int i = 1; i < n; i++) {
            rest[i - 1] = xs.get(i);
        }
        IDiffTensor batched = xs.get(0).stack(0, rest);

        BatchedDiffTensor bdt = new BatchedDiffTensor(batched);
        IDiffTensor result = fn.apply(bdt);

        IDiffTensor flat = (result instanceof BatchedDiffTensor b) ? b.unwrap() : result;
        return flat.sum(0, false).div(n);
    }
}
