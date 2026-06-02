package com.yishape.lab.math.autodiff;

import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

/**
 * Differentiable vector in reverse-mode automatic differentiation.
 * 反向模式自动微分中的可微向量。
 *
 * <p>Builds a computation graph through fluent operations; call {@link #backward()} on a scalar
 * or reduced output, then read gradients via {@link #getGradient()} on leaf nodes.
 *
 * <p>通过链式运算构建计算图；对标量或归约结果调用 {@link #backward()} 后，
 * 在叶子节点上通过 {@link #getGradient()} 读取梯度。</p>
 */
public interface IDiffVector extends IDoubleVector {

    long serialVersionUID = 4L;

    /** Primal (forward) value. / 前向传播值。 */
    IDoubleVector getValue();

    /** Element at index {@code i} of the primal value. / 前向值第 {@code i} 个元素。 */
    @Override
    default double get(int i) {
        return this.getValue().get(i);
    }

    /** Accumulated gradient after {@link #backward()}; {@code null} if not yet computed. / 反向后的累积梯度。 */
    IDoubleVector getGradient();

    /** Whether this node is a leaf (trainable parameter). / 是否为叶子（可训练参数）节点。 */
    boolean isLeaf();

    /** Reverse-mode AD with unit initial gradient (for scalar outputs). / 以单位初始梯度做反向传播。 */
    void backward();

    /** Reverse-mode AD with a custom upstream gradient. / 以自定义上游梯度做反向传播。 */
    void backward(IDoubleVector initialGradient);

    /** Clears accumulated gradients on this node. / 清除本节点累积梯度。 */
    void zeroGradient();

    /** Returns the gradient as a new differentiable node (tape-of-tape). / 将梯度包装为新的可微节点。 */
    IDiffVector grad();

    // -- arithmetic with another variable --
    IDiffVector add(IDiffVector other);

    IDiffVector sub(IDiffVector other);

    IDiffVector mul(IDiffVector other);

    IDiffVector div(IDiffVector other);

    // -- arithmetic with scalar --
    IDiffVector add(double scalar);

    IDiffVector sub(double scalar);

    IDiffVector mul(double scalar);

    IDiffVector div(double scalar);

    IDiffVector rsub(double scalar);

    IDiffVector rdiv(double scalar);

    // -- unary --
    IDiffVector neg();

    IDiffVector pow(double n);

    // -- element-wise math --
    IDiffVector exp();

    IDiffVector log();

    IDiffVector sin();

    IDiffVector cos();

    IDiffVector tan();

    IDiffVector tanh();

    IDiffVector sigmoid();

    IDiffVector relu();

    IDiffVector softmax();

    IDiffVector logSoftmax();

    IDiffVector gelu();

    IDiffVector leakyRelu(double alpha);

    IDiffVector elu(double alpha);

    IDiffVector selu();

    IDiffVector silu();

    IDiffVector mish();

    IDiffVector softplus(double beta);

    IDiffVector hardtanh(double minVal, double maxVal);

    IDiffVector clamp(double min, double max);

    /** Fused Layer Normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta. */
    IDiffVector layerNorm(IDiffVector gamma, IDiffVector beta, double eps);

    /**
     * Fused Batch Normalization (training mode).
     * Normalizes over the batch dimension for each feature.
     * y = gamma * (x - batch_mean) / sqrt(batch_var + eps) + beta
     *
     * @param gamma scale parameter (size = features)
     * @param beta  shift parameter (size = features)
     * @param eps   small constant for numerical stability
     * @return normalized output
     */
    IDiffVector batchNorm(IDiffVector gamma, IDiffVector beta, double eps);

    IDiffVector dropout(double p);

    IDiffVector abs();

    IDiffVector sqrt();

    IDiffVector square();

    // -- reductions --
    IDiffVector sum();

    IDiffVector mean();

    // -- vector operations --
    IDiffVector dot(IDiffVector other);

    IDiffVector broadcast(int n);

    /** Returns a differentiable sub-vector slice [start, end). Backward scatters gradient back to original positions. */
    IDiffVector slice(int start, int end);

    /**
     * Returns this 1-D vector as a 1-D {@link IDiffTensor}.
     * <p>Establishes the conceptual relationship: a vector IS a rank-1 tensor.</p>
     */
    default IDiffTensor asTensor() {
        return IDiffTensor.fromDiffVector(this, new int[]{length()});
    }

    // -- reshape --
    @Override
    default IDiffMatrix reshape(int rows, int cols) {
        throw new UnsupportedOperationException(
            "reshape not supported on " + this.getClass().getSimpleName());
    }

    // -- in-place operations (leaf variables only) / 原地运算（仅叶子变量） --

    /** In-place add; mutates primal value of a leaf. / 原地相加，仅修改叶子节点的前向值。 */
    IDiffVector addInPlace(IDiffVector other);

    /** In-place scalar multiply. / 原地标量乘法。 */
    IDiffVector mulInPlace(double scalar);

    // ======================================================================
    // Covariant overrides from IDoubleVector / IVector
    // ======================================================================

    // --- Basic accessors (delegate to primal value) ---
    @Override
    default double[] getData() { return getValue().getData(); }

    @Override
    default double[] toDoubleArray() { return getValue().toDoubleArray(); }

    @Override
    default int size() { return getValue().size(); }

    @Override
    default int length() { return getValue().length(); }

    // --- copy ---
    @Override
    IDiffVector copy();

    // --- Scalar ops (name aliases for add/sub/mul/div with scalars) ---
    @Override
    default IDiffVector addScalar(double p) { return this.add(p); }

    @Override
    default IDiffVector subScalar(double p) { return this.sub(p); }

    @Override
    default IDiffVector multiplyByScalar(double p) { return this.mul(p); }

    @Override
    default IDiffVector divideByScalar(double p) { return this.div(p); }

    @Override
    IDiffVector divideInPlace(double alpha);

    // --- log10 ---
    @Override
    default IDiffVector log10() {
        throw new UnsupportedOperationException(
            "log10 not supported on " + this.getClass().getSimpleName());
    }

    // --- clip (alias for clamp) ---
    @Override
    default IDiffVector clip(double lower, double upper) {
        return this.clamp(lower, upper);
    }

    // --- Slice variants ---
    @Override
    default IDiffVector slice(int start) {
        return this.slice(start, this.length());
    }

    @Override
    default IDiffVector slice(int start, int end, int step) {
        throw new UnsupportedOperationException(
            "slice(start,end,step) not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector slice(String sliceExpression) {
        throw new UnsupportedOperationException(
            "slice(String) not supported on " + this.getClass().getSimpleName());
    }

    // --- Fancy / boolean indexing ---
    @Override
    default IDiffVector fancyGet(int[] positions) {
        throw new UnsupportedOperationException(
            "fancyGet not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector booleanGet(boolean[] booleanIndex) {
        throw new UnsupportedOperationException(
            "booleanGet not supported on " + this.getClass().getSimpleName());
    }

    // --- Element-wise ops with IVector<Double> parameter (AD-aware implementations needed) ---
    @Override
    IDiffVector add(IVector<Double> vec);

    @Override
    IDiffVector sub(IVector<Double> vec);

    @Override
    IDiffVector multiply(IVector<Double> vec);

    @Override
    IDiffVector divide(IVector<Double> other);

    // --- Matrix operations ---
    @Override
    default IDiffVector mmul(IMatrix<Double> other) {
        throw new UnsupportedOperationException(
            "mmul not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector dot(IMatrix<Double> m) {
        throw new UnsupportedOperationException(
            "dot(IMatrix) not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDoubleMatrix asColumnVector() {
        throw new UnsupportedOperationException(
            "asColumnVector not supported on " + this.getClass().getSimpleName());
    }

    // --- Trig/hyperbolic (missing from IDiffVector) ---
    @Override
    default IDiffVector arcsin() {
        throw new UnsupportedOperationException(
            "arcsin not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector arccos() {
        throw new UnsupportedOperationException(
            "arccos not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector arctan() {
        throw new UnsupportedOperationException(
            "arctan not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector sinh() {
        throw new UnsupportedOperationException(
            "sinh not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector cosh() {
        throw new UnsupportedOperationException(
            "cosh not supported on " + this.getClass().getSimpleName());
    }

    // --- Rounding ---
    @Override
    default IDiffVector round() {
        throw new UnsupportedOperationException(
            "round not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector floor() {
        throw new UnsupportedOperationException(
            "floor not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector ceil() {
        throw new UnsupportedOperationException(
            "ceil not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector trunc() {
        throw new UnsupportedOperationException(
            "trunc not supported on " + this.getClass().getSimpleName());
    }

    // --- Array operations ---
    @Override
    default IDiffVector cumsum() {
        throw new UnsupportedOperationException(
            "cumsum not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector cumprod() {
        throw new UnsupportedOperationException(
            "cumprod not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector diff() {
        throw new UnsupportedOperationException(
            "diff not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector diff(int n) {
        throw new UnsupportedOperationException(
            "diff(int) not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector sort() {
        throw new UnsupportedOperationException(
            "sort not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector reverse() {
        throw new UnsupportedOperationException(
            "reverse not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector normalize() {
        throw new UnsupportedOperationException(
            "normalize not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector reciprocal() {
        throw new UnsupportedOperationException(
            "reciprocal not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector cross(IVector<Double> other) {
        throw new UnsupportedOperationException(
            "cross not supported on " + this.getClass().getSimpleName());
    }

    // --- Conditional / indexing ---
    @Override
    default IDiffVector where(boolean[] condition, Double x, Double y) {
        throw new UnsupportedOperationException(
            "where not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector where(boolean[] condition, IVector<Double> x, IVector<Double> y) {
        throw new UnsupportedOperationException(
            "where(IVector) not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector repeat(int repeats) {
        throw new UnsupportedOperationException(
            "repeat not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector tile(int reps) {
        throw new UnsupportedOperationException(
            "tile not supported on " + this.getClass().getSimpleName());
    }

    // --- Map / concat / sign ---
    @Override
    default IDiffVector map(Function<Double, Double> fun) {
        throw new UnsupportedOperationException(
            "map not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector concat(IVector<Double> other) {
        throw new UnsupportedOperationException(
            "concat not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector sign() {
        throw new UnsupportedOperationException(
            "sign not supported on " + this.getClass().getSimpleName());
    }

    // --- Reductions (not yet in IDiffVector) ---
    @Override
    default IDiffVector min() {
        throw new UnsupportedOperationException(
            "min not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector max() {
        throw new UnsupportedOperationException(
            "max not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector std() {
        throw new UnsupportedOperationException(
            "std not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector std(int ddof) {
        throw new UnsupportedOperationException(
            "std(int) not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector var() {
        throw new UnsupportedOperationException(
            "var not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector var(int ddof) {
        throw new UnsupportedOperationException(
            "var(int) not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector prod() {
        throw new UnsupportedOperationException(
            "prod not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector norm2() {
        throw new UnsupportedOperationException(
            "norm2 not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector norm1() {
        throw new UnsupportedOperationException(
            "norm1 not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default IDiffVector ptp() {
        throw new UnsupportedOperationException(
            "ptp not supported on " + this.getClass().getSimpleName());
    }

    @Override
    IDiffVector innerProduct(IVector<Double> vec);

    @Override
    IDiffVector dot(IVector<Double> vec);

    // --- Hessian matrix (inherited from IVector) ---
    @Override
    default IDoubleMatrix hessianMatrix() {
        throw new UnsupportedOperationException(
            "hessianMatrix not supported on " + this.getClass().getSimpleName());
    }

    // --- DTW distance (inherited from IVector) ---
    @Override
    default double dtw(IVector<Double> other) {
        return getValue().dtw(other);
    }

    // --- Kurtosis (inherited from IVector) ---
    @Override
    default double kurtosis() {
        return getValue().kurtosis();
    }

    // --- Skewness (inherited from IVector) ---
    @Override
    default double skewness() {
        return getValue().skewness();
    }

    // --- Norm inf (inherited from IVector) ---
    @Override
    default double normInf() {
        return getValue().normInf();
    }

    // --- Median (inherited from IVector) ---
    @Override
    default double median() {
        return getValue().median();
    }

    // --- Mode (inherited from IVector) ---
    @Override
    default double mode() {
        return getValue().mode();
    }

    // --- Distance / similarity (inherited from IVector) ---
    @Override
    default double euclideanDistance(IVector<Double> other) {
        return getValue().euclideanDistance(other);
    }

    @Override
    default double manhattanDistance(IVector<Double> other) {
        return getValue().manhattanDistance(other);
    }

    @Override
    default double cosineSimilarity(IVector<Double> other) {
        return getValue().cosineSimilarity(other);
    }

    // --- Percentile & norm (inherited from IVector) ---
    @Override
    default double percentile(double q) {
        return getValue().percentile(q);
    }

    @Override
    default double norm(double p) {
        return getValue().norm(p);
    }

    // --- Logical operations (inherited from IVector) ---
    @Override
    default boolean[] logicalXor(IVector<Double> other) {
        return getValue().logicalXor(other);
    }

    @Override
    default boolean[] logicalNot() {
        return getValue().logicalNot();
    }

    @Override
    default int[] toIntArray() { return getValue().toIntArray(); }

    @Override
    default float[] toFloatArray() { return getValue().toFloatArray(); }

    // --- Search sorted (inherited from IVector) ---
    @Override
    default int searchSorted(double value) { return getValue().searchSorted(value); }

    @Override
    default boolean[] logicalAnd(IVector<Double> other) {
        return getValue().logicalAnd(other);
    }

    @Override
    default boolean[] logicalOr(IVector<Double> other) {
        return getValue().logicalOr(other);
    }

    // --- Covariance (inherited from IVector) ---
    @Override
    default double cov(IVector<Double> other) {
        throw new UnsupportedOperationException(
            "cov not supported on " + this.getClass().getSimpleName());
    }

    // --- Correlation (inherited from IVector) ---
    @Override
    default double corr(IVector<Double> other) {
        throw new UnsupportedOperationException(
            "corr not supported on " + this.getClass().getSimpleName());
    }

    // --- Arg min/max (inherited from IVector) ---
    @Override
    default int argMin() { return getValue().argMin(); }

    @Override
    default int argMax() { return getValue().argMax(); }

    // --- Outer product (inherited from IVector) ---
    @Override
    default IMatrix<Double> outer(IVector<Double> other) {
        return getValue().outer(other);
    }

    // --- Remainder (inherited from IVector) ---
    @Override
    default IDiffVector remainder(Double value) {
        throw new UnsupportedOperationException(
            "remainder not supported on " + this.getClass().getSimpleName());
    }

    // --- Mutation operations not supported on AD nodes ---
    @Override
    default void set(int position, double value) {
        throw new UnsupportedOperationException(
            "set not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default void setFromTo(int start, int end, int step, Double[] values) {
        throw new UnsupportedOperationException(
            "setFromTo not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default void setFromTo(int start, int end, Double[] values) {
        throw new UnsupportedOperationException(
            "setFromTo not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default void fill(double value) {
        throw new UnsupportedOperationException(
            "fill not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default void fancySet(int[] positions, Double[] values) {
        throw new UnsupportedOperationException(
            "fancySet not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default void fancySetScalar(int[] positions, Double value) {
        throw new UnsupportedOperationException(
            "fancySetScalar not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default void booleanSet(boolean[] booleanIndex, Double[] values) {
        throw new UnsupportedOperationException(
            "booleanSet not supported on " + this.getClass().getSimpleName());
    }

    @Override
    default void booleanSetScalar(boolean[] booleanIndex, Double value) {
        throw new UnsupportedOperationException(
            "booleanSetScalar not supported on " + this.getClass().getSimpleName());
    }

    // --- Comparison operations (inherited from IVector) ---
    @Override
    default boolean[] eq(IVector<Double> other) { return getValue().eq(other); }

    @Override
    default boolean[] lt(IVector<Double> other) { return getValue().lt(other); }

    @Override
    default boolean[] gt(IVector<Double> other) { return getValue().gt(other); }

    @Override
    default boolean[] ge(IVector<Double> other) { return getValue().ge(other); }

    @Override
    default boolean[] le(IVector<Double> other) { return getValue().le(other); }

}
