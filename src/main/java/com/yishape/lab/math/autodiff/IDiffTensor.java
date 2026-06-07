package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;

/**
 * 可微张量：包装 IDiffVector，提供维度感知的自动微分.
 * <p>
 * 设计原则：不重新实现 AD 图，而是委托给底层的展平 IDiffVector。
 * 维度操作通过视图系统管理，backward 时将梯度反映射回展平表示.
 * <p>
 * 典型用途：Parameter.asLeafTensor() → 在 forward 中使用维度 API →
 * loss.backward() → flattenGrad() 供优化器 step.
 */
public interface IDiffTensor extends IDoubleTensor {

    // ==================== 可微运算 ====================

    @Override IDiffTensor add(IDoubleTensor other);
    @Override IDiffTensor sub(IDoubleTensor other);
    @Override IDiffTensor mul(IDoubleTensor other);
    @Override IDiffTensor div(IDoubleTensor other);
    @Override IDiffTensor add(double scalar);
    @Override IDiffTensor sub(double scalar);
    @Override IDiffTensor mul(double scalar);
    @Override IDiffTensor div(double scalar);

    /** Reverse-subtract: scalar - this. */
    IDiffTensor rsub(double scalar);
    /** Reverse-divide: scalar / this. */
    IDiffTensor rdiv(double scalar);
    /** Element-wise reciprocal (1/x). */
    IDiffTensor reciprocal();

    @Override IDiffTensor neg();
    @Override IDiffTensor abs();
    @Override IDiffTensor sqrt();
    @Override IDiffTensor exp();
    @Override IDiffTensor log();
    @Override IDiffTensor sigmoid();
    @Override IDiffTensor relu();
    @Override IDiffTensor square();
    @Override IDiffTensor pow(double n);
    @Override IDiffTensor clamp(double min, double max);

    @Override IDiffTensor sin();
    @Override IDiffTensor cos();
    @Override IDiffTensor tan();

    IDiffTensor tanh();
    IDiffTensor silu();
    IDiffTensor gelu();
    IDiffTensor softplus(double beta);
    IDiffTensor mish();
    IDiffTensor elu(double alpha);
    IDiffTensor leakyRelu(double alpha);
    IDiffTensor selu();
    IDiffTensor hardtanh(double minVal, double maxVal);
    IDiffTensor dropout(double p);

    @Override IDiffTensor sum(int dim, boolean keepdim);

    /**
     * Sum all elements. Returns a differentiable scalar tensor (shape [1]).
     * Equivalent to {@code flattenValue().sum()} wrapped as a scalar tensor.
     * 对所有元素求和，返回可微标量张量（shape [1]）。
     */
    default IDiffTensor sum() {
        IDiffVector flat = this.flattenValue();
        IDiffVector sumVec = flat.sum();
        return IDiffTensor.fromDiffVector(sumVec, 1);
    }

    @Override IDiffTensor mean(int dim, boolean keepdim);
    @Override IDiffTensor max(int dim, boolean keepdim);
    @Override IDiffTensor min(int dim, boolean keepdim);
    @Override IDiffTensor prod(int dim, boolean keepdim);
    @Override IDiffTensor cumsum(int dim);
    @Override IDiffTensor cumprod(int dim);
    @Override IDiffTensor argmax(int dim);
    @Override IDiffTensor argmin(int dim);
    @Override IDiffTensor std(int dim, boolean keepdim);
    @Override IDiffTensor var(int dim, boolean keepdim);

    @Override IDiffTensor softmax(int dim);
    @Override IDiffTensor logSoftmax(int dim);
    IDiffTensor softmaxCrossEntropy(IDoubleTensor labels, int dim);

    /**
     * Fused Layer Normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta.
     * Normalizes over the last dimension of the input tensor.
     * @param gamma scale parameter (size = last dimension)
     * @param beta  shift parameter (size = last dimension)
     * @param eps   small constant for numerical stability
     * @return normalized tensor with the same shape as input
     */
    IDiffTensor layerNorm(IDiffTensor gamma, IDiffTensor beta, double eps);

    /**
     * Fused Batch Normalization (training mode).
     * Normalizes over the batch dimension for each feature.
     * Input shape: [batch, features] — beta and gamma have size [features].
     * @param gamma scale parameter (size = features)
     * @param beta  shift parameter (size = features)
     * @param eps   small constant for numerical stability
     * @return normalized output with the same shape as input
     */
    IDiffTensor batchNorm(IDiffTensor gamma, IDiffTensor beta, double eps);

    @Override IDiffTensor mmul(IDoubleTensor other);
    @Override IDiffTensor bmm(IDoubleTensor other);
    @Override IDiffTensor einsum(String subscript, IDoubleTensor... others);

    @Override IDiffTensor reshape(int... newShape);
    @Override IDiffTensor permute(int... dims);
    @Override IDiffTensor transpose(int dim0, int dim1);
    @Override IDiffTensor transpose();
    @Override IDiffTensor squeeze(int... dims);
    @Override IDiffTensor unsqueeze(int dim);
    @Override IDiffTensor select(int dim, long index);
    @Override IDiffTensor flatten(int startDim, int endDim);
    @Override IDiffTensor slice(int dim, long start, long end);
    @Override IDiffTensor narrow(int dim, long start, long length);
    @Override IDiffTensor expand(int... shape);
    @Override IDiffTensor contiguous();
    @Override IDiffTensor tile(int... repeats);
    @Override IDiffTensor broadcastTo(int... shape);
    @Override IDiffTensor clone();

    @Override IDiffTensor gather(int dim, IDoubleTensor index);
    @Override IDiffTensor indexSelect(int dim, IDoubleTensor index);
    @Override IDiffTensor argsort(int dim, boolean descending);
    @Override IDiffTensor scatter(int dim, IDoubleTensor index, IDoubleTensor source);
    @Override IDiffTensor scatterAdd(int dim, IDoubleTensor index, IDoubleTensor source);
    @Override IDiffTensor where(IDoubleTensor condition, IDoubleTensor other);
    @Override IDiffTensor topk(int k, int dim, boolean largest);
    @Override IDiffTensor pad(int[][] padding, String mode, double value);
    @Override IDiffTensor unfold(int dim, int size, int stride, int dilation);
    @Override IDiffTensor nonzero();
    @Override IDiffTensor maskedSelect(IDoubleTensor mask);
    @Override IDiffTensor maskedFill(IDoubleTensor mask, double value);
    @Override IDiffTensor cat(int dim, IDoubleTensor... others);
    @Override IDiffTensor stack(int dim, IDoubleTensor... others);
    @Override IDiffTensor normalize(double p, int dim);

    @Override IDiffTensor add_(IDoubleTensor other);
    @Override IDiffTensor sub_(IDoubleTensor other);
    @Override IDiffTensor mul_(IDoubleTensor other);
    @Override IDiffTensor div_(IDoubleTensor other);
    @Override IDiffTensor fill_(double value);
    @Override IDiffTensor copy_(IDoubleTensor src);

    // ==================== 梯度方法 ====================

    /** 执行反向传播 */
    void backward();

    /** 使用指定梯度执行反向传播 */
    void backward(IDoubleTensor gradient);

    /** 清零梯度 */
    void zeroGradient();

    /** 展平梯度（供优化器使用） */
    IDiffVector flattenGrad();

    /** 展平值（供优化器使用） */
    IDiffVector flattenValue();

    /** 切断梯度追踪，返回普通 IDoubleTensor */
    IDoubleTensor detach();

    /** 是否追踪梯度 */
    boolean requiresGrad();

    /** 设置是否追踪梯度 */
    IDiffTensor setRequiresGrad(boolean requiresGrad);

    /** 当前累积梯度 */
    IDoubleTensor grad();

    // ==================== 工厂 ====================

    /** 从 IDiffVector + shape 创建可微张量 */
    static IDiffTensor fromDiffVector(IDiffVector vec, int... shape) {
        if (vec instanceof com.yishape.lab.math.autodiff.impl.RereDiffVector rv) {
            // Unwrap to underlying tensor; reshape only if shape differs
            int[] tensorShape = rv.tensor.shape();
            boolean sameShape = tensorShape.length == shape.length;
            if (sameShape) {
                for (int i = 0; i < shape.length; i++) {
                    if (tensorShape[i] != shape[i]) { sameShape = false; break; }
                }
            }
            if (sameShape) {
                return rv.tensor;
            }
            return rv.tensor.reshape(shape);
        }
        IDoubleVector val = vec.getValue();
        if (val == null) {
            throw new IllegalArgumentException("vec.getValue() returned null — cannot create tensor from null vector");
        }
        return new RereDiffTensor(val.toDoubleArray(), shape);
    }

    /** 从 IDoubleTensor + requiresGrad 创建可微张量（注：不会追踪前序梯度） */
    static IDiffTensor fromTensor(IDoubleTensor tensor, boolean requiresGrad) {
        RereDiffTensor t = new RereDiffTensor(tensor.toDoubleArray(), tensor.shape());
        return t.setRequiresGrad(requiresGrad);
    }

    /**
     * 创建不可微常量张量：不会构建 AD 计算图。
     */
    static IDiffTensor constantTensor(double[] data, int... shape) {
        RereDiffTensor t = new RereDiffTensor(data, shape);
        t.setRequiresGrad(false);
        return t;
    }
}
