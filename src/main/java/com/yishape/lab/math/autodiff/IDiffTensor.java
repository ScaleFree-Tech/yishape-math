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

    @Override IDiffTensor erf();
    @Override IDiffTensor round();
    @Override IDiffTensor floor();
    @Override IDiffTensor ceil();
    @Override IDiffTensor sign();
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

    /** Numerically stable log(sum(exp(x), dim)). */
    IDiffTensor logSumExp(int dim, boolean keepdim);

    @Override IDiffTensor softmax(int dim);
    @Override IDiffTensor logSoftmax(int dim);
    IDiffTensor softmaxCrossEntropy(IDoubleTensor labels, int dim);

    /**
     * Fused softmax + cross-entropy with sparse integer labels (class indices).
     * Single fused node — avoids the 6-node logSumExp→sub→gather→sum→div→neg chain.
     * @param labels integer class indices, one per sample, each in [0, classSize)
     * @param dim    the class dimension
     * @return scalar loss = mean(-log(softmax[target]))
     */
    IDiffTensor softmaxCrossEntropySparse(int[] labels, int dim);

    // ==================== Loss Functions ====================

    /** Mean Squared Error loss: mean((this - target)^2). Returns scalar tensor [1]. */
    default IDiffTensor mseLoss(IDoubleTensor target) {
        IDiffTensor diff = this.sub(target).square();
        return diff.sum().div(diff.totalSize());
    }

    /** L1 Loss: mean(|this - target|). Returns scalar tensor [1]. */
    default IDiffTensor l1Loss(IDoubleTensor target) {
        IDiffTensor diff = this.sub(target).abs();
        return diff.sum().div(diff.totalSize());
    }

    /**
     * Smooth L1 / Huber loss: 0.5*(diff^2)/beta if |diff|<=beta, |diff|-0.5*beta otherwise.
     * Returns scalar tensor [1].
     */
    IDiffTensor smoothL1Loss(IDiffTensor target, double beta);

    /**
     * Binary Cross Entropy loss (fused single-node).
     * Computes: -mean(y*log(clamp(p,eps,1-eps)) + (1-y)*log(clamp(1-p,eps,1-eps))).
     * Prediction p must be in (0,1) — applies clamp internally for numerical stability.
     * @param target binary labels, same shape as input
     * @return scalar tensor [1]
     */
    IDiffTensor bceLoss(IDiffTensor target);

    /**
     * Focal Loss (Lin et al., RetinaNet) — fused single-node.
     * FL(p_t) = mean(α_t * (1 - p_t)^γ * (-log(p_t)))
     * where p_t = p if y=1 else 1-p, α_t = α if y=1 else 1-α.
     * Prediction p must be in (0,1) — applies clamp internally.
     * @param target binary labels, same shape as input
     * @param alpha class weighting (α for foreground)
     * @param gamma focusing parameter (γ ≥ 0)
     * @return scalar tensor [1]
     */
    IDiffTensor focalLoss(IDiffTensor target, double alpha, double gamma);

    /**
     * Dice Loss for segmentation — fused single-node.
     * Computes: 1 - (2*sum(p*t) + smooth) / (sum(p) + sum(t) + smooth)
     * @param target binary labels, same shape as input
     * @param smooth Laplace smoothing constant (prevents division by zero)
     * @return scalar tensor [1]
     */
    IDiffTensor diceLoss(IDiffTensor target, double smooth);

    /**
     * Binary Cross Entropy with Logits Loss (numerically stable).
     * Computes: mean(log(1+exp(x)) - x*target), fused for stability.
     * @param target binary labels, same shape as input
     * @return scalar tensor [1]
     */
    default IDiffTensor bceWithLogitsLoss(IDiffTensor target) {
        // Numerically stable: max(x,0) - x*target + log(1+exp(-|x|))
        IDiffTensor posPart = this.relu();
        IDiffTensor negAbs = this.abs().neg().exp().add(1.0).log();
        IDiffTensor loss = posPart.sub(this.mul(target)).add(negAbs);
        return loss.sum().div(loss.totalSize());
    }

    /**
     * Negative Log-Likelihood loss (input = log-probabilities).
     * Computes: -mean(gather(input, dim=classDim, target)).
     * @param target   class indices (long/int tensor)
     * @param classDim dimension containing class scores
     * @return scalar tensor [1]
     */
    IDiffTensor nllLoss(IDiffTensor target, int classDim);

    /**
     * KL Divergence loss: sum(target * (log(target) - input_log_prob)).
     * Input should be log-probabilities (logSoftmax output).
     * @param target probability distribution, same shape as input
     * @return scalar tensor [1]
     */
    default IDiffTensor klDivLoss(IDiffTensor target) {
        // target * (log(target) - this); target is prob dist, this is log-prob
        IDiffTensor logTarget = target.add(1e-12).log(); // epsilon for numerical stability
        return target.mul(logTarget.sub(this)).sum();
    }

    // ==================== Pooling ====================

    /**
     * 2D Max Pooling. Input shape: [N, C, H, W].
     * @param kH      kernel height
     * @param kW      kernel width
     * @param stride  stride (defaults to kernel size if 0)
     * @param padding zero-padding
     * @return pooled output [N, C, outH, outW]
     */
    IDiffTensor maxPool2d(int kH, int kW, int stride, int padding);

    /**
     * 2D Average Pooling. Input shape: [N, C, H, W].
     * @param kH      kernel height
     * @param kW      kernel width
     * @param stride  stride (defaults to kernel size if 0)
     * @param padding zero-padding
     * @return pooled output [N, C, outH, outW]
     */
    IDiffTensor avgPool2d(int kH, int kW, int stride, int padding);

    /**
     * 2D Adaptive Average Pooling. Input shape: [N, C, H, W] or [N, C, H, W, ...].
     * Output spatial dims are exactly (outH, outW) regardless of input size.
     * @param outH target output height
     * @param outW target output width
     * @return pooled output [N, C, outH, outW]
     */
    IDiffTensor adaptiveAvgPool2d(int outH, int outW);

    // ==================== Similarity & Distance ====================

    /**
     * Cosine similarity along a dimension.
     * Returns per-sample cosine similarity: dot(x,y,dim) / (||x|| * ||y|| + eps).
     * @param other tensor of same shape
     * @param dim   dimension along which to compute similarity
     * @param eps   small constant for numerical stability
     * @return cosine similarity, shape = input shape with dim reduced
     */
    default IDiffTensor cosineSimilarity(IDiffTensor other, int dim, double eps) {
        IDiffTensor dot = this.mul(other).sum(dim, true);
        IDiffTensor normX = this.square().sum(dim, true).add(eps).sqrt();
        IDiffTensor normY = other.square().sum(dim, true).add(eps).sqrt();
        return dot.div(normX).div(normY);
    }

    // ==================== Matrix Ops ====================

    /**
     * Differentiable one-hot encoding. Input values are class indices (floored to int).
     * Returns a tensor with an extra dimension of size numClasses.
     * @param numClasses total number of classes
     * @return one-hot tensor: shape = [*this.shape, numClasses]
     */
    IDiffTensor oneHot(int numClasses);

    /**
     * Fused addmm: alpha * this + beta * (this @ mat1).
     * D13: Note this differs from PyTorch's addmm(bias, mat1, mat2, beta, alpha)
     * which computes beta * bias + alpha * (mat1 @ mat2). The current signature
     * uses `this` as both the bias and the left mmul operand for simplicity.
     * @param mat1  right matrix
     * @param alpha scalar multiplier for this (bias)
     * @param beta  scalar multiplier for this @ mat1
     * @return result = alpha*this + beta*(this @ mat1)
     */
    default IDiffTensor addmm(IDiffTensor mat1, double alpha, double beta) {
        return this.mul(alpha).add(this.mmul(mat1).mul(beta));
    }

    /**
     * Fused baddbmm: alpha * this + beta * (batch1 @ batch2).
     * @param batch1 left batched matrix
     * @param alpha  scalar multiplier for this
     * @param beta   scalar multiplier for batch1 @ batch2
     * @return result = alpha*this + beta*(this @ batch1) via bmm
     */
    default IDiffTensor baddbmm(IDiffTensor batch1, double alpha, double beta) {
        return this.mul(alpha).add(this.bmm(batch1).mul(beta));
    }

    // ==================== Normalization — Tier 3 ====================

    /**
     * Instance Normalization. Normalizes over spatial dims (H*W) per sample per channel.
     * Input shape: [N, C, H, W] or [N, C, L].
     * @param gamma scale parameter [C]
     * @param beta  shift parameter [C] (can be null)
     * @param eps   small constant for numerical stability
     * @return normalized output with the same shape as input
     */
    IDiffTensor instanceNorm(IDiffTensor gamma, IDiffTensor beta, double eps);

    // ==================== Matrix Ops — Tier 3 ====================

    /** Frobenius norm: sqrt(sum(x^2)). For full Frobenius norm use sum() of square. */
    default IDiffTensor frobeniusNorm(int... dims) {
        IDiffTensor sq = this.square();
        boolean keepdim = true;
        return sq.sum(dims[0], keepdim).sqrt();
    }

    /**
     * Create a diagonal embedding: expand input into a matrix with input along diagonal.
     * Input shape: [..., D]. Output shape: [..., D, D] or specified dims.
     * @param offset diagonal offset (0 = main diagonal)
     * @param dim1   first dimension for the output matrix
     * @param dim2   second dimension for the output matrix
     * @return tensor with input placed along diagonal
     */
    IDiffTensor diagEmbed(int offset, int dim1, int dim2);

    // ==================== Matrix Decomposition Ops ====================

    /**
     * Log absolute determinant of a square matrix (last two dims).
     * log(|det(A)|). Numerically stable via LU decomposition.
     * Input shape: [N, N] (2D square matrix).
     * @return scalar tensor [1] containing log(|det|)
     */
    IDiffTensor logDet();

    /**
     * Sign and log-absolute-determinant of a square matrix.
     * Returns [sign, log|det|] where sign is the determinant sign (±1).
     * Input shape: [N, N] (2D square matrix). The sign entry does NOT receive gradient.
     * @return two-element array: [sign_tensor, logDet_tensor] both shape [1]
     */
    IDiffTensor[] slogDet();

    /**
     * Nuclear norm (sum of singular values, aka trace norm / Schatten-1 norm).
     * Computed via SVD: sum(s_i) where s_i are singular values of the input matrix.
     * Input shape: [M, N] (2D matrix).
     * @return scalar tensor [1] containing sum of singular values
     */
    IDiffTensor nuclearNorm();

    /**
     * CTC (Connectionist Temporal Classification) loss.
     * Input (this): log-probabilities of shape [T, N, C] where
     * T = time steps, N = batch size, C = number of classes.
     * Uses GPU→HPC→SIMD→SISD fallback chain.
     * @param targets       target label sequences [N, S] (padded with 0)
     * @param inputLengths  lengths of each input sequence [N] (max T)
     * @param targetLengths lengths of each target sequence [N] (max S)
     * @return scalar tensor [1] containing the CTC loss
     */
    IDiffTensor ctcLoss(IDiffTensor targets, IDiffTensor inputLengths, IDiffTensor targetLengths);

    /**
     * Cross product of 3D vectors: a × b.
     * Input tensors must have last dimension = 3. Supports arbitrary batch dims via broadcast.
     * @param other tensor with last dim = 3
     * @return cross product with same shape as broadcast(this, other)
     */
    IDiffTensor cross(IDiffTensor other);

    /**
     * Differentiable image sampling (spatial grid sampling).
     * Input shape: [N, C, H, W]. Grid shape: [N, outH, outW, 2] with normalized coordinates in [-1, 1].
     * @param grid         sampling grid [N, outH, outW, 2]
     * @param mode         "bilinear" or "nearest"
     * @param paddingMode  "zeros", "border", or "reflection"
     * @return sampled output [N, C, outH, outW]
     */
    IDiffTensor gridSample(IDiffTensor grid, String mode, String paddingMode);

    /**
     * Trapezoidal selective scan (Mamba SSM core).
     * Discretizes the continuous SSM using the trapezoidal rule.
     * Input (this): U — input sequence [B, L, D].
     * @param delta  time delta [B, L, D] or [B, 1, D]
     * @param A      state matrix diagonal [D] or [B, D]
     * @param B      input-to-state [B, L, D]
     * @param C      state-to-output [B, L, D]
     * @param D      skip connection [D] or scalar (broadcast)
     * @return Y output [B, L, D]
     */
    IDiffTensor trapezoidalScan(IDiffTensor delta, IDiffTensor A, IDiffTensor B, IDiffTensor C, IDiffTensor D);

    // ==================== Dropout — Tier 3 ====================

    /**
     * 2D Dropout: randomly zeroes entire channels with probability p.
     * During training, each channel is dropped independently.
     * Input shape: [N, C, H, W].
     */
    IDiffTensor dropout2d(double p);

    // ==================== Convolution — Tier 3 ====================

    /**
     * Depthwise 1D Convolution: each input channel is convolved with its own kernel.
     * Input shape: [N, C, L]. Weight shape: [C, kernelSize].
     * @param weight   depthwise kernel [C, kernelSize]
     * @param stride   convolution stride
     * @param padding  zero-padding
     * @return output [N, C, outL] where outL = (L + 2*padding - kernelSize) / stride + 1
     */
    IDiffTensor depthwiseConv1d(IDiffTensor weight, int stride, int padding);

    // ==================== Interpolation — Tier 3 ====================

    /**
     * 2D interpolation (upsampling). Input shape: [N, C, H, W].
     * @param scaleFactor scaling factor (>1 for upsample, <1 for downsample)
     * @param mode        "bilinear" or "nearest"
     * @return interpolated output [N, C, floor(H*scale), floor(W*scale)]
     */
    IDiffTensor interpolate(double scaleFactor, String mode);

    /**
     * 2D Convolution (im2col + gemm).
     * Input shape: [N, C, H, W]. Weight shape: [outC, C, kH, kW]. Bias shape: [outC].
     * Output shape: [N, outC, outH, outW] where
     * outH = (H + 2*padding - dilation*(kH-1) - 1) / stride + 1,
     * outW = (W + 2*padding - dilation*(kW-1) - 1) / stride + 1.
     * Uses GPU→HPC→SIMD→SISD fallback for matrix operations.
     *
     * @param weight   convolution kernel [outC, inC, kH, kW]
     * @param bias     bias term [outC] (can be null)
     * @param stride   convolution stride
     * @param padding  zero-padding
     * @param dilation kernel dilation (1 = no dilation)
     * @return convolution output [N, outC, outH, outW]
     */
    IDiffTensor conv2d(IDiffTensor weight, IDiffTensor bias,
                       int stride, int padding, int dilation);

    /**
     * Scaled Dot-Product Attention: softmax(Q @ K^T / sqrt(d_k) + mask) @ V.
     * Q shape: [batch, seqQ, d_k], K shape: [batch, seqK, d_k], V shape: [batch, seqK, d_v].
     * mask shape: [batch, 1, seqQ, seqK] (broadcastable, can be null).
     * For multi-head attention, reshape heads into batch dimension before calling.
     *
     * @param key     key tensor [batch, seqK, d_k]
     * @param value   value tensor [batch, seqK, d_v]
     * @param mask    additive mask [batch, 1, seqQ, seqK] or null
     * @param dropout dropout probability (0 = no dropout)
     * @return attention output [batch, seqQ, d_v]
     */
    IDiffTensor scaledDotProductAttention(IDiffTensor key, IDiffTensor vTensor,
                                           IDiffTensor mask, double dropout);

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

    /**
     * RMS Normalization (LLaMA-style). Normalizes over the last dimension.
     * y = x / sqrt(mean(x^2) + eps) * gamma
     * @param gamma scale parameter (size = last dimension)
     * @param eps   small constant for numerical stability
     * @return normalized output with the same shape as input
     */
    IDiffTensor rmsNorm(IDiffTensor gamma, double eps);

    /**
     * Rotary Position Embedding (RoPE). Applies rotation to pairs of elements.
     * @param dim    half-dimension for frequency computation (typically headDim)
     * @param maxLen maximum sequence length (for pre-computing frequencies, unused if dynamic)
     * @param base   base for frequency computation (default 10000.0)
     * @return tensor with RoPE applied (same shape)
     */
    IDiffTensor rope(int dim, int maxLen, double base);

    /**
     * Group Normalization. Divides channels into groups, normalizes each group independently.
     * Input shape: [N, C, H, W] or [N, C, L]. Channels are split into numGroups groups.
     * gamma/beta shape: [C] (per-channel scale/shift broadcast across groups).
     * @param numGroups number of groups to split channels into (must divide C)
     * @param gamma     scale parameter [C]
     * @param beta      shift parameter [C] (can be null)
     * @param eps       small constant for numerical stability
     * @return normalized output with the same shape as input
     */
    IDiffTensor groupNorm(int numGroups, IDiffTensor gamma, IDiffTensor beta, double eps);

    /**
     * Reverse the order of elements along specified dimensions.
     * @param dims dimensions to flip (varargs)
     * @return flipped tensor (same shape)
     */
    IDiffTensor flip(int... dims);

    /**
     * Roll (circular shift) elements along specified dimensions.
     * @param shifts amount to shift by (positive = right/down, negative = left/up)
     * @param dims   dimensions to roll along (same length as shifts)
     * @return rolled tensor (same shape)
     */
    IDiffTensor roll(int[] shifts, int[] dims);

    /**
     * Repeat each element along a dimension the given number of times.
     * Equivalent to torch.repeat_interleave.
     * @param repeats number of repetitions per element
     * @param dim     dimension along which to repeat
     * @return tensor with dim expanded by factor repeats
     */
    IDiffTensor repeatInterleave(int repeats, int dim);

    /**
     * LSTM cell single timestep. Decomposes to existing differentiable ops.
     * Gates: i = sigmoid(Wi·x + bi + Whi·h + bhi)
     *        f = sigmoid(Wf·x + bf + Whf·h + bhf)
     *        o = sigmoid(Wo·x + bo + Who·h + bho)
     *        g = tanh(Wg·x + bg + Whg·h + bhg)
     * Cell: c = f * cPrev + i * g
     * Hidden: h = o * tanh(c)
     * @param x       input [batch, inputSize]
     * @param hPrev   previous hidden state [batch, hiddenSize]
     * @param cPrev   previous cell state [batch, hiddenSize]
     * @param wInput  input weights [4*hiddenSize, inputSize] (stacked i,f,o,g)
     * @param wHidden hidden weights [4*hiddenSize, hiddenSize]
     * @param bias    bias [4*hiddenSize] (optional, can be null)
     * @return [h, c] as IDiffTensor[2]
     */
    IDiffTensor[] lstmCell(IDiffTensor x, IDiffTensor hPrev, IDiffTensor cPrev,
                           IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias);

    /**
     * GRU cell single timestep. Decomposes to existing differentiable ops.
     * Gates: z = sigmoid(Wz·x + bz + Whz·h + bhz)
     *        r = sigmoid(Wr·x + br + Whr·h + bhr)
     *        n = tanh(Wn·x + bn + r * (Whn·h + bhn))
     * Hidden: h = (1 - z) * n + z * hPrev
     * @param x       input [batch, inputSize]
     * @param hPrev   previous hidden state [batch, hiddenSize]
     * @param wInput  input weights [3*hiddenSize, inputSize] (stacked z,r,n)
     * @param wHidden hidden weights [3*hiddenSize, hiddenSize]
     * @param bias    bias [3*hiddenSize] (optional, can be null)
     * @return h (new hidden state)
     */
    IDiffTensor gruCell(IDiffTensor x, IDiffTensor hPrev,
                        IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias);

    /**
     * Differentiable embedding lookup: gathers rows from embedding table.
     * Equivalent to gather(0, indices) but with optimized GPU/HPC support.
     * @param indices integer indices tensor (any shape)
     * @return gathered embeddings [*indices.shape, embeddingDim]
     */
    IDiffTensor embedding(IDiffTensor indices);

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
    @Override IDiffTensor tril(int diagonal);
    /** Upper triangular view (mirrors tril). Elements below diagonal are zeroed. */
    IDiffTensor triu(int diagonal);
    /** Extract the diagonal of a matrix (last two dims). Returns shape [min(M,N)]. */
    IDiffTensor diag();
    /** Extract a diagonal with offset from dim1/dim2. */
    IDiffTensor diagonal(int offset, int dim1, int dim2);
    /** Trace (sum of diagonal elements). Equivalent to diag().sum(). */
    IDiffTensor trace();
    @Override IDiffTensor unfold(int dim, int size, int stride, int dilation);
    @Override IDiffTensor nonzero();
    @Override IDiffTensor maskedSelect(IDoubleTensor mask);
    @Override IDiffTensor maskedFill(IDoubleTensor mask, double value);
    @Override IDiffTensor cat(int dim, IDoubleTensor... others);
    @Override IDiffTensor stack(int dim, IDoubleTensor... others);

    /** Split tensor into sections of size splitSize along dim. Last section may be smaller. */
    IDiffTensor[] split(int splitSize, int dim);
    /** Split tensor into sections of given sizes along dim. Sum of sizes must equal dim(dim). */
    IDiffTensor[] split(int[] splitSizes, int dim);
    /** Split tensor into chunks equally along dim (last may be smaller). */
    IDiffTensor[] chunk(int chunks, int dim);
    /** Unbind tensor along dim (remove dim, return array of slices). */
    IDiffTensor[] unbind(int dim);
    @Override IDiffTensor normalize(double p, int dim);

    @Override IDiffTensor add_(IDoubleTensor other);
    @Override IDiffTensor sub_(IDoubleTensor other);
    @Override IDiffTensor mul_(IDoubleTensor other);
    @Override IDiffTensor div_(IDoubleTensor other);
    @Override IDiffTensor fill_(double value);
    @Override IDiffTensor copy_(IDoubleTensor src);

    // ==================== 内存 ====================

    /**
     * Estimated activation memory in bytes for this node's value.
     * Useful for memory-budget-aware checkpointing decisions.
     * Default: {@code totalSize() * 8} (double-precision).
     */
    default long activationBytes() {
        return totalSize() * 8L;
    }

    // ==================== 梯度裁剪 ====================

    /**
     * Clip this tensor's gradient by global L2 norm (in-place).
     * If {@code ||grad||_2 > maxNorm}, scales grad to have norm = maxNorm.
     * No-op if grad is null or requiresGrad is false.
     * Uses GPU→HPC→SIMD→SISD fallback chain for norm computation.
     */
    void clipGradNorm(double maxNorm);

    /**
     * Clip this tensor's gradient by value (in-place, element-wise).
     * Each element is clamped to {@code [-maxValue, maxValue]}.
     * No-op if grad is null or requiresGrad is false.
     * Uses GPU→HPC→SIMD→SISD fallback chain for min/max operations.
     */
    void clipGradValue(double maxValue);

    // ==================== 梯度方法 ====================

    /** 执行反向传播 */
    void backward();

    /**
     * 执行反向传播，可选择是否保留计算图。
     *
     * @param retainGraph if true, backward functions and graph edges are preserved,
     *                    allowing subsequent {@code backward()} calls for higher-order AD.
     *                    if false (default), graph edges are released for GC.
     */
    default void backward(boolean retainGraph) {
        if (!retainGraph) { backward(); return; }
        throw new UnsupportedOperationException(
            "backward(retainGraph=true) not supported on " + this.getClass().getSimpleName());
    }

    /** 使用指定梯度执行反向传播 */
    void backward(IDoubleTensor gradient);

    /**
     * 使用指定梯度执行反向传播，可选择是否保留计算图。
     */
    default void backward(IDoubleTensor gradient, boolean retainGraph) {
        if (!retainGraph) { backward(gradient); return; }
        throw new UnsupportedOperationException(
            "backward(gradient, retainGraph=true) not supported on " + this.getClass().getSimpleName());
    }

    /** 清零梯度 */
    void zeroGradient();

    /** 展平梯度（供优化器使用） */
    IDiffVector flattenGrad();

    /**
     * Flatten this tensor's value to a 1D vector. / 展平值（供优化器使用）。
     *
     * <p><b>⚠️ This is a VIEW, not a copy.</b> The returned vector shares
     * backing storage with this tensor. Mutations propagate bidirectionally.
     * For a detached copy, call {@code detach().flattenValue()}.</p>
     *
     * <p>When this tensor is multi-dimensional, the returned vector's
     * {@code slice()} method may treat flat indices as dim-0 indices.
     * {@code RereDiffTensor} auto-handles this; other implementations may not.</p>
     */
    IDiffVector flattenValue();

    /**
     * Returns a new tensor that shares data with this one but is detached from the
     * computation graph. Equivalent to PyTorch's {@code tensor.detach()}.
     * The result is an {@code IDiffTensor} with {@code requiresGrad=false},
     * meaning {@code backward()} will not propagate gradients to this tensor's ancestors.
     *
     * @return a detached copy sharing the same data
     */
    IDiffTensor detach();

    /** 是否追踪梯度 */
    boolean requiresGrad();

    /** 设置是否追踪梯度 */
    IDiffTensor setRequiresGrad(boolean requiresGrad);

    /** 当前累积梯度 */
    IDoubleTensor grad();

    // ==================== 工厂 ====================

    /**
     * 从 IDiffVector + shape 创建可微张量。
     *
     * <p><b>⚠️ Downstream trap:</b> When {@code shape} has rank > 1, the resulting
     * multi-dim tensor becomes the backing storage for any {@code RereDiffVector}
     * produced by {@link IDiffTensor#flattenValue()}. Subsequent calls to
     * {@link IDiffVector#slice(int, int)} on that flattened vector will call
     * {@code tensor.slice(0, start, end)}, which interprets {@code start}/{@code end}
     * as dim-0 indices rather than flat indices — causing
     * {@link IndexOutOfBoundsException} when {@code end > shape[0]}.</p>
     *
     * <p>Callers that need vector slicing on the result should keep the tensor
     * 1D (shape {@code [N]}) and use {@link IDiffTensor#slice(int, int, int)}
     * with explicit dimension arguments instead.</p>
     */
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
        // C2: non-RereDiffVector types (e.g. TangentDiffVector) cannot be safely unwrapped
        // to a properly-connected tensor. Throw rather than creating a disconnected leaf.
        throw new UnsupportedOperationException(
            "fromDiffVector does not support " + vec.getClass().getSimpleName()
            + " — only RereDiffVector is supported. Use AD.leafTensor() for detached tensors.");
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
