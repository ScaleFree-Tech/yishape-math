package com.yishape.lab.math.linalg.tensor;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;

import java.util.List;

/**
 * 维度感知的多维张量接口.
 * <p>
 * 扩展 ITensor 添加 stride 视图系统、维度归约（keepdim）、广播、
 * 高级索引和线性代数操作.
 * <p>
 * API 风格与 yishape-math 现有 IVector/IMatrix 一致.
 */
public interface IDoubleTensor extends ITensor {

    // ==================== Shape 查询 ====================

    /** 返回 strides（视图时可能与 C-order 不同） */
    int[] strides();

    /** 返回指定维度的 stride */
    int stride(int axis);

    /** 数据偏移（视图切片时 &gt;0） */
    int offset();

    /** 是否连续内存（strides 等于 C-order） */
    boolean isContiguous();

    /** 标量张量取值（totalSize==1 时可用） */
    double item();


    // ==================== 视图操作（零拷贝） ====================

    /** 维度置换 */
    IDoubleTensor permute(int... dims);

    /** 2D 转置 */
    IDoubleTensor transpose(int dim0, int dim1);

    /** 全转置（2D 快捷方式） */
    IDoubleTensor transpose();

    /** 去除尺寸为 1 的维度 */
    IDoubleTensor squeeze(int... dims);

    /** 在指定位置插入尺寸为 1 的维度 */
    IDoubleTensor unsqueeze(int dim);

    /** 沿 dim 切片 [start, end) */
    IDoubleTensor slice(int dim, long start, long end);

    /** 沿 dim 取长度为 length 的子区域 */
    IDoubleTensor narrow(int dim, long start, long length);

    /** 沿 dim 选取单个索引（返回低一阶的张量） */
    IDoubleTensor select(int dim, long index);

    /** 展平从 startDim 到 endDim 的范围 */
    IDoubleTensor flatten(int startDim, int endDim);

    /** 扩维到目标 shape（沿 dim=1 的维度广播） */
    IDoubleTensor expand(int... shape);

    /** 强制转换为连续内存（视图 materialize） */
    IDoubleTensor contiguous();

    /** 重复平铺 */
    IDoubleTensor tile(int... repeats);

    /** 广播到目标 shape */
    IDoubleTensor broadcastTo(int... shape);


    // ==================== 逐元素运算 ====================

    IDoubleTensor add(IDoubleTensor other);
    IDoubleTensor sub(IDoubleTensor other);
    IDoubleTensor mul(IDoubleTensor other);
    IDoubleTensor div(IDoubleTensor other);
    IDoubleTensor add(double scalar);
    IDoubleTensor sub(double scalar);
    IDoubleTensor mul(double scalar);
    IDoubleTensor div(double scalar);
    IDoubleTensor neg();
    IDoubleTensor abs();
    IDoubleTensor sqrt();
    IDoubleTensor exp();
    IDoubleTensor log();
    IDoubleTensor sin();
    IDoubleTensor cos();
    IDoubleTensor tan();
    IDoubleTensor sigmoid();
    IDoubleTensor relu();
    IDoubleTensor tanh();
    IDoubleTensor gelu();
    IDoubleTensor leakyRelu(double alpha);
    IDoubleTensor elu(double alpha);
    IDoubleTensor selu();
    IDoubleTensor silu();
    IDoubleTensor mish();
    IDoubleTensor softplus(double beta);
    IDoubleTensor hardtanh(double minVal, double maxVal);
    IDoubleTensor square();
    IDoubleTensor pow(double n);
    IDoubleTensor clamp(double min, double max);


    // ==================== 归约操作 ====================

    /** 沿指定维度求和 */
    IDoubleTensor sum(int dim, boolean keepdim);

    /** 沿指定维度求均值 */
    IDoubleTensor mean(int dim, boolean keepdim);

    /** 沿指定维度求最大值 */
    IDoubleTensor max(int dim, boolean keepdim);

    /** 沿指定维度求最小值 */
    IDoubleTensor min(int dim, boolean keepdim);

    /** 沿指定维度求积 */
    IDoubleTensor prod(int dim, boolean keepdim);

    /** 全量归约 */
    double sumAll();
    double meanAll();
    double maxAll();
    double minAll();
    double prodAll();

    /** 沿指定维度累积和 */
    IDoubleTensor cumsum(int dim);

    /** 沿指定维度累积积 */
    IDoubleTensor cumprod(int dim);

    /** 沿指定维度返回最大值索引 */
    IDoubleTensor argmax(int dim);

    /** 沿指定维度返回最小值索引 */
    IDoubleTensor argmin(int dim);

    /** 沿指定维度的标准差 */
    IDoubleTensor std(int dim, boolean keepdim);

    /** 沿指定维度的方差 */
    IDoubleTensor var(int dim, boolean keepdim);


    // ==================== 高级操作 ====================

    /** 2D 矩阵乘法 */
    IDoubleTensor mmul(IDoubleTensor other);

    /** 批量矩阵乘法 (B,N,M) @ (B,M,P) → (B,N,P) */
    IDoubleTensor bmm(IDoubleTensor other);

    /** Einstein 求和约定 */
    IDoubleTensor einsum(String subscript, IDoubleTensor... others);

    /** Softmax（沿指定维度） */
    IDoubleTensor softmax(int dim);

    /** LogSoftmax（沿指定维度） */
    IDoubleTensor logSoftmax(int dim);

    /** 沿 dim 根据 index 收集元素 */
    IDoubleTensor gather(int dim, IDoubleTensor index);

    /** 沿 dim 按索引选择整片: 等效于 gather(dim, index.expand(...)) */
    IDoubleTensor indexSelect(int dim, IDoubleTensor index);

    /** 沿 dim 返回排序索引 (非可微) */
    IDoubleTensor argsort(int dim, boolean descending);

    /** 沿 dim 根据 index 散布 source 的值 */
    IDoubleTensor scatter(int dim, IDoubleTensor index, IDoubleTensor source);

    /** 沿 dim 根据 index 进行加法散布 */
    IDoubleTensor scatterAdd(int dim, IDoubleTensor index, IDoubleTensor source);

    /** 条件选择: result[i] = condition[i] ? this[i] : other[i] */
    IDoubleTensor where(IDoubleTensor condition, IDoubleTensor other);

    /** 沿 dim 取前 k 个最大值 */
    IDoubleTensor topk(int k, int dim, boolean largest);

    /** Padding */
    IDoubleTensor pad(int[][] padding, String mode, double value);

    /**
     * Lower triangular mask: sets elements above diagonal+k to zero.
     * Operates on the last two dimensions (matrix rows, cols).
     * @param diagonal k: 0 = main diagonal, k > 0 includes k super-diagonals,
     *                  k < 0 excludes |k| sub-diagonals
     */
    IDoubleTensor tril(int diagonal);

    /** 滑动窗口展开（im2col 等价） */
    IDoubleTensor unfold(int dim, int size, int stride, int dilation);

    /** 非零元素索引 */
    IDoubleTensor nonzero();

    /** 掩码选择 */
    IDoubleTensor maskedSelect(IDoubleTensor mask);

    /** 掩码填充 */
    IDoubleTensor maskedFill(IDoubleTensor mask, double value);

    /** 沿 dim 拼接 */
    IDoubleTensor cat(int dim, IDoubleTensor... others);

    /** 沿新 dim 堆叠 */
    IDoubleTensor stack(int dim, IDoubleTensor... others);

    /** 沿 dim 拆分 */
    List<IDoubleTensor> unstack(int dim);

    /** p-范数归一化 */
    IDoubleTensor normalize(double p, int dim);


    // ==================== 就地操作 ====================

    IDoubleTensor add_(IDoubleTensor other);
    IDoubleTensor sub_(IDoubleTensor other);
    IDoubleTensor mul_(IDoubleTensor other);
    IDoubleTensor div_(IDoubleTensor other);
    IDoubleTensor fill_(double value);
    IDoubleTensor copy_(IDoubleTensor src);


    // ==================== 转换桥接 ====================

    /** 展平为 IDoubleVector */
    IDoubleVector toVector();

    /** 展平为 IDoubleVector（强制拷贝） */
    IDoubleVector toVectorCopy();

    /** 转为 IMatrix（仅 2D） */
    IMatrix toMatrix();

    /** 深拷贝 */
    IDoubleTensor clone();
}
