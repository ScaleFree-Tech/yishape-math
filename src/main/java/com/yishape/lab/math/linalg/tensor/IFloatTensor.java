package com.yishape.lab.math.linalg.tensor;

import com.yishape.lab.math.linalg.IFloatVector;
import com.yishape.lab.math.linalg.IFloatMatrix;

import java.util.Arrays;
import java.util.List;

/**
 * 单精度浮点张量接口.
 * <p>
 * 镜像 {@link IDoubleTensor} 的全部能力，但使用 float 精度，
 * 专为推理路径设计，节省一半内存带宽。
 * <p>
 * 不继承 {@link ITensor}（ITensor 的工厂方法和 get/set 为 double 语义）。
 */
public interface IFloatTensor {

    // ==================== 工厂方法 ====================

    static IFloatTensor tensor(float[] data, int... shape) {
        return new RereFloatTensor(data.clone(), shape);
    }

    /** Convenience overload accepting long[] shape (common in ONNX). */
    static IFloatTensor tensor(float[] data, long[] shape) {
        int[] s = new int[shape.length];
        for (int i = 0; i < shape.length; i++) s[i] = (int) shape[i];
        return new RereFloatTensor(data.clone(), s);
    }

    /**
     * Zero-copy factory: wraps the given array directly without cloning.
     * Caller warrants that the array is not shared/aliased.
     */
    static IFloatTensor wrap(float[] data, int... shape) {
        return new RereFloatTensor(data, shape);
    }

    /** Zero-copy factory with long[] shape. */
    static IFloatTensor wrap(float[] data, long[] shape) {
        int[] s = new int[shape.length];
        for (int i = 0; i < shape.length; i++) s[i] = (int) shape[i];
        return new RereFloatTensor(data, s);
    }

    static IFloatTensor tensor(float[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        float[] flat = new float[rows * cols];
        int idx = 0;
        for (float[] row : data) {
            for (int j = 0; j < cols; j++) {
                flat[idx++] = row[j];
            }
        }
        return new RereFloatTensor(flat, rows, cols);
    }

    static IFloatTensor tensor(float[][][] data) {
        int dim0 = data.length;
        int dim1 = data[0].length;
        int dim2 = data[0][0].length;
        float[] flat = new float[dim0 * dim1 * dim2];
        int idx = 0;
        for (float[][] d1 : data) {
            for (float[] d2 : d1) {
                for (float v : d2) {
                    flat[idx++] = v;
                }
            }
        }
        return new RereFloatTensor(flat, dim0, dim1, dim2);
    }

    static IFloatTensor ones(int... shape) {
        long size = 1;
        for (int d : shape) size *= d;
        float[] data = new float[(int) size];
        Arrays.fill(data, 1.0f);
        return new RereFloatTensor(data, shape);
    }

    static IFloatTensor zeros(int... shape) {
        long size = 1;
        for (int d : shape) size *= d;
        return new RereFloatTensor(new float[(int) size], shape);
    }

    static IFloatTensor rand(int... shape) {
        long size = 1;
        for (int d : shape) size *= d;
        float[] data = new float[(int) size];
        for (int i = 0; i < size; i++) {
            data[i] = (float) Math.random();
        }
        return new RereFloatTensor(data, shape);
    }

    static IFloatTensor randn(int... shape) {
        long size = 1;
        for (int d : shape) size *= d;
        float[] data = new float[(int) size];
        for (int i = 0; i < size; i++) {
            data[i] = (float) nextGaussian();
        }
        return new RereFloatTensor(data, shape);
    }

    static IFloatTensor eye(int n, int... extraDims) {
        int[] shape = new int[extraDims.length + 2];
        shape[0] = n;
        shape[1] = n;
        System.arraycopy(extraDims, 0, shape, 2, extraDims.length);
        long size = 1;
        for (int d : shape) size *= d;
        float[] data = new float[(int) size];
        int[] strides = TensorShape.computeCStrides(shape);
        int diagStride = strides[0] + strides[1];
        long batchCount = extraDims.length == 0 ? 1 :
            size / (n * n);
        for (long b = 0; b < batchCount; b++) {
            for (int i = 0; i < n; i++) {
                data[(int) (i * diagStride + b)] = 1.0f;
            }
        }
        return new RereFloatTensor(data, shape);
    }

    static IFloatTensor scalar(float value) {
        return new RereFloatTensor(new float[]{value}, 1);
    }

    static IFloatTensor empty(int... shape) {
        long size = 1;
        for (int d : shape) size *= d;
        return new RereFloatTensor(new float[(int) size], shape);
    }

    static IFloatTensor fromVector(IFloatVector vec, int... shape) {
        return new RereFloatTensor(vec.toFloatArray(), shape);
    }

    static IFloatTensor fromStrided(float[] data, int offset, int[] shape, int[] strides) {
        return new RereFloatTensor(data, offset, new TensorShape(shape, strides));
    }

    static IFloatTensor full(int[] shape, float value) {
        long size = 1;
        for (int d : shape) size *= d;
        float[] data = new float[(int) size];
        Arrays.fill(data, value);
        return new RereFloatTensor(data, shape);
    }

    static IFloatTensor arange(float start, float end, float step) {
        int n = (int) Math.ceil((end - start) / step);
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = start + i * step;
        }
        return new RereFloatTensor(data, n);
    }

    private static double nextGaussian() {
        double u1 = Math.random();
        double u2 = Math.random();
        return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    }


    // ==================== Shape 查询 ====================

    int rank();
    int[] shape();
    int dim(int axis);
    long totalSize();

    /** 返回 strides（视图时可能与 C-order 不同） */
    int[] strides();

    /** 返回指定维度的 stride */
    int stride(int axis);

    /** 数据偏移（视图切片时 &gt;0） */
    int offset();

    /** 是否连续内存（strides 等于 C-order） */
    boolean isContiguous();

    /** 标量张量取值（totalSize==1 时可用） */
    float item();

    /** 按索引取值 */
    float get(int... indices);

    /** 按索引设值 */
    IFloatTensor set(float value, int... indices);

    /** 填充 */
    IFloatTensor fill(float value);

    /** 导出为 flat float[] */
    float[] toFloatArray();

    /** 深拷贝 */
    IFloatTensor copy();
    IFloatTensor clone();

    /** 重塑（仅连续时零拷贝） */
    IFloatTensor reshape(int... newShape);


    // ==================== 视图操作（零拷贝） ====================

    /** 维度置换 */
    IFloatTensor permute(int... dims);

    /** 2D 转置 */
    IFloatTensor transpose(int dim0, int dim1);

    /** 全转置（2D 快捷方式） */
    IFloatTensor transpose();

    /** 去除尺寸为 1 的维度 */
    IFloatTensor squeeze(int... dims);

    /** 在指定位置插入尺寸为 1 的维度 */
    IFloatTensor unsqueeze(int dim);

    /** 沿 dim 切片 [start, end) */
    IFloatTensor slice(int dim, long start, long end);

    /** 沿 dim 取长度为 length 的子区域 */
    IFloatTensor narrow(int dim, long start, long length);

    /** 沿 dim 选取单个索引（返回低一阶的张量） */
    IFloatTensor select(int dim, long index);

    /** 展平从 startDim 到 endDim 的范围 */
    IFloatTensor flatten(int startDim, int endDim);

    /** 扩维到目标 shape（沿 dim=1 的维度广播） */
    IFloatTensor expand(int... shape);

    /** 强制转换为连续内存（视图 materialize） */
    IFloatTensor contiguous();

    /** 重复平铺 */
    IFloatTensor tile(int... repeats);

    /** 广播到目标 shape */
    IFloatTensor broadcastTo(int... shape);


    // ==================== 逐元素运算 ====================

    IFloatTensor add(IFloatTensor other);
    IFloatTensor sub(IFloatTensor other);
    IFloatTensor mul(IFloatTensor other);
    IFloatTensor div(IFloatTensor other);
    IFloatTensor add(float scalar);
    IFloatTensor sub(float scalar);
    IFloatTensor mul(float scalar);
    IFloatTensor div(float scalar);
    IFloatTensor neg();
    IFloatTensor abs();
    IFloatTensor sqrt();
    IFloatTensor exp();
    IFloatTensor log();
    IFloatTensor sin();
    IFloatTensor cos();
    IFloatTensor tan();
    IFloatTensor sigmoid();
    IFloatTensor relu();
    IFloatTensor tanh();
    IFloatTensor gelu();
    IFloatTensor leakyRelu(float alpha);
    IFloatTensor elu(float alpha);
    IFloatTensor selu();
    IFloatTensor silu();
    IFloatTensor mish();
    IFloatTensor softplus(float beta);
    IFloatTensor hardtanh(float minVal, float maxVal);

    /** Error function (erf), element-wise. */
    IFloatTensor erf();
    /** Round to nearest integer, element-wise. */
    IFloatTensor round();
    /** Floor, element-wise. */
    IFloatTensor floor();
    /** Ceil, element-wise. */
    IFloatTensor ceil();
    /** Sign function, element-wise: -1, 0, or 1. */
    IFloatTensor sign();
    IFloatTensor square();
    IFloatTensor pow(float n);
    IFloatTensor clamp(float min, float max);


    // ==================== 归约操作 ====================

    /** 沿指定维度求和 */
    IFloatTensor sum(int dim, boolean keepdim);

    /** 沿指定维度求均值 */
    IFloatTensor mean(int dim, boolean keepdim);

    /** 沿指定维度求最大值 */
    IFloatTensor max(int dim, boolean keepdim);

    /** 沿指定维度求最小值 */
    IFloatTensor min(int dim, boolean keepdim);

    /** 沿指定维度求积 */
    IFloatTensor prod(int dim, boolean keepdim);

    /** 全量归约 */
    float sumAll();
    float meanAll();
    float maxAll();
    float minAll();
    float prodAll();

    /** 沿指定维度累积和 */
    IFloatTensor cumsum(int dim);

    /** 沿指定维度累积积 */
    IFloatTensor cumprod(int dim);

    /** 沿指定维度返回最大值索引 */
    IFloatTensor argmax(int dim);

    /** 沿指定维度返回最小值索引 */
    IFloatTensor argmin(int dim);

    /** 沿指定维度的标准差 */
    IFloatTensor std(int dim, boolean keepdim);

    /** 沿指定维度的方差 */
    IFloatTensor var(int dim, boolean keepdim);


    // ==================== 高级操作 ====================

    /** 2D 矩阵乘法 */
    IFloatTensor mmul(IFloatTensor other);

    /** 批量矩阵乘法 (B,N,M) @ (B,M,P) → (B,N,P) */
    IFloatTensor bmm(IFloatTensor other);

    /** Einstein 求和约定 */
    IFloatTensor einsum(String subscript, IFloatTensor... others);

    /** Softmax（沿指定维度） */
    IFloatTensor softmax(int dim);

    /** LogSoftmax（沿指定维度） */
    IFloatTensor logSoftmax(int dim);

    /** 沿 dim 根据 index 收集元素 */
    IFloatTensor gather(int dim, IFloatTensor index);

    /** 沿 dim 根据 index 散布 source 的值 */
    IFloatTensor scatter(int dim, IFloatTensor index, IFloatTensor source);

    /** 沿 dim 根据 index 进行加法散布 */
    IFloatTensor scatterAdd(int dim, IFloatTensor index, IFloatTensor source);

    /** 条件选择: result[i] = condition[i] ? this[i] : other[i] */
    IFloatTensor where(IFloatTensor condition, IFloatTensor other);

    /** 沿 dim 取前 k 个最大值 */
    IFloatTensor topk(int k, int dim, boolean largest);

    /** Padding */
    IFloatTensor pad(int[][] padding, String mode, float value);

    /** 滑动窗口展开（im2col 等价） */
    IFloatTensor unfold(int dim, int size, int stride, int dilation);

    /** 非零元素索引 */
    IFloatTensor nonzero();

    /** 掩码选择 */
    IFloatTensor maskedSelect(IFloatTensor mask);

    /** 掩码填充 */
    IFloatTensor maskedFill(IFloatTensor mask, float value);

    /** 沿 dim 拼接 */
    IFloatTensor cat(int dim, IFloatTensor... others);

    /** 沿新 dim 堆叠 */
    IFloatTensor stack(int dim, IFloatTensor... others);

    /** 沿 dim 拆分 */
    List<IFloatTensor> unstack(int dim);

    /** p-范数归一化 */
    IFloatTensor normalize(float p, int dim);


    // ==================== 就地操作 ====================

    IFloatTensor add_(IFloatTensor other);
    IFloatTensor sub_(IFloatTensor other);
    IFloatTensor mul_(IFloatTensor other);
    IFloatTensor div_(IFloatTensor other);
    IFloatTensor fill_(float value);
    IFloatTensor copy_(IFloatTensor src);

    /** 就地取反 */
    IFloatTensor neg_();
    /** 就地绝对值 */
    IFloatTensor abs_();
    /** 就地 ReLU */
    IFloatTensor relu_();
    /** 就地 Sigmoid */
    IFloatTensor sigmoid_();
    /** 就地 Tanh */
    IFloatTensor tanh_();


    // ==================== 转换桥接 ====================

    /** 展平为 IFloatVector */
    IFloatVector toVector();

    /** 展平为 IFloatVector（强制拷贝） */
    IFloatVector toVectorCopy();

    /** 转为 IFloatMatrix（仅 2D） */
    IFloatMatrix toMatrix();

    /** 转为 IDoubleTensor（精度提升） */
    IDoubleTensor toDoubleTensor();
}
