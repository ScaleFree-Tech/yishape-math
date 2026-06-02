package com.yishape.lab.math.linalg.tensor;

import com.yishape.lab.math.compute.FloatVectorComputer;
import com.yishape.lab.math.compute.FloatFlatGemm;
import com.yishape.lab.math.compute.IFloatVectorComputer;
import com.yishape.lab.math.linalg.IFloatVector;
import com.yishape.lab.math.linalg.IFloatMatrix;
import com.yishape.lab.math.linalg.Linalg;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Float unary operator (avoids boxing vs java.util.function). */
@FunctionalInterface
interface FloatUnaryOp {
    float apply(float value);
}

/** Float binary operator (avoids boxing vs java.util.function). */
@FunctionalInterface
interface FloatBinaryOp {
    float apply(float a, float b);
}



/**
 * 基于 stride 的单精度张量实现.
 * <p>
 * 核心：flat float[] + TensorShape + int[] strides + int offset.
 * 视图操作（permute/slice/expand）通过修改 strides/offset 实现零拷贝.
 */
public class RereFloatTensor implements IFloatTensor {

    // ==================== 字段 ====================

    protected final float[] data;
    protected TensorShape shape;
    protected int[] strides;
    protected int offset;
    private static final FloatVectorComputer COMPUTER = new FloatVectorComputer();

    // ==================== 构造方法 ====================

    /** 主构造：新分配连续数据 */
    public RereFloatTensor(float[] data, int... shape) {
        validateShape(data.length, shape);
        this.data = data;
        this.shape = new TensorShape(shape);
        this.strides = TensorShape.computeCStrides(shape);
        this.offset = 0;
    }

    /** 视图构造：共享数据 */
    RereFloatTensor(float[] data, int offset, int[] shape, int[] strides) {
        if (shape.length != strides.length) {
            throw new IllegalArgumentException("shape and strides must have same length");
        }
        long expectedSize = 1;
        for (int d : shape) expectedSize *= d;
        if (expectedSize == 0) expectedSize = data.length;
        this.data = data;
        this.shape = new TensorShape(shape);
        this.strides = strides.clone();
        this.offset = offset;
    }

    /** View constructor with TensorShape */
    RereFloatTensor(float[] data, int offset, TensorShape shape) {
        this(data, offset, shape.shape(), shape.strides());
    }

    // ==================== Shape 查询 ====================

    @Override
    public int rank() { return shape.rank(); }

    @Override
    public int[] shape() { return shape.shape(); }

    @Override
    public int dim(int axis) { return shape.dim(axis); }

    @Override
    public long totalSize() { return shape.totalSize(); }

    @Override
    public int[] strides() { return strides.clone(); }

    @Override
    public int stride(int axis) { return shape.stride(axis); }

    @Override
    public int offset() { return offset; }

    @Override
    public boolean isContiguous() {
        return Arrays.equals(strides, TensorShape.computeCStrides(shape()));
    }

    @Override
    public float item() {
        if (totalSize() != 1) {
            throw new IllegalStateException("item() requires exactly 1 element, got " + totalSize());
        }
        return data[offset];
    }

    // ==================== 元素访问 ====================

    @Override
    public float get(int... indices) {
        if (indices.length != rank()) {
            // 尝试用可变参数的 get(int, int...) 风格处理
            // 但这里我们期望精确匹配
        }
        long idx = offset;
        for (int i = 0; i < indices.length; i++) {
            idx += indices[i] * (long) strides[i];
        }
        return data[(int) idx];
    }

    @Override
    public IFloatTensor set(float value, int... indices) {
        long idx = offset;
        for (int i = 0; i < indices.length; i++) {
            idx += indices[i] * (long) strides[i];
        }
        data[(int) idx] = value;
        return this;
    }

    @Override
    public IFloatTensor fill(float value) {
        if (isContiguous() && offset == 0) {
            Arrays.fill(data, value);
        } else {
            int nElem = (int) totalSize();
            for (int i = 0; i < nElem; i++) {
                linearSet(i, value);
            }
        }
        return this;
    }

    @Override
    public float[] toFloatArray() {
        if (isContiguous() && offset == 0 && totalSize() == data.length) {
            return data.clone();
        }
        int n = (int) totalSize();
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = linearGet(i);
        }
        return out;
    }

    /**
     * Return data as a contiguous row-major flat array.
     * If already contiguous with offset==0, returns a clone of the internal data.
     * Otherwise materializes via toFloatArray().
     */
    private float[] toContiguousFlat() {
        if (isContiguous() && offset == 0 && (int) totalSize() == data.length) {
            return data.clone();
        }
        return toFloatArray();
    }

    // ==================== 线性索引（stride 遍历） ====================

    protected long linearToOffset(long flatIndex) {
        long off = offset;
        long remaining = flatIndex;
        for (int i = rank() - 1; i >= 0; i--) {
            long idx = remaining % dim(i);
            off += idx * strides[i];
            remaining /= dim(i);
        }
        return off;
    }

    protected float linearGet(long flatIndex) {
        return data[(int) linearToOffset(flatIndex)];
    }

    protected void linearSet(long flatIndex, float value) {
        data[(int) linearToOffset(flatIndex)] = value;
    }

    // ==================== 视图操作（零拷贝） ====================

    @Override
    public IFloatTensor permute(int... dims) {
        int r = rank();
        if (dims.length != r) {
            throw new IllegalArgumentException("permute dims length " + dims.length + " != rank " + r);
        }
        int[] newShape = new int[r];
        int[] newStrides = new int[r];
        for (int i = 0; i < r; i++) {
            int ax = dims[i];
            if (ax < 0) ax += r;
            newShape[i] = shape.dim(ax);
            newStrides[i] = strides[ax];
        }
        return new RereFloatTensor(data, offset, newShape, newStrides);
    }

    @Override
    public IFloatTensor transpose(int dim0, int dim1) {
        int r = rank();
        if (dim0 < 0) dim0 += r;
        if (dim1 < 0) dim1 += r;
        int[] dims = new int[r];
        for (int i = 0; i < r; i++) dims[i] = i;
        dims[dim0] = dim1;
        dims[dim1] = dim0;
        return permute(dims);
    }

    @Override
    public IFloatTensor transpose() {
        if (rank() != 2) {
            throw new UnsupportedOperationException("transpose() requires 2D tensor");
        }
        return transpose(0, 1);
    }

    @Override
    public IFloatTensor squeeze(int... dims) {
        if (dims.length == 0) {
            // 去除所有 size==1 的维度
            int[] newShape = Arrays.stream(shape()).filter(d -> d != 1).toArray();
            int[] newStrides = new int[newShape.length];
            int idx = 0;
            for (int i = 0; i < rank(); i++) {
                if (dim(i) != 1) {
                    newStrides[idx++] = strides[i];
                }
            }
            if (newShape.length == 0) {
                return new RereFloatTensor(data, offset, new int[]{1}, new int[]{1});
            }
            return new RereFloatTensor(data, offset, newShape, newStrides);
        }
        boolean[] remove = new boolean[rank()];
        for (int ax : dims) {
            if (ax < 0) ax += rank();
            if (dim(ax) != 1) {
                throw new IllegalArgumentException("Cannot squeeze dim " + ax + " with size " + dim(ax));
            }
            remove[ax] = true;
        }
        int newRank = rank() - dims.length;
        int[] newShape = new int[newRank];
        int[] newStrides = new int[newRank];
        int idx = 0;
        for (int i = 0; i < rank(); i++) {
            if (!remove[i]) {
                newShape[idx] = dim(i);
                newStrides[idx] = strides[i];
                idx++;
            }
        }
        if (newRank == 0) {
            return new RereFloatTensor(data, offset, new int[]{1}, new int[]{1});
        }
        return new RereFloatTensor(data, offset, newShape, newStrides);
    }

    @Override
    public IFloatTensor unsqueeze(int axis) {
        int r = rank();
        if (axis < 0) axis = r + 1;
        if (axis < 0 || axis > r) {
            throw new IllegalArgumentException("Invalid unsqueeze axis: " + axis);
        }
        int[] newShape = new int[r + 1];
        int[] newStrides = new int[r + 1];
        for (int i = 0; i < axis; i++) {
            newShape[i] = dim(i);
            newStrides[i] = strides[i];
        }
        newShape[axis] = 1;
        newStrides[axis] = 0; // 广播时可沿此维度扩展
        for (int i = axis; i < r; i++) {
            newShape[i + 1] = dim(i);
            newStrides[i + 1] = strides[i];
        }
        return new RereFloatTensor(data, offset, newShape, newStrides);
    }

    @Override
    public IFloatTensor slice(int dim, long start, long end) {
        int[] newShape = shape().clone();
        if (dim < 0) dim += rank();
        int dimSize = newShape[dim];
        if (start < 0) start += dimSize;
        if (end < 0) end += dimSize;
        if (start < 0 || start >= dimSize)
            throw new IndexOutOfBoundsException("start out of bounds: " + start + " for dim size " + dimSize);
        if (end < start || end > dimSize)
            throw new IndexOutOfBoundsException("end out of bounds: " + end + " for dim size " + dimSize);
        newShape[dim] = (int) (end - start);
        int newOffset = offset + (int) start * strides[dim];
        return new RereFloatTensor(data, newOffset, newShape, strides);
    }

    @Override
    public IFloatTensor narrow(int dim, long start, long length) {
        return slice(dim, start, start + length);
    }

    @Override
    public IFloatTensor select(int dim, long index) {
        if (dim < 0) dim += rank();
        int newLen = rank() - 1;
        int[] newShape = new int[newLen];
        int[] newStrides = new int[newLen];
        int idx = 0;
        for (int i = 0; i < rank(); i++) {
            if (i != dim) {
                newShape[idx] = dim(i);
                newStrides[idx] = strides[i];
                idx++;
            }
        }
        int newOffset = offset + (int) index * strides[dim];
        if (newLen == 0) {
            // 返回标量
            return new RereFloatTensor(data, newOffset, new int[]{1}, new int[]{1});
        }
        return new RereFloatTensor(data, newOffset, newShape, newStrides);
    }

    @Override
    public IFloatTensor flatten(int startDim, int endDim) {
        int r = rank();
        if (startDim < 0) startDim += r;
        if (endDim < 0) endDim += r;
        if (!isContiguous()) {
            // 非连续：先 materialize
            return contiguous().flatten(startDim, endDim);
        }
        int flatDim = 1;
        for (int i = startDim; i <= endDim; i++) {
            flatDim *= dim(i);
        }
        int newRank = r - (endDim - startDim);
        int[] newShape = new int[newRank];
        int[] newStrides = new int[newRank];
        int idx = 0;
        for (int i = 0; i < startDim; i++) {
            newShape[idx] = dim(i);
            newStrides[idx] = strides[i];
            idx++;
        }
        newShape[idx] = flatDim;
        newStrides[idx] = (endDim + 1 < r) ? strides[endDim + 1] * dim(endDim + 1) : 1;
        idx++;
        for (int i = endDim + 1; i < r; i++) {
            newShape[idx] = dim(i);
            newStrides[idx] = strides[i];
            idx++;
        }
        return new RereFloatTensor(data, offset, newShape, newStrides);
    }

    @Override
    public IFloatTensor expand(int... shape) {
        int[] current = shape();
        if (shape.length != current.length) {
            throw new IllegalArgumentException("expand: target rank " + shape.length + " != current rank " + current.length);
        }
        int[] newStrides = strides.clone();
        int[] newShape = shape.clone();
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] == current[i]) {
                // keep stride
            } else if (current[i] == 1) {
                newStrides[i] = 0; // broadcast
            } else {
                throw new IllegalArgumentException("expand: dim " + i + " size " + current[i] + " cannot expand to " + shape[i]);
            }
        }
        return new RereFloatTensor(data, offset, newShape, newStrides);
    }

    @Override
    public IFloatTensor contiguous() {
        if (isContiguous() && offset == 0 && totalSize() == data.length) {
            return this;
        }
        int n = (int) totalSize();
        float[] newData = new float[n];
        for (int i = 0; i < n; i++) {
            newData[i] = linearGet(i);
        }
        return new RereFloatTensor(newData, shape());
    }

    @Override
    public IFloatTensor tile(int... repeats) {
        if (repeats.length != rank()) {
            throw new IllegalArgumentException("tile: repeats length " + repeats.length + " != rank " + rank());
        }
        int[] resultShape = new int[rank()];
        long total = 1;
        for (int i = 0; i < rank(); i++) {
            resultShape[i] = dim(i) * repeats[i];
            total *= resultShape[i];
        }
        float[] result = new float[(int) total];
        int[] indices = new int[rank()];
        int[] srcIndices = new int[rank()];
        for (long flat = 0; flat < total; flat++) {
            long remaining = flat;
            for (int j = rank() - 1; j >= 0; j--) {
                indices[j] = (int) (remaining % resultShape[j]);
                remaining /= resultShape[j];
                srcIndices[j] = indices[j] % dim(j);
            }
            result[(int) flat] = get(srcIndices);
        }
        return new RereFloatTensor(result, resultShape);
    }

    @Override
    public IFloatTensor broadcastTo(int... targetShape) {
        int[] current = shape();
        int[] resultStrides = new int[targetShape.length];
        int[] resultShape = new int[targetShape.length];
        int diff = targetShape.length - current.length;
        for (int i = 0; i < diff; i++) {
            resultShape[i] = targetShape[i];
            resultStrides[i] = 0;
        }
        for (int i = 0; i < current.length; i++) {
            int ax = diff + i;
            resultShape[ax] = targetShape[ax];
            if (current[i] == 1) {
                resultStrides[ax] = 0;
            } else {
                resultStrides[ax] = strides[i];
            }
        }
        return new RereFloatTensor(data, offset, resultShape, resultStrides);
    }

    @Override
    public IFloatTensor reshape(int... newShape) {
        long newSize = 1;
        newShape = shape.inferReshape(newShape);
        for (int d : newShape) newSize *= d;
        if (newSize != totalSize()) {
            throw new IllegalArgumentException("Cannot reshape " + totalSize() + " to " + newSize);
        }
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(data, offset, newShape,
                TensorShape.computeCStrides(newShape));
        }
        // 非连续：materialize 后再 reshape
        return contiguous().reshape(newShape);
    }

    // ==================== 逐元素运算（一元） ====================

    private IFloatTensor applyUnary(FloatUnaryOp op) {
        int n = (int) totalSize();
        if (isContiguous() && offset == 0 && n == data.length) {
            float[] result = new float[n];
            for (int i = 0; i < n; i++) result[i] = op.apply(data[i]);
            return new RereFloatTensor(result, shape());
        }
        float[] result = new float[n];
        for (int i = 0; i < n; i++) result[i] = op.apply(linearGet(i));
        return new RereFloatTensor(result, shape());
    }

    @Override public IFloatTensor neg() {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.negate(data), shape());
        }
        return applyUnary(v -> -v);
    }
    @Override public IFloatTensor abs() {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.universalOperate(data, IFloatVectorComputer.UniversalOperation.ABS, 0), shape());
        }
        return applyUnary(v -> (float) Math.abs(v));
    }
    @Override public IFloatTensor sqrt() {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.universalOperate(data, IFloatVectorComputer.UniversalOperation.SQRT, 0), shape());
        }
        return applyUnary(v -> (float) Math.sqrt(v));
    }
    @Override public IFloatTensor exp() {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.universalOperate(data, IFloatVectorComputer.UniversalOperation.EXP, 0), shape());
        }
        return applyUnary(v -> (float) Math.exp(v));
    }
    @Override public IFloatTensor log() {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.universalOperate(data, IFloatVectorComputer.UniversalOperation.LOG, 0), shape());
        }
        return applyUnary(v -> (float) Math.log(v));
    }
    @Override public IFloatTensor sin() {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.universalOperate(data, IFloatVectorComputer.UniversalOperation.SIN, 0), shape());
        }
        return applyUnary(v -> (float) Math.sin(v));
    }
    @Override public IFloatTensor cos() {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.universalOperate(data, IFloatVectorComputer.UniversalOperation.COS, 0), shape());
        }
        return applyUnary(v -> (float) Math.cos(v));
    }
    @Override public IFloatTensor tan() {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.universalOperate(data, IFloatVectorComputer.UniversalOperation.TAN, 0), shape());
        }
        return applyUnary(v -> (float) Math.tan(v));
    }
    @Override public IFloatTensor square() { return applyUnary(v -> v * v); }
    @Override public IFloatTensor sigmoid() {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.universalOperate(data, IFloatVectorComputer.UniversalOperation.SIGMOID, 0), shape());
        }
        return applyUnary(v -> (float)(1.0 / (1.0 + Math.exp(-v))));
    }
    @Override public IFloatTensor relu() {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.universalOperate(data, IFloatVectorComputer.UniversalOperation.RELU, 0), shape());
        }
        return applyUnary(v -> v > 0 ? v : 0);
    }
    @Override public IFloatTensor tanh() {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.universalOperate(data, IFloatVectorComputer.UniversalOperation.TANH, 0), shape());
        }
        return applyUnary(v -> (float) Math.tanh(v));
    }
    @Override public IFloatTensor gelu() {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.universalOperate(data, IFloatVectorComputer.UniversalOperation.GELU, 0), shape());
        }
        return applyUnary(x -> {
            float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1f + (float) Math.tanh(inner));
        });
    }
    @Override public IFloatTensor leakyRelu(float alpha) {
        return applyUnary(x -> x >= 0 ? x : alpha * x);
    }
    @Override public IFloatTensor elu(float alpha) {
        return applyUnary(x -> (float)(x >= 0 ? x : alpha * (Math.exp(x) - 1)));
    }
    @Override public IFloatTensor selu() {
        float alpha = 1.6732632f, scale = 1.0507010f;
        return applyUnary(x -> (float)(scale * (x >= 0 ? x : alpha * (Math.exp(x) - 1))));
    }
    @Override public IFloatTensor silu() {
        return applyUnary(x -> (float)(x / (1.0 + Math.exp(-x))));
    }
    @Override public IFloatTensor mish() {
        return applyUnary(x -> (float)(x * Math.tanh(Math.log(1.0 + Math.exp(x)))));
    }
    @Override public IFloatTensor softplus(float beta) {
        return applyUnary(x -> {
            float bx = beta * x;
            return bx > 20 ? x : (float)(Math.log(1.0 + Math.exp(bx)) / beta);
        });
    }
    @Override public IFloatTensor hardtanh(float minVal, float maxVal) {
        return applyUnary(x -> (float) Math.min(Math.max(x, minVal), maxVal));
    }
    @Override public IFloatTensor pow(float n) { return applyUnary(v -> (float) Math.pow(v, n)); }

    @Override
    public IFloatTensor clamp(float min, float max) {
        return applyUnary(v -> (float) Math.max(min, Math.min(max, v)));
    }

    // ==================== 逐元素运算（标量） ====================

    @Override public IFloatTensor add(float scalar) {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.binaryOperate(data, scalar, IFloatVectorComputer.BinaryOperation.ADD), shape());
        }
        return applyUnary(v -> v + scalar);
    }
    @Override public IFloatTensor sub(float scalar) {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.binaryOperate(data, scalar, IFloatVectorComputer.BinaryOperation.SUBTRACT), shape());
        }
        return applyUnary(v -> v - scalar);
    }
    @Override public IFloatTensor mul(float scalar) {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.binaryOperate(data, scalar, IFloatVectorComputer.BinaryOperation.MULTIPLY), shape());
        }
        return applyUnary(v -> v * scalar);
    }
    @Override public IFloatTensor div(float scalar) {
        if (isContiguous() && offset == 0) {
            return new RereFloatTensor(COMPUTER.binaryOperate(data, scalar, IFloatVectorComputer.BinaryOperation.DIVIDE), shape());
        }
        return applyUnary(v -> v / scalar);
    }

    // ==================== 逐元素运算（张量 + 广播） ====================

    @Override
    public IFloatTensor add(IFloatTensor other) {
        if (other instanceof RereFloatTensor rt && isContiguous() && offset == 0
                && rt.isContiguous() && rt.offset == 0 && data.length == rt.data.length) {
            return new RereFloatTensor(COMPUTER.binaryOperate(data, rt.data, IFloatVectorComputer.BinaryOperation.ADD), shape());
        }
        return elementWiseBinary(other, Float::sum);
    }

    @Override
    public IFloatTensor sub(IFloatTensor other) {
        if (other instanceof RereFloatTensor rt && isContiguous() && offset == 0
                && rt.isContiguous() && rt.offset == 0 && data.length == rt.data.length) {
            return new RereFloatTensor(COMPUTER.binaryOperate(data, rt.data, IFloatVectorComputer.BinaryOperation.SUBTRACT), shape());
        }
        return elementWiseBinary(other, (a, b) -> a - b);
    }

    @Override
    public IFloatTensor mul(IFloatTensor other) {
        if (other instanceof RereFloatTensor rt && isContiguous() && offset == 0
                && rt.isContiguous() && rt.offset == 0 && data.length == rt.data.length) {
            return new RereFloatTensor(COMPUTER.binaryOperate(data, rt.data, IFloatVectorComputer.BinaryOperation.MULTIPLY), shape());
        }
        return elementWiseBinary(other, (a, b) -> a * b);
    }

    @Override
    public IFloatTensor div(IFloatTensor other) {
        if (other instanceof RereFloatTensor rt && isContiguous() && offset == 0
                && rt.isContiguous() && rt.offset == 0 && data.length == rt.data.length) {
            return new RereFloatTensor(COMPUTER.binaryOperate(data, rt.data, IFloatVectorComputer.BinaryOperation.DIVIDE), shape());
        }
        return elementWiseBinary(other, (a, b) -> a / b);
    }

    private IFloatTensor elementWiseBinary(IFloatTensor other, FloatBinaryOp op) {
        int[] shapeA = shape();
        int[] shapeB = other.shape();
        int[] resultShape = TensorShape.broadcastShape(shapeA, shapeB);
        long total = 1;
        for (int d : resultShape) total *= d;
        float[] result = new float[(int) total];
        int rank = resultShape.length;

        // 对齐两个输入到结果秩
        RereFloatTensor aAligned = (RereFloatTensor) broadcastTo(resultShape);
        RereFloatTensor bAligned = (RereFloatTensor) other.broadcastTo(resultShape);

        for (long i = 0; i < total; i++) {
            long remaining = i;
            long aOff = aAligned.offset;
            long bOff = bAligned.offset;
            for (int j = rank - 1; j >= 0; j--) {
                int idx = (int) (remaining % resultShape[j]);
                remaining /= resultShape[j];
                aOff += idx * (long) aAligned.strides[j];
                bOff += idx * (long) bAligned.strides[j];
            }
            result[(int) i] = op.apply(aAligned.data[(int) aOff], bAligned.data[(int) bOff]);
        }
        return new RereFloatTensor(result, resultShape);
    }

    // ==================== 归约 ====================

    @Override
    public IFloatTensor sum(int dim, boolean keepdim) {
        if (dim < 0) dim += rank();
        long total = totalSize();
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dim(i);
        int reduce = dim(dim);
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= dim(i);

        int[] newShape;
        if (keepdim) {
            newShape = shape().clone();
            newShape[dim] = 1;
        } else {
            newShape = new int[rank() - 1];
            int idx = 0;
            for (int i = 0; i < rank(); i++) {
                if (i != dim) newShape[idx++] = dim(i);
            }
        }
        float[] result = new float[(int) computeSize(newShape)];

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                float sum = 0;
                for (int r = 0; r < reduce; r++) {
                    int[] indices = new int[rank()];
                    int tmp = o;
                    for (int j = 0; j < dim; j++) {
                        indices[j] = tmp % dim(j);
                        tmp /= dim(j);
                    }
                    indices[dim] = r;
                    tmp = i;
                    for (int j = rank() - 1; j > dim; j--) {
                        indices[j] = tmp % dim(j);
                        tmp /= dim(j);
                    }
                    sum += get(indices);
                }
                // write to result
                int[] outIdx;
                if (keepdim) {
                    outIdx = new int[rank()];
                    int tmp = o;
                    for (int j = 0; j < dim; j++) {
                        outIdx[j] = tmp % dim(j);
                        tmp /= dim(j);
                    }
                    outIdx[dim] = 0;
                    tmp = i;
                    for (int j = rank() - 1; j > dim; j--) {
                        outIdx[j] = tmp % dim(j);
                        tmp /= dim(j);
                    }
                } else {
                    outIdx = new int[rank() - 1];
                    int tmp = o;
                    for (int j = 0; j < dim; j++) {
                        outIdx[j] = tmp % dim(j);
                        tmp /= dim(j);
                    }
                    tmp = i;
                    for (int j = rank() - 1; j > dim; j--) {
                        outIdx[j - 1] = tmp % dim(j);
                        tmp /= dim(j);
                    }
                }
                long linear = 0;
                int stride = 1;
                for (int j = newShape.length - 1; j >= 0; j--) {
                    linear += outIdx[j] * stride;
                    stride *= newShape[j];
                }
                result[(int) linear] = sum;
            }
        }
        return new RereFloatTensor(result, newShape);
    }

    @Override
    public IFloatTensor mean(int dim, boolean keepdim) {
        IFloatTensor s = sum(dim, keepdim);
        return s.div(this.dim(dim));
    }

    @Override
    public IFloatTensor max(int dim, boolean keepdim) {
        if (dim < 0) dim += rank();
        return reduceDim(dim, keepdim, Float.NEGATIVE_INFINITY,
            (a, b) -> a > b ? a : b);
    }

    @Override
    public IFloatTensor min(int dim, boolean keepdim) {
        if (dim < 0) dim += rank();
        return reduceDim(dim, keepdim, Float.POSITIVE_INFINITY,
            (a, b) -> a < b ? a : b);
    }

    @Override
    public IFloatTensor prod(int dim, boolean keepdim) {
        if (dim < 0) dim += rank();
        return reduceDim(dim, keepdim, 1.0f, (a, b) -> a * b);
    }

    private IFloatTensor reduceDim(int dim, boolean keepdim,
                                     float init, FloatBinaryOp reducer) {
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dim(i);
        int reduce = dim(dim);
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= dim(i);

        int[] newShape;
        if (keepdim) {
            newShape = shape().clone();
            newShape[dim] = 1;
        } else {
            newShape = new int[rank() - 1];
            int idx = 0;
            for (int i = 0; i < rank(); i++) {
                if (i != dim) newShape[idx++] = dim(i);
            }
        }
        float[] result = new float[(int) computeSize(newShape)];
        Arrays.fill(result, init);

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                float val = init;
                for (int r = 0; r < reduce; r++) {
                    val = reducer.apply(val, getStrided(o, dim, r, i));
                }
                // write to result
                long linear;
                if (keepdim) {
                    linear = flatIndexKeepdim(o, dim, i, newShape);
                } else {
                    linear = flatIndexReduce(o, dim, i, newShape);
                }
                result[(int) linear] = val;
            }
        }
        return new RereFloatTensor(result, newShape);
    }

    /** 使用 stride 获取元素（o=outer index, dim=target, r=reduce index, i=inner index） */
    private float getStrided(int o, int dim, int r, int i) {
        long off = offset;
        // outer
        for (int j = 0; j < dim; j++) {
            off += (o % dim(j)) * (long) strides[j];
            o /= dim(j);
        }
        off += r * (long) strides[dim];
        // inner
        for (int j = rank() - 1; j > dim; j--) {
            off += (i % dim(j)) * (long) strides[j];
            i /= dim(j);
        }
        return data[(int) off];
    }

    private long flatIndexKeepdim(int o, int dim, int i, int[] shape) {
        // Pre-compute mixed-radix decomposition of o for dimensions 0..dim-1
        int[] outerVals = new int[dim];
        int tmp = o;
        for (int j = 0; j < dim; j++) {
            outerVals[j] = tmp % dim(j);
            tmp /= dim(j);
        }
        long idx = 0;
        int stride = 1;
        for (int j = shape.length - 1; j >= 0; j--) {
            int v;
            if (j == dim) {
                v = 0;
            } else if (j < dim) {
                v = outerVals[j];
            } else {
                v = i % dim(j);
                i /= dim(j);
            }
            idx += v * stride;
            stride *= shape[j];
        }
        return idx;
    }

    private long flatIndexReduce(int o, int dim, int i, int[] shape) {
        // Pre-compute mixed-radix decomposition of o for outer dims
        int[] outerVals = new int[dim];
        int tmp = o;
        for (int j = 0; j < dim; j++) {
            outerVals[j] = tmp % dim(j);
            tmp /= dim(j);
        }
        long idx = 0;
        int stride = 1;
        for (int j = shape.length - 1; j >= 0; j--) {
            int actualDim = (j < dim) ? j : j + 1;
            int v;
            if (actualDim < dim) {
                v = outerVals[actualDim];
            } else {
                v = i % dim(actualDim);
                i /= dim(actualDim);
            }
            idx += v * stride;
            stride *= shape[j];
        }
        return idx;
    }

    @Override
    public float sumAll() {
        return reduceAll(0.0f, Float::sum);
    }

    @Override
    public float meanAll() { return sumAll() / totalSize(); }

    @Override
    public float maxAll() {
        return reduceAll(Float.NEGATIVE_INFINITY, Math::max);
    }

    @Override
    public float minAll() {
        return reduceAll(Float.POSITIVE_INFINITY, Math::min);
    }

    @Override
    public float prodAll() { return reduceAll(1.0f, (a, b) -> a * b); }

    private float reduceAll(float init, FloatBinaryOp op) {
        int n = (int) totalSize();
        float result = init;
        for (int i = 0; i < n; i++) {
            result = op.apply(result, linearGet(i));
        }
        return result;
    }

    @Override
    public IFloatTensor argmax(int dim) {
        // argmax returns indices as float tensor
        if (dim < 0) dim += rank();
        int[] newShape;
        if (rank() == 1) {
            newShape = new int[]{1};
        } else {
            newShape = new int[rank() - 1];
            int idx = 0;
            for (int i = 0; i < rank(); i++) {
                if (i != dim) newShape[idx++] = dim(i);
            }
        }
        float[] result = new float[(int) computeSize(newShape)];
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dim(i);
        int reduce = dim(dim);
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= dim(i);

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                float maxVal = Float.NEGATIVE_INFINITY;
                int maxIdx = 0;
                for (int r = 0; r < reduce; r++) {
                    float v = getStrided(o, dim, r, i);
                    if (v > maxVal) {
                        maxVal = v;
                        maxIdx = r;
                    }
                }
                long linear = rank() == 1 ? 0 :
                    flatIndexReduce(o, dim, i, newShape);
                result[(int) linear] = maxIdx;
            }
        }
        return new RereFloatTensor(result, newShape);
    }

    @Override
    public IFloatTensor argmin(int dim) {
        if (dim < 0) dim += rank();
        int[] newShape;
        if (rank() == 1) {
            newShape = new int[]{1};
        } else {
            newShape = new int[rank() - 1];
            int idx = 0;
            for (int i = 0; i < rank(); i++) {
                if (i != dim) newShape[idx++] = dim(i);
            }
        }
        float[] result = new float[(int) computeSize(newShape)];
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dim(i);
        int reduce = dim(dim);
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= dim(i);

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                float minVal = Float.POSITIVE_INFINITY;
                int minIdx = 0;
                for (int r = 0; r < reduce; r++) {
                    float v = getStrided(o, dim, r, i);
                    if (v < minVal) {
                        minVal = v;
                        minIdx = r;
                    }
                }
                long linear = rank() == 1 ? 0 :
                    flatIndexReduce(o, dim, i, newShape);
                result[(int) linear] = minIdx;
            }
        }
        return new RereFloatTensor(result, newShape);
    }

    @Override
    public IFloatTensor cumsum(int dim) {
        // cumulative sum along dim
        int[] resultShape = shape();
        float[] result = new float[(int) totalSize()];
        int n = dim(dim);
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dim(i);
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= dim(i);

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                float cum = 0;
                for (int r = 0; r < n; r++) {
                    cum += getStrided(o, dim, r, i);
                    int[] outIdx = mixedIndex(o, dim, r, i, resultShape);
                    long linear = linearIndex(outIdx, resultShape);
                    result[(int) linear] = cum;
                }
            }
        }
        return new RereFloatTensor(result, resultShape);
    }

    @Override
    public IFloatTensor cumprod(int dim) {
        int[] resultShape = shape();
        float[] result = new float[(int) totalSize()];
        int n = dim(dim);
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dim(i);
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= dim(i);

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                float cum = 1;
                for (int r = 0; r < n; r++) {
                    cum *= getStrided(o, dim, r, i);
                    int[] outIdx = mixedIndex(o, dim, r, i, resultShape);
                    long linear = linearIndex(outIdx, resultShape);
                    result[(int) linear] = cum;
                }
            }
        }
        return new RereFloatTensor(result, resultShape);
    }

    @Override
    public IFloatTensor std(int dim, boolean keepdim) {
        IFloatTensor m = mean(dim, keepdim);
        int n = dim(dim);
        // broadcast mean back, compute variance
        IFloatTensor centered = sub(m);
        IFloatTensor sq = centered.pow(2);
        IFloatTensor var = sq.sum(dim, keepdim).div(n - 1); // unbiased
        return var.sqrt();
    }

    @Override
    public IFloatTensor var(int dim, boolean keepdim) {
        IFloatTensor m = mean(dim, keepdim);
        int n = dim(dim);
        IFloatTensor centered = sub(m);
        IFloatTensor sq = centered.pow(2);
        return sq.sum(dim, keepdim).div(n - 1);
    }

    // ==================== Softmax / LogSoftmax ====================

    @Override
    public IFloatTensor softmax(int dim) {
        if (dim < 0) dim += rank();
        IFloatTensor maxVals = max(dim, true);
        IFloatTensor shifted = sub(maxVals);
        IFloatTensor exps = shifted.exp();
        IFloatTensor sumExps = exps.sum(dim, true);
        return exps.div(sumExps);
    }

    @Override
    public IFloatTensor logSoftmax(int dim) {
        if (dim < 0) dim += rank();
        IFloatTensor maxVals = max(dim, true);
        IFloatTensor shifted = sub(maxVals);
        IFloatTensor exps = shifted.exp();
        IFloatTensor sumExps = exps.sum(dim, true);
        return shifted.sub(sumExps.log());
    }

    // ==================== 线性代数 ====================

    @Override
    public IFloatTensor mmul(IFloatTensor other) {
        if (rank() != 2 || other.rank() != 2) {
            throw new UnsupportedOperationException("mmul only supports 2D tensors, use bmm for batched");
        }
        int m = dim(0), k = dim(1), n = other.dim(1);
        if (dim(1) != other.dim(0)) {
            throw new IllegalArgumentException("mmul: " + dim(0) + "x" + dim(1)
                + " @ " + other.dim(0) + "x" + other.dim(1) + " incompatible");
        }
        float[] aFlat = toContiguousFlat();
        float[] bFlat = ((RereFloatTensor) other).toContiguousFlat();
        float[] result = FloatFlatGemm.flatMmul(aFlat, m, k, bFlat, n);
        return new RereFloatTensor(result, m, n);
    }

    @Override
    public IFloatTensor bmm(IFloatTensor other) {
        int r = rank();
        int or = other.rank();
        if (r < 2 || or < 2) {
            throw new IllegalArgumentException("bmm requires rank >= 2 for both tensors");
        }
        int batchRank = Math.max(r, or) - 2;
        // 对齐 batch dims
        int[] thisShape = shape();
        int[] otherShape = other.shape();
        int[] batchShape = new int[batchRank];
        for (int i = 0; i < batchRank; i++) {
            int thisDim = i < r - 2 ? thisShape[i] : 1;
            int otherDim = i < or - 2 ? otherShape[i] : 1;
            if (thisDim != otherDim && thisDim != 1 && otherDim != 1) {
                throw new IllegalArgumentException("bmm batch dims incompatible: " + Arrays.toString(thisShape)
                    + " @ " + Arrays.toString(otherShape));
            }
            batchShape[i] = Math.max(thisDim, otherDim);
        }
        int m = dim(-2);
        int k = dim(-1);
        int n = other.dim(-1);
        if (k != other.dim(-2)) {
            throw new IllegalArgumentException("bmm: " + m + "x" + k + " @ " + other.dim(-2) + "x" + n + " incompatible");
        }

        // Flat batch iteration
        long batchTotal = 1;
        for (int bd : batchShape) batchTotal *= bd;
        float[] result = new float[(int) (batchTotal * m * n)];

        boolean bothContig = isContiguous() && other.isContiguous()
            && offset == 0 && ((RereFloatTensor) other).offset == 0;
        if (bothContig && batchTotal > 0) {
            float[] aFull = data.clone();
            float[] bFull = ((RereFloatTensor) other).data.clone();
            for (long b = 0; b < batchTotal; b++) {
                int aOff = (int) (b * m * k);
                int bOff = (int) (b * k * n);
                int cOff = (int) (b * m * n);
                float[] cSlice = FloatFlatGemm.flatMmul(aFull, aOff, m, k, bFull, bOff, n);
                System.arraycopy(cSlice, 0, result, cOff, m * n);
            }
        } else {
            for (long b = 0; b < batchTotal; b++) {
                int off = (int) (b * m * n);
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        float sum = 0;
                        for (int t = 0; t < k; t++) {
                            RereFloatTensor otherRDT = (RereFloatTensor) other;
                            sum += getWithBatch(b, batchShape, i, t) * otherRDT.getWithBatch(b, batchShape, t, j);
                        }
                        result[off + i * n + j] = sum;
                    }
                }
            }
        }

        int[] resultShape = new int[batchRank + 2];
        for (int i = 0; i < batchRank; i++) resultShape[i] = batchShape[i];
        resultShape[batchRank] = m;
        resultShape[batchRank + 1] = n;
        return new RereFloatTensor(result, resultShape);
    }

    private float getWithBatch(long batchFlat, int[] batchShape, int dim0Idx, int dim1Idx) {
        long off = offset;
        // batch dims
        long remaining = batchFlat;
        for (int j = batchShape.length - 1; j >= 0; j--) {
            long idx = remaining % batchShape[j];
            remaining /= batchShape[j];
            int thisBatchDim = j < rank() - 2 ? dim(j) : 1;
            off += (idx % thisBatchDim) * (long) strides[j];
        }
        // last 2 dims
        int r = rank();
        off += dim0Idx * (long) strides[r - 2];
        off += dim1Idx * (long) strides[r - 1];
        return data[(int) off];
    }

    @Override
    public IFloatTensor einsum(String subscript, IFloatTensor... others) {
        // Delegate to existing mechanism - this is a placeholder for full implementation
        // Full einsum is complex and will be expanded in Phase 4
        String[] parts = subscript.split("->");
        if (parts.length != 2) {
            throw new IllegalArgumentException("Invalid einsum subscript: " + subscript);
        }
        String[] inputs = parts[0].split(",");
        if (inputs.length != others.length + 1) {
            throw new IllegalArgumentException("Number of subscripts must match number of inputs");
        }

        // Handle common cases
        if (subscript.equals("ij,jk->ik")) {
            return mmul(others[0]);
        }
        if (subscript.equals("bij,bjk->bik")) {
            return bmm(others[0]);
        }

        // For multi-head attention: bqhd,bkhd->bhqk
        if (subscript.equals("bqhd,bkhd->bhqk")) {
            IFloatTensor a = this;
            IFloatTensor b = others[0];
            int B = a.dim(0), Q = a.dim(1), H = a.dim(2), D = a.dim(3);
            int K = b.dim(1);
            // Permute a: (B,Q,H,D) -> (B,H,Q,D)
            // Permute b: (B,K,H,D) -> (B,H,K,D)
            // bmm: (B,H,Q,D) @ (B,H,D,K) -> (B,H,Q,K)
            IFloatTensor aP = a.permute(0, 2, 1, 3);
            IFloatTensor bP = b.permute(0, 2, 3, 1);
            IFloatTensor result = aP.bmm(bP);
            return result.permute(0, 2, 1, 3).reshape(B, H, Q, K);
        }

        throw new UnsupportedOperationException("einsum subscript not yet implemented: " + subscript);
    }

    // ==================== 高级操作 ====================

    @Override
    public IFloatTensor gather(int dim, IFloatTensor index) {
        if (dim < 0) dim += rank();
        int srcRank = rank();
        int[] srcShape = shape();
        int[] idxShape = index.shape();
        int idxRank = idxShape.length;
        // Output shape: index.shape + source.shape[dim+1:]
        int trailingRank = srcRank - dim - 1;
        int[] resultShape = new int[idxRank + trailingRank];
        System.arraycopy(idxShape, 0, resultShape, 0, idxRank);
        for (int i = 0; i < trailingRank; i++) resultShape[idxRank + i] = srcShape[dim + 1 + i];
        int resultSize = 1;
        for (int d : resultShape) resultSize *= d;
        float[] result = new float[resultSize];
        int[] idxIdx = new int[idxRank];
        for (int i = 0; i < resultSize; i++) {
            int[] outIdx = unlinearize(i, resultShape);
            // Extract only the first idxRank indices for the index tensor lookup
            System.arraycopy(outIdx, 0, idxIdx, 0, idxRank);
            int gatherIdx = (int) index.get(idxIdx);
            // Build source index: outIdx[0..idxRank-1] with gatherIdx at dim
            int[] srcIdx = new int[srcRank];
            for (int j = 0; j < dim; j++) srcIdx[j] = outIdx[j];
            srcIdx[dim] = gatherIdx;
            for (int j = 0; j < trailingRank; j++) srcIdx[dim + 1 + j] = outIdx[idxRank + j];
            result[i] = get(srcIdx);
        }
        return new RereFloatTensor(result, resultShape);
    }

    @Override
    public IFloatTensor scatter(int dim, IFloatTensor index, IFloatTensor source) {
        if (dim < 0) dim += rank();
        float[] result = toFloatArray();
        int[] resultShape = shape();
        for (long i = 0; i < index.totalSize(); i++) {
            int[] idx = unlinearize(i, index.shape());
            int[] tgtIdx = new int[rank()];
            for (int j = 0; j < rank(); j++) {
                tgtIdx[j] = j == dim ? (int) index.get(idx) : idx[j];
            }
            long linear = linearIndex(tgtIdx, resultShape);
            result[(int) linear] = source.get(idx);
        }
        return new RereFloatTensor(result, resultShape);
    }

    @Override
    public IFloatTensor scatterAdd(int dim, IFloatTensor index, IFloatTensor source) {
        if (dim < 0) dim += rank();
        float[] result = toFloatArray();
        int[] resultShape = shape();
        for (long i = 0; i < index.totalSize(); i++) {
            int[] idx = unlinearize(i, index.shape());
            int[] tgtIdx = new int[rank()];
            for (int j = 0; j < rank(); j++) {
                tgtIdx[j] = j == dim ? (int) index.get(idx) : idx[j];
            }
            long linear = linearIndex(tgtIdx, resultShape);
            result[(int) linear] += source.get(idx);
        }
        return new RereFloatTensor(result, resultShape);
    }

    @Override
    public IFloatTensor where(IFloatTensor condition, IFloatTensor other) {
        int[] resultShape = TensorShape.broadcastShape(
            shape(), TensorShape.broadcastShape(other.shape(), condition.shape()));
        float[] result = new float[(int) computeSize(resultShape)];
        for (long i = 0; i < result.length; i++) {
            int[] idx = unlinearize(i, resultShape);
            float condVal = getWithBroadcast(idx, condition);
            result[(int) i] = condVal > 0.5 ?
                getWithBroadcast(idx, this) : getWithBroadcast(idx, other);
        }
        return new RereFloatTensor(result, resultShape);
    }

    @Override
    public IFloatTensor topk(int k, int dim, boolean largest) {
        if (dim < 0) dim += rank();
        int n = dim(dim);
        int[] resultShape = shape().clone();
        resultShape[dim] = k;
        float[] values = new float[(int) computeSize(resultShape)];
        float[] indices = new float[(int) computeSize(resultShape)];
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dim(i);
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= dim(i);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                // collect all values along dim
                float[] vals = new float[n];
                for (int r = 0; r < n; r++) {
                    vals[r] = getStrided(o, dim, r, i);
                }
                // argsort
                Integer[] idxs = new Integer[n];
                for (int r = 0; r < n; r++) idxs[r] = r;
                Arrays.sort(idxs, (a, b) -> largest ?
                    Float.compare(vals[b], vals[a]) : Float.compare(vals[a], vals[b]));
                // write top k
                for (int r = 0; r < k; r++) {
                    int[] outIdx;
                    if (rank() == 1) {
                        outIdx = new int[]{r};
                    } else {
                        outIdx = new int[rank()];
                        int tmp = o;
                        for (int j = 0; j < dim; j++) {
                            outIdx[j] = tmp % dim(j);
                            tmp /= dim(j);
                        }
                        outIdx[dim] = r;
                        tmp = i;
                        for (int j = rank() - 1; j > dim; j--) {
                            outIdx[j] = tmp % dim(j);
                            tmp /= dim(j);
                        }
                    }
                    long linear = linearIndex(outIdx, resultShape);
                    values[(int) linear] = vals[idxs[r]];
                    indices[(int) linear] = idxs[r];
                }
            }
        }
        // Return values only (user can use topk with indices via separate call)
        return new RereFloatTensor(values, resultShape);
    }

    @Override
    public IFloatTensor pad(int[][] padding, String mode, float value) {
        if (padding.length != rank()) {
            throw new IllegalArgumentException("padding length must match rank");
        }
        int[] newShape = new int[rank()];
        for (int i = 0; i < rank(); i++) {
            newShape[i] = dim(i) + padding[i][0] + padding[i][1];
        }
        long total = 1;
        for (int d : newShape) total *= d;
        float[] result = new float[(int) total];
        if (!"constant".equals(mode)) {
            throw new UnsupportedOperationException("pad mode not implemented: " + mode);
        }
        // Fill with pad value
        Arrays.fill(result, value);
        // Copy original data
        for (long i = 0; i < totalSize(); i++) {
            int[] srcIdx = unlinearize((int) i, shape());
            int[] tgtIdx = new int[rank()];
            boolean valid = true;
            for (int j = 0; j < rank(); j++) {
                tgtIdx[j] = srcIdx[j] + padding[j][0];
                if (tgtIdx[j] < 0 || tgtIdx[j] >= newShape[j]) valid = false;
            }
            if (valid) {
                long linear = linearIndex(tgtIdx, newShape);
                result[(int) linear] = get(srcIdx);
            }
        }
        return new RereFloatTensor(result, newShape);
    }

    @Override
    public IFloatTensor unfold(int dim, int size, int stride, int dilation) {
        if (dim < 0) dim += rank();
        int n = dim(dim);
        int outSize = (n - dilation * (size - 1) - 1) / stride + 1;
        int[] resultShape = new int[rank() + 1];
        for (int i = 0; i < rank(); i++) {
            resultShape[i] = i == dim ? outSize : dim(i);
        }
        resultShape[rank()] = size;
        long total = 1;
        for (int d : resultShape) total *= d;
        float[] result = new float[(int) total];
        for (long i = 0; i < result.length; i++) {
            int[] idx = unlinearize((int) i, resultShape);
            int[] srcIdx = new int[rank()];
            for (int j = 0; j < rank(); j++) {
                if (j == dim) {
                    srcIdx[j] = idx[j] * stride + idx[rank()] * dilation;
                } else {
                    srcIdx[j] = idx[j];
                }
            }
            result[(int) i] = get(srcIdx);
        }
        return new RereFloatTensor(result, resultShape);
    }

    @Override
    public IFloatTensor nonzero() {
        java.util.ArrayList<Long> positions = new java.util.ArrayList<>();
        int n = (int) totalSize();
        for (int i = 0; i < n; i++) {
            if (linearGet(i) != 0) {
                positions.add((long) i);
            }
        }
        float[] result = new float[positions.size() * rank()];
        int idx = 0;
        for (long pos : positions) {
            int[] indices = unlinearize((int) pos, shape());
            for (int j = 0; j < rank(); j++) {
                result[idx++] = indices[j];
            }
        }
        return new RereFloatTensor(result, positions.size(), rank());
    }

    @Override
    public IFloatTensor maskedSelect(IFloatTensor mask) {
        java.util.ArrayList<Float> selected = new java.util.ArrayList<>();
        for (int i = 0; i < (int) totalSize(); i++) {
            int[] idx = unlinearize(i, shape());
            if (mask.get(idx) > 0.5) {
                selected.add(linearGet(i));
            }
        }
        float[] result = new float[selected.size()];
        for (int i = 0; i < selected.size(); i++) result[i] = selected.get(i);
        return new RereFloatTensor(result, result.length);
    }

    @Override
    public IFloatTensor maskedFill(IFloatTensor mask, float value) {
        float[] result = toFloatArray();
        for (int i = 0; i < (int) totalSize(); i++) {
            int[] idx = unlinearize(i, shape());
            if (mask.get(idx) > 0.5) {
                result[i] = value;
            }
        }
        return new RereFloatTensor(result, shape());
    }

    @Override
    public IFloatTensor cat(int dim, IFloatTensor... others) {
        if (dim < 0) dim += rank();
        int totalDim = dim(dim);
        for (IFloatTensor t : others) totalDim += t.dim(dim);
        int[] newShape = shape().clone();
        newShape[dim] = totalDim;
        float[] result = new float[(int) computeSize(newShape)];
        // Copy this tensor
        copyInto(result, this, newShape, 0, dim);
        int offset = dim(dim);
        for (IFloatTensor t : others) {
            copyInto(result, t, newShape, offset, dim);
            offset += t.dim(dim);
        }
        return new RereFloatTensor(result, newShape);
    }

    private void copyInto(float[] target, IFloatTensor src, int[] targetShape,
                           int dimOffset, int dim) {
        for (long i = 0; i < src.totalSize(); i++) {
            int[] srcIdx = unlinearize((int) i, src.shape());
            int[] tgtIdx = new int[targetShape.length];
            for (int j = 0; j < targetShape.length; j++) {
                tgtIdx[j] = j == dim ? srcIdx[j] + dimOffset : srcIdx[j];
            }
            long linear = linearIndex(tgtIdx, targetShape);
            target[(int) linear] = ((RereFloatTensor) src).linearGet(i);
        }
    }

    @Override
    public IFloatTensor stack(int dim, IFloatTensor... tensors) {
        if (dim < 0) dim = rank() + 1;
        int[] elemShape = shape();
        for (IFloatTensor t : tensors) {
            if (!Arrays.equals(t.shape(), elemShape)) {
                throw new IllegalArgumentException("stack: all tensors must have same shape");
            }
        }
        int[] newShape = new int[rank() + 1];
        for (int i = 0; i < dim; i++) newShape[i] = elemShape[i];
        newShape[dim] = 1 + tensors.length;
        for (int i = dim + 1; i < newShape.length; i++) {
            newShape[i] = elemShape[i - 1];
        }
        // Concatenate along new axis by first unsqueezing each, then cat
        IFloatTensor[] all = new IFloatTensor[1 + tensors.length];
        all[0] = unsqueeze(dim);
        for (int i = 0; i < tensors.length; i++) {
            all[i + 1] = tensors[i].unsqueeze(dim);
        }
        return all[0].cat(dim, java.util.Arrays.copyOfRange(all, 1, all.length));
    }

    @Override
    public List<IFloatTensor> unstack(int dim) {
        if (dim < 0) dim += rank();
        int n = dim(dim);
        List<IFloatTensor> result = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            result.add(select(dim, i));
        }
        return result;
    }

    @Override
    public IFloatTensor normalize(float p, int dim) {
        if (dim < 0) dim += rank();
        IFloatTensor norm = abs().pow(p).sum(dim, true).pow(1.0f / p);
        return div(norm);
    }

    // ==================== 就地操作 ====================

    @Override
    public IFloatTensor add_(IFloatTensor other) {
        int n = (int) totalSize();
        for (int i = 0; i < n; i++) {
            linearSet(i, linearGet(i) + ((RereFloatTensor) other).linearGet(i));
        }
        return this;
    }

    @Override
    public IFloatTensor sub_(IFloatTensor other) {
        int n = (int) totalSize();
        for (int i = 0; i < n; i++) {
            linearSet(i, linearGet(i) - ((RereFloatTensor) other).linearGet(i));
        }
        return this;
    }

    @Override
    public IFloatTensor mul_(IFloatTensor other) {
        int n = (int) totalSize();
        for (int i = 0; i < n; i++) {
            linearSet(i, linearGet(i) * ((RereFloatTensor) other).linearGet(i));
        }
        return this;
    }

    @Override
    public IFloatTensor div_(IFloatTensor other) {
        int n = (int) totalSize();
        for (int i = 0; i < n; i++) {
            linearSet(i, linearGet(i) / ((RereFloatTensor) other).linearGet(i));
        }
        return this;
    }

    @Override
    public IFloatTensor fill_(float value) {
        fill(value);
        return this;
    }

    @Override public IFloatTensor neg_() {
        int n = (int) totalSize();
        if (isContiguous() && offset == 0) {
            for (int i = 0; i < n; i++) data[i] = -data[i];
        } else {
            for (int i = 0; i < n; i++) linearSet(i, -linearGet(i));
        }
        return this;
    }

    @Override public IFloatTensor abs_() {
        int n = (int) totalSize();
        if (isContiguous() && offset == 0) {
            for (int i = 0; i < n; i++) { if (data[i] < 0) data[i] = -data[i]; }
        } else {
            for (int i = 0; i < n; i++) { float v = linearGet(i); if (v < 0) linearSet(i, -v); }
        }
        return this;
    }

    @Override public IFloatTensor relu_() {
        int n = (int) totalSize();
        if (isContiguous() && offset == 0) {
            for (int i = 0; i < n; i++) { if (data[i] < 0) data[i] = 0; }
        } else {
            for (int i = 0; i < n; i++) { float v = linearGet(i); if (v < 0) linearSet(i, 0); }
        }
        return this;
    }

    @Override public IFloatTensor sigmoid_() {
        int n = (int) totalSize();
        if (isContiguous() && offset == 0) {
            for (int i = 0; i < n; i++) data[i] = (float) (1.0 / (1.0 + Math.exp(-data[i])));
        } else {
            for (int i = 0; i < n; i++) {
                linearSet(i, (float) (1.0 / (1.0 + Math.exp(-linearGet(i)))));
            }
        }
        return this;
    }

    @Override public IFloatTensor tanh_() {
        int n = (int) totalSize();
        if (isContiguous() && offset == 0) {
            for (int i = 0; i < n; i++) data[i] = (float) Math.tanh(data[i]);
        } else {
            for (int i = 0; i < n; i++) linearSet(i, (float) Math.tanh(linearGet(i)));
        }
        return this;
    }

    @Override
    public IFloatTensor copy_(IFloatTensor src) {
        if (totalSize() != src.totalSize()) {
            throw new IllegalArgumentException("copy_ size mismatch");
        }
        int n = (int) totalSize();
        for (int i = 0; i < n; i++) {
            linearSet(i, ((RereFloatTensor) src).linearGet(i));
        }
        return this;
    }

    // ==================== 转换桥接 ====================

    @Override
    public IFloatVector toVector() {
        if (isContiguous() && offset == 0 && rank() == 1) {
            return IFloatVector.of(data);
        }
        return IFloatVector.of(toFloatArray());
    }

    @Override
    public IFloatVector toVectorCopy() {
        return IFloatVector.of(toFloatArray());
    }

    @Override
    public IFloatMatrix toMatrix() {
        if (rank() != 2) {
            throw new IllegalStateException("toMatrix requires 2D tensor, got rank " + rank());
        }
        return Linalg.fromArray(toFloatArray(), dim(0), dim(1));
    }

    @Override
    public IDoubleTensor toDoubleTensor() {
        int n = (int) totalSize();
        double[] dData = new double[n];
        float[] fData = toFloatArray();
        for (int i = 0; i < n; i++) dData[i] = fData[i];
        return new RereDoubleTensor(dData, shape());
    }

    @Override
    public IFloatTensor copy() {
        return clone();
    }

    @Override
    public IFloatTensor clone() {
        return new RereFloatTensor(toFloatArray(), shape());
    }

    @Override
    public String toString() {
        if (rank() == 1) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < Math.min(dim(0), 20); i++) {
                sb.append(String.format("%.4f", get(i)));
                if (i < dim(0) - 1 && i < 19) sb.append(", ");
            }
            if (dim(0) > 20) sb.append(", ...");
            sb.append("]");
            return sb.toString();
        } else if (rank() == 2) {
            StringBuilder sb = new StringBuilder("[\n");
            for (int i = 0; i < dim(0); i++) {
                sb.append("  [");
                for (int j = 0; j < Math.min(dim(1), 12); j++) {
                    sb.append(String.format("%.4f", get(i, j)));
                    if (j < dim(1) - 1 && j < 11) sb.append(", ");
                }
                if (dim(1) > 12) sb.append(", ...");
                sb.append("]");
                if (i < dim(0) - 1) sb.append(",\n");
            }
            sb.append("\n]");
            return sb.toString();
        } else {
            StringBuilder sb = new StringBuilder("Tensor(");
            sb.append(Arrays.toString(shape()));
            sb.append(") {\n  data: [");
            int show = Math.min((int) totalSize(), 30);
            for (int i = 0; i < show; i++) {
                int[] idx = unlinearize(i, shape());
                sb.append(String.format("%.4f", get(idx)));
                if (i < show - 1) sb.append(", ");
            }
            if (totalSize() > 30) sb.append(", ...");
            sb.append("]\n}");
            return sb.toString();
        }
    }

    // ==================== 内部工具 ====================

    private static long computeSize(int[] shape) {
        long size = 1;
        for (int d : shape) size *= d;
        return size;
    }

    private static void validateShape(long dataLen, int[] shape) {
        long expected = 1;
        for (int d : shape) expected *= d;
        if (expected != dataLen) {
            throw new IllegalArgumentException("Data length " + dataLen + " != shape size " + expected);
        }
    }

    private static int[] unlinearize(long flat, int[] shape) {
        int[] idx = new int[shape.length];
        long remaining = flat;
        for (int j = shape.length - 1; j >= 0; j--) {
            idx[j] = (int) (remaining % shape[j]);
            remaining /= shape[j];
        }
        return idx;
    }

    private static long linearIndex(int[] indices, int[] shape) {
        long idx = 0;
        long stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            idx += indices[i] * stride;
            stride *= shape[i];
        }
        return idx;
    }

    private static int[] mixedIndex(int o, int dim, int r, int i, int[] shape) {
        int[] idx = new int[shape.length];
        int tmp = o;
        for (int j = 0; j < dim; j++) {
            idx[j] = tmp % shape[j];
            tmp /= shape[j];
        }
        idx[dim] = r;
        tmp = i;
        for (int j = shape.length - 1; j > dim; j--) {
            idx[j] = tmp % shape[j];
            tmp /= shape[j];
        }
        return idx;
    }

    /** 带广播的 get：对每一维取模 source 的对应大小 */
    private static float getWithBroadcast(int[] idx, IFloatTensor tensor) {
        int[] shapedIdx = new int[tensor.rank()];
        int diff = idx.length - tensor.rank();
        for (int i = 0; i < tensor.rank(); i++) {
            int sourceDim = tensor.dim(i);
            shapedIdx[i] = sourceDim == 1 ? 0 : idx[diff + i];
        }
        return tensor.get(shapedIdx);
    }
}
