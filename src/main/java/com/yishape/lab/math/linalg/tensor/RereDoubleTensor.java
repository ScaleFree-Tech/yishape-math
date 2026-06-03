package com.yishape.lab.math.linalg.tensor;

import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.FlatGemm;
import com.yishape.lab.math.compute.IDoubleVectorComputer;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * 基于 stride 的双精度张量实现.
 * <p>
 * 核心：flat double[] + TensorShape + int[] strides + int offset.
 * 视图操作（permute/slice/expand）通过修改 strides/offset 实现零拷贝.
 */
public class RereDoubleTensor implements IDoubleTensor {

    // ==================== 字段 ====================

    protected final double[] data;
    protected TensorShape shape;
    protected int[] strides;
    protected int offset;
    private static final DoubleVectorComputer COMPUTER = new DoubleVectorComputer();

    // ==================== 构造方法 ====================

    /** 主构造：新分配连续数据 */
    public RereDoubleTensor(double[] data, int... shape) {
        validateShape(data.length, shape);
        this.data = data;
        this.shape = new TensorShape(shape);
        this.strides = TensorShape.computeCStrides(shape);
        this.offset = 0;
    }

    /** 视图构造：共享数据 */
    RereDoubleTensor(double[] data, int offset, int[] shape, int[] strides) {
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
    RereDoubleTensor(double[] data, int offset, TensorShape shape) {
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
    public double item() {
        if (totalSize() != 1) {
            throw new IllegalStateException("item() requires exactly 1 element, got " + totalSize());
        }
        return data[offset];
    }

    // ==================== 元素访问 ====================

    @Override
    public double get(int... indices) {
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
    public ITensor set(double value, int... indices) {
        long idx = offset;
        for (int i = 0; i < indices.length; i++) {
            idx += indices[i] * (long) strides[i];
        }
        data[(int) idx] = value;
        return this;
    }

    @Override
    public ITensor fill(double value) {
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
    public double[] toDoubleArray() {
        if (isContiguous() && offset == 0 && totalSize() == data.length) {
            return data.clone();
        }
        int n = (int) totalSize();
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = linearGet(i);
        }
        return out;
    }

    /**
     * Return data as a contiguous row-major flat array.
     * If already contiguous with offset==0, returns a clone of the internal data.
     * Otherwise materializes via toDoubleArray().
     */
    private double[] toContiguousFlat() {
        if (isContiguous() && offset == 0 && (int) totalSize() == data.length) {
            return data.clone();
        }
        return toDoubleArray();
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

    protected double linearGet(long flatIndex) {
        return data[(int) linearToOffset(flatIndex)];
    }

    protected void linearSet(long flatIndex, double value) {
        data[(int) linearToOffset(flatIndex)] = value;
    }

    // ==================== 视图操作（零拷贝） ====================

    @Override
    public IDoubleTensor permute(int... dims) {
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
        return new RereDoubleTensor(data, offset, newShape, newStrides);
    }

    @Override
    public IDoubleTensor transpose(int dim0, int dim1) {
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
    public IDoubleTensor transpose() {
        if (rank() != 2) {
            throw new UnsupportedOperationException("transpose() requires 2D tensor");
        }
        return transpose(0, 1);
    }

    @Override
    public IDoubleTensor squeeze(int... dims) {
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
                return new RereDoubleTensor(data, offset, new int[]{1}, new int[]{1});
            }
            return new RereDoubleTensor(data, offset, newShape, newStrides);
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
            return new RereDoubleTensor(data, offset, new int[]{1}, new int[]{1});
        }
        return new RereDoubleTensor(data, offset, newShape, newStrides);
    }

    @Override
    public IDoubleTensor unsqueeze(int axis) {
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
        return new RereDoubleTensor(data, offset, newShape, newStrides);
    }

    @Override
    public IDoubleTensor slice(int dim, long start, long end) {
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
        return new RereDoubleTensor(data, newOffset, newShape, strides);
    }

    @Override
    public IDoubleTensor narrow(int dim, long start, long length) {
        return slice(dim, start, start + length);
    }

    @Override
    public IDoubleTensor select(int dim, long index) {
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
            return new RereDoubleTensor(data, newOffset, new int[]{1}, new int[]{1});
        }
        return new RereDoubleTensor(data, newOffset, newShape, newStrides);
    }

    @Override
    public IDoubleTensor flatten(int startDim, int endDim) {
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
        return new RereDoubleTensor(data, offset, newShape, newStrides);
    }

    @Override
    public IDoubleTensor expand(int... shape) {
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
        return new RereDoubleTensor(data, offset, newShape, newStrides);
    }

    @Override
    public IDoubleTensor contiguous() {
        if (isContiguous() && offset == 0 && totalSize() == data.length) {
            return this;
        }
        int n = (int) totalSize();
        double[] pooled = ContiguousPool.acquire(n);
        for (int i = 0; i < n; i++) {
            pooled[i] = linearGet(i);
        }
        if (pooled.length == n) {
            return new RereDoubleTensor(pooled, shape());
        }
        double[] newData = new double[n];
        System.arraycopy(pooled, 0, newData, 0, n);
        ContiguousPool.release(pooled);
        return new RereDoubleTensor(newData, shape());
    }

    @Override
    public IDoubleTensor tile(int... repeats) {
        if (repeats.length != rank()) {
            throw new IllegalArgumentException("tile: repeats length " + repeats.length + " != rank " + rank());
        }
        int[] resultShape = new int[rank()];
        long total = 1;
        for (int i = 0; i < rank(); i++) {
            resultShape[i] = dim(i) * repeats[i];
            total *= resultShape[i];
        }
        double[] result = new double[(int) total];
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
        return new RereDoubleTensor(result, resultShape);
    }

    @Override
    public IDoubleTensor broadcastTo(int... targetShape) {
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
        return new RereDoubleTensor(data, offset, resultShape, resultStrides);
    }

    @Override
    public IDoubleTensor reshape(int... newShape) {
        long newSize = 1;
        newShape = shape.inferReshape(newShape);
        for (int d : newShape) newSize *= d;
        if (newSize != totalSize()) {
            throw new IllegalArgumentException("Cannot reshape " + totalSize() + " to " + newSize);
        }
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(data, offset, newShape,
                TensorShape.computeCStrides(newShape));
        }
        // 非连续：materialize 后再 reshape
        return contiguous().reshape(newShape);
    }

    // ==================== 逐元素运算（一元） ====================

    private IDoubleTensor applyUnary(DoubleUnaryOperator op) {
        int n = (int) totalSize();
        if (isContiguous() && offset == 0 && n == data.length) {
            double[] result = new double[n];
            for (int i = 0; i < n; i++) result[i] = op.applyAsDouble(data[i]);
            return new RereDoubleTensor(result, shape());
        }
        double[] result = new double[n];
        for (int i = 0; i < n; i++) result[i] = op.applyAsDouble(linearGet(i));
        return new RereDoubleTensor(result, shape());
    }

    @Override public IDoubleTensor neg() {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.negate(data), shape());
        }
        return applyUnary(v -> -v);
    }
    @Override public IDoubleTensor abs() {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.universalOperate(data, IDoubleVectorComputer.UniversalOperation.ABS, 0), shape());
        }
        return applyUnary(Math::abs);
    }
    @Override public IDoubleTensor sqrt() {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.universalOperate(data, IDoubleVectorComputer.UniversalOperation.SQRT, 0), shape());
        }
        return applyUnary(Math::sqrt);
    }
    @Override public IDoubleTensor exp() {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.universalOperate(data, IDoubleVectorComputer.UniversalOperation.EXP, 0), shape());
        }
        return applyUnary(Math::exp);
    }
    @Override public IDoubleTensor log() {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.universalOperate(data, IDoubleVectorComputer.UniversalOperation.LOG, 0), shape());
        }
        return applyUnary(Math::log);
    }
    @Override public IDoubleTensor sin() {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.universalOperate(data, IDoubleVectorComputer.UniversalOperation.SIN, 0), shape());
        }
        return applyUnary(Math::sin);
    }
    @Override public IDoubleTensor cos() {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.universalOperate(data, IDoubleVectorComputer.UniversalOperation.COS, 0), shape());
        }
        return applyUnary(Math::cos);
    }
    @Override public IDoubleTensor tan() {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.universalOperate(data, IDoubleVectorComputer.UniversalOperation.TAN, 0), shape());
        }
        return applyUnary(Math::tan);
    }
    @Override public IDoubleTensor square() { return applyUnary(v -> v * v); }
    @Override public IDoubleTensor sigmoid() {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.universalOperate(data, IDoubleVectorComputer.UniversalOperation.SIGMOID, 0), shape());
        }
        return applyUnary(v -> 1.0 / (1.0 + Math.exp(-v)));
    }
    @Override public IDoubleTensor relu() {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.universalOperate(data, IDoubleVectorComputer.UniversalOperation.RELU, 0), shape());
        }
        return applyUnary(v -> v > 0 ? v : 0);
    }
    @Override public IDoubleTensor tanh() {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.universalOperate(data, IDoubleVectorComputer.UniversalOperation.TANH, 0), shape());
        }
        return applyUnary(Math::tanh);
    }
    @Override public IDoubleTensor gelu() {
        return applyUnary(x -> {
            double cdf = 0.5 * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
            return x * cdf;
        });
    }
    @Override public IDoubleTensor leakyRelu(double alpha) {
        return applyUnary(x -> x >= 0 ? x : alpha * x);
    }
    @Override public IDoubleTensor elu(double alpha) {
        return applyUnary(x -> x >= 0 ? x : alpha * (Math.exp(x) - 1));
    }
    @Override public IDoubleTensor selu() {
        double alpha = 1.6732632423543772, scale = 1.0507009873554804;
        return applyUnary(x -> scale * (x >= 0 ? x : alpha * (Math.exp(x) - 1)));
    }
    @Override public IDoubleTensor silu() {
        return applyUnary(x -> x / (1.0 + Math.exp(-x)));
    }
    @Override public IDoubleTensor mish() {
        return applyUnary(x -> x * Math.tanh(Math.log(1.0 + Math.exp(x))));
    }
    @Override public IDoubleTensor softplus(double beta) {
        return applyUnary(x -> {
            double bx = beta * x;
            return bx > 20 ? x : Math.log(1.0 + Math.exp(bx)) / beta;
        });
    }
    @Override public IDoubleTensor hardtanh(double minVal, double maxVal) {
        return applyUnary(x -> Math.min(Math.max(x, minVal), maxVal));
    }
    @Override public IDoubleTensor pow(double n) { return applyUnary(v -> Math.pow(v, n)); }

    @Override
    public IDoubleTensor clamp(double min, double max) {
        return applyUnary(v -> Math.max(min, Math.min(max, v)));
    }

    // ==================== 逐元素运算（标量） ====================

    @Override public IDoubleTensor add(double scalar) {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.binaryOperate(data, scalar, IDoubleVectorComputer.BinaryOperation.ADD), shape());
        }
        return applyUnary(v -> v + scalar);
    }
    @Override public IDoubleTensor sub(double scalar) {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.binaryOperate(data, scalar, IDoubleVectorComputer.BinaryOperation.SUBTRACT), shape());
        }
        return applyUnary(v -> v - scalar);
    }
    @Override public IDoubleTensor mul(double scalar) {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.binaryOperate(data, scalar, IDoubleVectorComputer.BinaryOperation.MULTIPLY), shape());
        }
        return applyUnary(v -> v * scalar);
    }
    @Override public IDoubleTensor div(double scalar) {
        if (isContiguous() && offset == 0) {
            return new RereDoubleTensor(COMPUTER.binaryOperate(data, scalar, IDoubleVectorComputer.BinaryOperation.DIVIDE), shape());
        }
        return applyUnary(v -> v / scalar);
    }

    // ==================== 逐元素运算（张量 + 广播） ====================

    @Override
    public IDoubleTensor add(IDoubleTensor other) {
        if (other instanceof RereDoubleTensor rt && isContiguous() && offset == 0
                && rt.isContiguous() && rt.offset == 0 && data.length == rt.data.length
                && java.util.Arrays.equals(shape(), rt.shape())) {
            return new RereDoubleTensor(COMPUTER.binaryOperate(data, rt.data, IDoubleVectorComputer.BinaryOperation.ADD), shape());
        }
        return elementWiseBinary(other, Double::sum);
    }

    @Override
    public IDoubleTensor sub(IDoubleTensor other) {
        if (other instanceof RereDoubleTensor rt && isContiguous() && offset == 0
                && rt.isContiguous() && rt.offset == 0 && data.length == rt.data.length
                && java.util.Arrays.equals(shape(), rt.shape())) {
            return new RereDoubleTensor(COMPUTER.binaryOperate(data, rt.data, IDoubleVectorComputer.BinaryOperation.SUBTRACT), shape());
        }
        return elementWiseBinary(other, (a, b) -> a - b);
    }

    @Override
    public IDoubleTensor mul(IDoubleTensor other) {
        if (other instanceof RereDoubleTensor rt && isContiguous() && offset == 0
                && rt.isContiguous() && rt.offset == 0 && data.length == rt.data.length
                && java.util.Arrays.equals(shape(), rt.shape())) {
            return new RereDoubleTensor(COMPUTER.binaryOperate(data, rt.data, IDoubleVectorComputer.BinaryOperation.MULTIPLY), shape());
        }
        return elementWiseBinary(other, (a, b) -> a * b);
    }

    @Override
    public IDoubleTensor div(IDoubleTensor other) {
        if (other instanceof RereDoubleTensor rt && isContiguous() && offset == 0
                && rt.isContiguous() && rt.offset == 0 && data.length == rt.data.length
                && java.util.Arrays.equals(shape(), rt.shape())) {
            return new RereDoubleTensor(COMPUTER.binaryOperate(data, rt.data, IDoubleVectorComputer.BinaryOperation.DIVIDE), shape());
        }
        return elementWiseBinary(other, (a, b) -> a / b);
    }

    private IDoubleTensor elementWiseBinary(IDoubleTensor other, DoubleBinaryOperator op) {
        int[] shapeA = shape();
        int[] shapeB = other.shape();
        int[] resultShape = TensorShape.broadcastShape(shapeA, shapeB);
        long total = 1;
        for (int d : resultShape) total *= d;
        double[] result = new double[(int) total];
        int rank = resultShape.length;

        // 对齐两个输入到结果秩
        RereDoubleTensor aAligned = (RereDoubleTensor) broadcastTo(resultShape);
        RereDoubleTensor bAligned = (RereDoubleTensor) other.broadcastTo(resultShape);

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
            result[(int) i] = op.applyAsDouble(aAligned.data[(int) aOff], bAligned.data[(int) bOff]);
        }
        return new RereDoubleTensor(result, resultShape);
    }

    // ==================== 归约 ====================

    @Override
    public IDoubleTensor sum(int dim, boolean keepdim) {
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
        double[] result = new double[(int) computeSize(newShape)];

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double sum = 0;
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
        return new RereDoubleTensor(result, newShape);
    }

    @Override
    public IDoubleTensor mean(int dim, boolean keepdim) {
        IDoubleTensor s = sum(dim, keepdim);
        return s.div(this.dim(dim));
    }

    @Override
    public IDoubleTensor max(int dim, boolean keepdim) {
        if (dim < 0) dim += rank();
        return reduceDim(dim, keepdim, Double.NEGATIVE_INFINITY,
            (a, b) -> a > b ? a : b);
    }

    @Override
    public IDoubleTensor min(int dim, boolean keepdim) {
        if (dim < 0) dim += rank();
        return reduceDim(dim, keepdim, Double.POSITIVE_INFINITY,
            (a, b) -> a < b ? a : b);
    }

    @Override
    public IDoubleTensor prod(int dim, boolean keepdim) {
        if (dim < 0) dim += rank();
        return reduceDim(dim, keepdim, 1.0, (a, b) -> a * b);
    }

    private IDoubleTensor reduceDim(int dim, boolean keepdim,
                                     double init, DoubleBinaryOperator reducer) {
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
        double[] result = new double[(int) computeSize(newShape)];
        Arrays.fill(result, init);

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double val = init;
                for (int r = 0; r < reduce; r++) {
                    val = reducer.applyAsDouble(val, getStrided(o, dim, r, i));
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
        return new RereDoubleTensor(result, newShape);
    }

    /** 使用 stride 获取元素（o=outer index, dim=target, r=reduce index, i=inner index） */
    private double getStrided(int o, int dim, int r, int i) {
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
        long idx = 0;
        int stride = 1;
        for (int j = shape.length - 1; j >= 0; j--) {
            int v = (j == dim) ? 0 : (j < dim ? o % dim(j) : i % dim(j));
            idx += v * stride;
            stride *= shape[j];
        }
        return idx;
    }

    private long flatIndexReduce(int o, int dim, int i, int[] shape) {
        long idx = 0;
        int stride = 1;
        for (int j = shape.length - 1; j >= 0; j--) {
            int actualDim = (j < dim) ? j : j + 1;
            int v = (actualDim < dim) ? o % dim(actualDim) : i % dim(actualDim);
            idx += v * stride;
            stride *= shape[j];
        }
        return idx;
    }

    @Override
    public double sumAll() {
        return reduceAll(0.0, Double::sum);
    }

    @Override
    public double meanAll() { return sumAll() / totalSize(); }

    @Override
    public double maxAll() {
        return reduceAll(Double.NEGATIVE_INFINITY, Math::max);
    }

    @Override
    public double minAll() {
        return reduceAll(Double.POSITIVE_INFINITY, Math::min);
    }

    @Override
    public double prodAll() { return reduceAll(1.0, (a, b) -> a * b); }

    private double reduceAll(double init, DoubleBinaryOperator op) {
        int n = (int) totalSize();
        double result = init;
        for (int i = 0; i < n; i++) {
            result = op.applyAsDouble(result, linearGet(i));
        }
        return result;
    }

    @Override
    public IDoubleTensor argmax(int dim) {
        // argmax returns indices as double tensor
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
        double[] result = new double[(int) computeSize(newShape)];
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dim(i);
        int reduce = dim(dim);
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= dim(i);

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double maxVal = Double.NEGATIVE_INFINITY;
                int maxIdx = 0;
                for (int r = 0; r < reduce; r++) {
                    double v = getStrided(o, dim, r, i);
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
        return new RereDoubleTensor(result, newShape);
    }

    @Override
    public IDoubleTensor argmin(int dim) {
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
        double[] result = new double[(int) computeSize(newShape)];
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dim(i);
        int reduce = dim(dim);
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= dim(i);

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double minVal = Double.POSITIVE_INFINITY;
                int minIdx = 0;
                for (int r = 0; r < reduce; r++) {
                    double v = getStrided(o, dim, r, i);
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
        return new RereDoubleTensor(result, newShape);
    }

    @Override
    public IDoubleTensor cumsum(int dim) {
        // cumulative sum along dim
        int[] resultShape = shape();
        double[] result = new double[(int) totalSize()];
        int n = dim(dim);
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dim(i);
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= dim(i);

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double cum = 0;
                for (int r = 0; r < n; r++) {
                    cum += getStrided(o, dim, r, i);
                    int[] outIdx = mixedIndex(o, dim, r, i, resultShape);
                    long linear = linearIndex(outIdx, resultShape);
                    result[(int) linear] = cum;
                }
            }
        }
        return new RereDoubleTensor(result, resultShape);
    }

    @Override
    public IDoubleTensor cumprod(int dim) {
        int[] resultShape = shape();
        double[] result = new double[(int) totalSize()];
        int n = dim(dim);
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dim(i);
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= dim(i);

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double cum = 1;
                for (int r = 0; r < n; r++) {
                    cum *= getStrided(o, dim, r, i);
                    int[] outIdx = mixedIndex(o, dim, r, i, resultShape);
                    long linear = linearIndex(outIdx, resultShape);
                    result[(int) linear] = cum;
                }
            }
        }
        return new RereDoubleTensor(result, resultShape);
    }

    @Override
    public IDoubleTensor std(int dim, boolean keepdim) {
        IDoubleTensor m = mean(dim, keepdim);
        int n = dim(dim);
        // broadcast mean back, compute variance
        IDoubleTensor centered = sub(m);
        IDoubleTensor sq = centered.pow(2);
        IDoubleTensor var = sq.sum(dim, keepdim).div(n - 1); // unbiased
        return var.sqrt();
    }

    @Override
    public IDoubleTensor var(int dim, boolean keepdim) {
        IDoubleTensor m = mean(dim, keepdim);
        int n = dim(dim);
        IDoubleTensor centered = sub(m);
        IDoubleTensor sq = centered.pow(2);
        return sq.sum(dim, keepdim).div(n - 1);
    }

    // ==================== Softmax / LogSoftmax ====================

    @Override
    public IDoubleTensor softmax(int dim) {
        if (dim < 0) dim += rank();
        IDoubleTensor maxVals = max(dim, true);
        IDoubleTensor shifted = sub(maxVals);
        IDoubleTensor exps = shifted.exp();
        IDoubleTensor sumExps = exps.sum(dim, true);
        return exps.div(sumExps);
    }

    @Override
    public IDoubleTensor logSoftmax(int dim) {
        if (dim < 0) dim += rank();
        IDoubleTensor maxVals = max(dim, true);
        IDoubleTensor shifted = sub(maxVals);
        IDoubleTensor exps = shifted.exp();
        IDoubleTensor sumExps = exps.sum(dim, true);
        return shifted.sub(sumExps.log());
    }

    // ==================== 线性代数 ====================

    @Override
    public IDoubleTensor mmul(IDoubleTensor other) {
        if (rank() != 2 || other.rank() != 2) {
            throw new UnsupportedOperationException("mmul only supports 2D tensors, use bmm for batched");
        }
        int m = dim(0), k = dim(1), n = other.dim(1);
        if (dim(1) != other.dim(0)) {
            throw new IllegalArgumentException("mmul: " + dim(0) + "x" + dim(1)
                + " @ " + other.dim(0) + "x" + other.dim(1) + " incompatible");
        }
        double[] aFlat = toContiguousFlat();
        double[] bFlat = ((RereDoubleTensor) other).toContiguousFlat();
        double[] result = FlatGemm.flatMmul(aFlat, m, k, bFlat, n);
        return new RereDoubleTensor(result, m, n);
    }

    @Override
    public IDoubleTensor bmm(IDoubleTensor other) {
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
        double[] result = new double[(int) (batchTotal * m * n)];

        boolean bothContig = isContiguous() && other.isContiguous()
            && offset == 0 && ((RereDoubleTensor) other).offset == 0;
        if (bothContig && batchTotal > 0) {
            double[] aFull = data.clone();
            double[] bFull = ((RereDoubleTensor) other).data.clone();
            for (long b = 0; b < batchTotal; b++) {
                int aOff = (int) (b * m * k);
                int bOff = (int) (b * k * n);
                int cOff = (int) (b * m * n);
                double[] cSlice = FlatGemm.flatMmul(aFull, aOff, m, k, bFull, bOff, n);
                System.arraycopy(cSlice, 0, result, cOff, m * n);
            }
        } else {
            for (long b = 0; b < batchTotal; b++) {
                int off = (int) (b * m * n);
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0;
                        for (int t = 0; t < k; t++) {
                            RereDoubleTensor otherRDT = (RereDoubleTensor) other;
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
        return new RereDoubleTensor(result, resultShape);
    }

    private double getWithBatch(long batchFlat, int[] batchShape, int dim0Idx, int dim1Idx) {
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
    public IDoubleTensor einsum(String subscript, IDoubleTensor... others) {
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
            IDoubleTensor a = this;
            IDoubleTensor b = others[0];
            int B = a.dim(0), Q = a.dim(1), H = a.dim(2), D = a.dim(3);
            int K = b.dim(1);
            // Permute a: (B,Q,H,D) -> (B,H,Q,D)
            // Permute b: (B,K,H,D) -> (B,H,K,D)
            // bmm: (B,H,Q,D) @ (B,H,D,K) -> (B,H,Q,K)
            IDoubleTensor aP = a.permute(0, 2, 1, 3);
            IDoubleTensor bP = b.permute(0, 2, 3, 1);
            IDoubleTensor result = aP.bmm(bP);
            return result.permute(0, 2, 1, 3).reshape(B, H, Q, K);
        }

        throw new UnsupportedOperationException("einsum subscript not yet implemented: " + subscript);
    }

    // ==================== 高级操作 ====================

    @Override
    public IDoubleTensor gather(int dim, IDoubleTensor index) {
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
        double[] result = new double[resultSize];
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
        return new RereDoubleTensor(result, resultShape);
    }

    @Override
    public IDoubleTensor scatter(int dim, IDoubleTensor index, IDoubleTensor source) {
        if (dim < 0) dim += rank();
        double[] result = toDoubleArray();
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
        return new RereDoubleTensor(result, resultShape);
    }

    @Override
    public IDoubleTensor scatterAdd(int dim, IDoubleTensor index, IDoubleTensor source) {
        if (dim < 0) dim += rank();
        double[] result = toDoubleArray();
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
        return new RereDoubleTensor(result, resultShape);
    }

    @Override
    public IDoubleTensor indexSelect(int dim, IDoubleTensor index) {
        return gather(dim, index);
    }

    @Override
    public IDoubleTensor argsort(int dim, boolean descending) {
        int d = dim < 0 ? dim + rank() : dim;
        int[] s = shape();
        int dimSize = s[d];
        int outerTotal = 1;
        for (int i = 0; i < d; i++) outerTotal *= s[i];
        int innerTotal = 1;
        for (int i = d + 1; i < rank(); i++) innerTotal *= s[i];

        double[] inData = toDoubleArray();
        double[] outData = new double[inData.length];

        for (int outer = 0; outer < outerTotal; outer++) {
            for (int inner = 0; inner < innerTotal; inner++) {
                int baseStride = outer * dimSize * innerTotal + inner;
                int[] origIdx = new int[dimSize];
                double[] sliceVals = new double[dimSize];
                for (int i = 0; i < dimSize; i++) {
                    origIdx[i] = i;
                    sliceVals[i] = inData[baseStride + i * innerTotal];
                }
                for (int i = 1; i < dimSize; i++) {
                    int ki = origIdx[i];
                    double kv = sliceVals[i];
                    int j = i - 1;
                    while (j >= 0 && (descending ? sliceVals[j] < kv : sliceVals[j] > kv)) {
                        origIdx[j + 1] = origIdx[j];
                        sliceVals[j + 1] = sliceVals[j];
                        j--;
                    }
                    origIdx[j + 1] = ki;
                    sliceVals[j + 1] = kv;
                }
                for (int i = 0; i < dimSize; i++) {
                    outData[baseStride + i * innerTotal] = origIdx[i];
                }
            }
        }
        return new RereDoubleTensor(outData, s);
    }

    @Override
    public IDoubleTensor where(IDoubleTensor condition, IDoubleTensor other) {
        int[] resultShape = TensorShape.broadcastShape(
            shape(), TensorShape.broadcastShape(other.shape(), condition.shape()));
        double[] result = new double[(int) computeSize(resultShape)];
        for (long i = 0; i < result.length; i++) {
            int[] idx = unlinearize(i, resultShape);
            double condVal = getWithBroadcast(idx, condition);
            result[(int) i] = condVal > 0.5 ?
                getWithBroadcast(idx, this) : getWithBroadcast(idx, other);
        }
        return new RereDoubleTensor(result, resultShape);
    }

    @Override
    public IDoubleTensor topk(int k, int dim, boolean largest) {
        if (dim < 0) dim += rank();
        int n = dim(dim);
        int[] resultShape = shape().clone();
        resultShape[dim] = k;
        double[] values = new double[(int) computeSize(resultShape)];
        double[] indices = new double[(int) computeSize(resultShape)];
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= dim(i);
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= dim(i);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                // collect all values along dim
                double[] vals = new double[n];
                for (int r = 0; r < n; r++) {
                    vals[r] = getStrided(o, dim, r, i);
                }
                // argsort
                Integer[] idxs = new Integer[n];
                for (int r = 0; r < n; r++) idxs[r] = r;
                Arrays.sort(idxs, (a, b) -> largest ?
                    Double.compare(vals[b], vals[a]) : Double.compare(vals[a], vals[b]));
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
        return new RereDoubleTensor(values, resultShape);
    }

    @Override
    public IDoubleTensor pad(int[][] padding, String mode, double value) {
        if (padding.length != rank()) {
            throw new IllegalArgumentException("padding length must match rank");
        }
        int[] newShape = new int[rank()];
        for (int i = 0; i < rank(); i++) {
            newShape[i] = dim(i) + padding[i][0] + padding[i][1];
        }
        long total = 1;
        for (int d : newShape) total *= d;
        double[] result = new double[(int) total];
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
        return new RereDoubleTensor(result, newShape);
    }

    @Override
    public IDoubleTensor unfold(int dim, int size, int stride, int dilation) {
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
        double[] result = new double[(int) total];
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
        return new RereDoubleTensor(result, resultShape);
    }

    @Override
    public IDoubleTensor nonzero() {
        java.util.ArrayList<Long> positions = new java.util.ArrayList<>();
        int n = (int) totalSize();
        for (int i = 0; i < n; i++) {
            if (linearGet(i) != 0) {
                positions.add((long) i);
            }
        }
        double[] result = new double[positions.size() * rank()];
        int idx = 0;
        for (long pos : positions) {
            int[] indices = unlinearize((int) pos, shape());
            for (int j = 0; j < rank(); j++) {
                result[idx++] = indices[j];
            }
        }
        return new RereDoubleTensor(result, positions.size(), rank());
    }

    @Override
    public IDoubleTensor maskedSelect(IDoubleTensor mask) {
        java.util.ArrayList<Double> selected = new java.util.ArrayList<>();
        for (int i = 0; i < (int) totalSize(); i++) {
            int[] idx = unlinearize(i, shape());
            if (mask.get(idx) > 0.5) {
                selected.add(linearGet(i));
            }
        }
        double[] result = new double[selected.size()];
        for (int i = 0; i < selected.size(); i++) result[i] = selected.get(i);
        return new RereDoubleTensor(result, result.length);
    }

    @Override
    public IDoubleTensor maskedFill(IDoubleTensor mask, double value) {
        double[] result = toDoubleArray();
        for (int i = 0; i < (int) totalSize(); i++) {
            int[] idx = unlinearize(i, shape());
            if (mask.get(idx) > 0.5) {
                result[i] = value;
            }
        }
        return new RereDoubleTensor(result, shape());
    }

    @Override
    public IDoubleTensor cat(int dim, IDoubleTensor... others) {
        if (dim < 0) dim += rank();
        int totalDim = dim(dim);
        for (IDoubleTensor t : others) totalDim += t.dim(dim);
        int[] newShape = shape().clone();
        newShape[dim] = totalDim;
        double[] result = new double[(int) computeSize(newShape)];
        // Copy this tensor
        copyInto(result, this, newShape, 0, dim);
        int offset = dim(dim);
        for (IDoubleTensor t : others) {
            copyInto(result, t, newShape, offset, dim);
            offset += t.dim(dim);
        }
        return new RereDoubleTensor(result, newShape);
    }

    private void copyInto(double[] target, IDoubleTensor src, int[] targetShape,
                           int dimOffset, int dim) {
        for (long i = 0; i < src.totalSize(); i++) {
            int[] srcIdx = unlinearize((int) i, src.shape());
            int[] tgtIdx = new int[targetShape.length];
            for (int j = 0; j < targetShape.length; j++) {
                tgtIdx[j] = j == dim ? srcIdx[j] + dimOffset : srcIdx[j];
            }
            long linear = linearIndex(tgtIdx, targetShape);
            target[(int) linear] = ((RereDoubleTensor) src).linearGet(i);
        }
    }

    @Override
    public IDoubleTensor stack(int dim, IDoubleTensor... tensors) {
        if (dim < 0) dim = rank() + 1;
        int[] elemShape = shape();
        for (IDoubleTensor t : tensors) {
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
        IDoubleTensor[] all = new IDoubleTensor[1 + tensors.length];
        all[0] = unsqueeze(dim);
        for (int i = 0; i < tensors.length; i++) {
            all[i + 1] = tensors[i].unsqueeze(dim);
        }
        return all[0].cat(dim, java.util.Arrays.copyOfRange(all, 1, all.length));
    }

    @Override
    public List<IDoubleTensor> unstack(int dim) {
        if (dim < 0) dim += rank();
        int n = dim(dim);
        List<IDoubleTensor> result = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            result.add(select(dim, i));
        }
        return result;
    }

    @Override
    public IDoubleTensor normalize(double p, int dim) {
        if (dim < 0) dim += rank();
        IDoubleTensor norm = abs().pow(p).sum(dim, true).pow(1.0 / p);
        return div(norm);
    }

    // ==================== 就地操作 ====================

    @Override
    public IDoubleTensor add_(IDoubleTensor other) {
        int n = (int) totalSize();
        for (int i = 0; i < n; i++) {
            linearSet(i, linearGet(i) + ((RereDoubleTensor) other).linearGet(i));
        }
        return this;
    }

    @Override
    public IDoubleTensor sub_(IDoubleTensor other) {
        int n = (int) totalSize();
        for (int i = 0; i < n; i++) {
            linearSet(i, linearGet(i) - ((RereDoubleTensor) other).linearGet(i));
        }
        return this;
    }

    @Override
    public IDoubleTensor mul_(IDoubleTensor other) {
        int n = (int) totalSize();
        for (int i = 0; i < n; i++) {
            linearSet(i, linearGet(i) * ((RereDoubleTensor) other).linearGet(i));
        }
        return this;
    }

    @Override
    public IDoubleTensor div_(IDoubleTensor other) {
        int n = (int) totalSize();
        for (int i = 0; i < n; i++) {
            linearSet(i, linearGet(i) / ((RereDoubleTensor) other).linearGet(i));
        }
        return this;
    }

    @Override
    public IDoubleTensor fill_(double value) {
        fill(value);
        return this;
    }

    @Override
    public IDoubleTensor copy_(IDoubleTensor src) {
        if (totalSize() != src.totalSize()) {
            throw new IllegalArgumentException("copy_ size mismatch");
        }
        int n = (int) totalSize();
        for (int i = 0; i < n; i++) {
            linearSet(i, ((RereDoubleTensor) src).linearGet(i));
        }
        return this;
    }

    // ==================== 转换桥接 ====================

    @Override
    public IDoubleVector toVector() {
        if (isContiguous() && offset == 0 && rank() == 1) {
            return IDoubleVector.of(data);
        }
        return IDoubleVector.of(toDoubleArray());
    }

    @Override
    public IDoubleVector toVectorCopy() {
        return IDoubleVector.of(toDoubleArray());
    }

    @Override
    public IMatrix toMatrix() {
        if (rank() != 2) {
            throw new IllegalStateException("toMatrix requires 2D tensor, got rank " + rank());
        }
        return Linalg.fromArray(toDoubleArray(), dim(0), dim(1));
    }

    @Override
    public IDoubleTensor copy() {
        return clone();
    }

    @Override
    public IDoubleTensor clone() {
        return new RereDoubleTensor(toDoubleArray(), shape());
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
    private static double getWithBroadcast(int[] idx, IDoubleTensor tensor) {
        int[] shapedIdx = new int[tensor.rank()];
        int diff = idx.length - tensor.rank();
        for (int i = 0; i < tensor.rank(); i++) {
            int sourceDim = tensor.dim(i);
            shapedIdx[i] = sourceDim == 1 ? 0 : idx[diff + i];
        }
        return tensor.get(shapedIdx);
    }
}
