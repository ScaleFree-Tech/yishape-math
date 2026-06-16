package com.yishape.lab.math.autodiff.impl.delegate;

import com.yishape.lab.math.autodiff.impl.DiffTensorUtil;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.compute.DoubleFlatGemm;
import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.gpu.GpuActivation;
import com.yishape.lab.math.compute.gpu.GpuReduce;
import com.yishape.lab.math.compute.hpc.HpcIm2col;
import com.yishape.lab.math.compute.hpc.HpcLoss;
import com.yishape.lab.math.compute.hpc.HpcCross;
import com.yishape.lab.math.compute.hpc.HpcGridSample;
import com.yishape.lab.math.compute.hpc.HpcTrapezoidalScan;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.tensor.EinsumParser;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import com.yishape.lab.math.linalg.tensor.TensorShape;
import com.yishape.lab.math.compute.gpu.GpuGroupNorm;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;

/**
 * Extracted from {@link RereDiffTensor}.
 * All methods are static, taking the tensor as first parameter.
 */
public final class DiffTensorInPlace {
    private DiffTensorInPlace() { /* utility class */ }

// ==================== In-place ops ====================

public static IDiffTensor add_(RereDiffTensor tensor, IDoubleTensor other) {
    return inPlaceBinaryOp(tensor, other, (a, b) -> a + b);
}

public static IDiffTensor sub_(RereDiffTensor tensor, IDoubleTensor other) {
    return inPlaceBinaryOp(tensor, other, (a, b) -> a - b);
}

public static IDiffTensor mul_(RereDiffTensor tensor, IDoubleTensor other) {
    return inPlaceBinaryOp(tensor, other, (a, b) -> a * b);
}

public static IDiffTensor div_(RereDiffTensor tensor, IDoubleTensor other) {
    return inPlaceBinaryOp(tensor, other, (a, b) -> a / b);
}

static IDiffTensor inPlaceBinaryOp(RereDiffTensor tensor, IDoubleTensor other, DoubleBinaryOperator op) {
    if (!tensor.isLeaf) throw new IllegalStateException("In-place ops only allowed on leaf tensors");
    IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
    if (Arrays.equals(tensor.shape(), other.shape())) {
        long n = tensor.value.totalSize();
        double[] oData = detOther.toDoubleArray();
        for (long i = 0; i < n; i++) {
            tensor.value.linearSet(i, op.applyAsDouble(tensor.value.linearGet(i), oData[(int) i]));
        }
    } else {
        // Broadcast in-place
        int[] bc = TensorShape.broadcastShape(tensor.shape(), other.shape());
        long n = 1;
        for (int d : bc) n *= d;
        double[] oData = detOther.toDoubleArray();
        for (long i = 0; i < n; i++) {
            int[] bcIdx = DiffTensorUtil.unlinearizeInt((int) i, bc);
            int flatSelf = DiffTensorUtil.flatIndexFromBroadcast(bcIdx, tensor.shape(), bc);
            int flatOther = DiffTensorUtil.flatIndexFromBroadcast(bcIdx, other.shape(), bc);
            tensor.value.linearSet(flatSelf, op.applyAsDouble(tensor.value.linearGet(flatSelf), oData[flatOther]));
        }
    }
    tensor.grad = null;
    return tensor;
}

public static IDiffTensor fill_(RereDiffTensor tensor, double val) {
    if (!tensor.isLeaf) throw new IllegalStateException("fill_ only allowed on leaf tensors");
    tensor.value.fill(val);
    tensor.grad = null;
    return tensor;
}

public static IDiffTensor copy_(RereDiffTensor tensor, IDoubleTensor src) {
    if (!tensor.isLeaf) throw new IllegalStateException("copy_ only allowed on leaf tensors");
    IDoubleTensor detSrc = (src instanceof IDiffTensor dt) ? dt.detach() : src;
    double[] sData = detSrc.toDoubleArray();
    long n = tensor.value.totalSize();
    for (long i = 0; i < n && i < sData.length; i++) tensor.value.linearSet(i, sData[(int) i]);
    tensor.grad = null;
    return tensor;
}

}
