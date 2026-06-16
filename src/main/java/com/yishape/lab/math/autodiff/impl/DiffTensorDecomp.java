package com.yishape.lab.math.autodiff.impl;

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

/**
 * Extracted from {@link RereDiffTensor}.
 * All methods are static, taking the tensor as first parameter.
 */
public final class DiffTensorDecomp {
    private DiffTensorDecomp() { /* utility class */ }

// ==================== Matrix Decomposition Ops ====================

public static IDiffTensor logDet(RereDiffTensor tensor) {
    int[] s = tensor.shape();
    if (tensor.rank() != 2 || s[0] != s[1]) {
        throw new IllegalArgumentException(
            "logDet requires square 2D matrix, got shape " + Arrays.toString(s));
    }
    int n = s[0];
    double[] xd = tensor.value.toDoubleArray();
    IMatrix<Double> A = Linalg.fromArray(xd, n, n);
    double logDet;
    com.yishape.lab.math.linalg.decomposition.ILUDecomposition cachedLU = null;
    try {
        cachedLU = com.yishape.lab.math.linalg.decomposition.Decomps.createLU();
        cachedLU.decompose(A);
        double det = cachedLU.getDeterminant();
        logDet = Math.log(Math.abs(det));
    } catch (Exception e) {
        // Singular matrix: log|det| = -inf
        logDet = Double.NEGATIVE_INFINITY;
    }
    double[] result = {logDet};
    final int fN = n;
    final double fLogDet = logDet;
    final var fLU = cachedLU;

    Consumer<RereDiffTensor> bw = self -> {
        if (Double.isInfinite(fLogDet)) return; // no gradient for singular matrix
        RereDiffTensor inp = self.inputs.get(0);
        double gradOut = self.grad[0];
        // ∂log|det(A)|/∂A = A^{-T} — reuse cached LU from forward pass
        IMatrix<Double> I = Linalg.eye(fN);
        IMatrix<Double> Ainv;
        if (fLU != null) {
            Ainv = fLU.getSolver().solve(I);
        } else {
            double[] inpData = inp.value.toDoubleArray();
            IMatrix<Double> Amat = Linalg.fromArray(inpData, fN, fN);
            Ainv = Amat.solve(I);
        }
        IMatrix<Double> AinvT = Ainv.transpose();
        double[] gradA = AinvT.flatten().toDoubleArray();
        for (int i = 0; i < gradA.length; i++) gradA[i] *= gradOut;
        inp.accGrad(gradA);
    };

    return new RereDiffTensor(result, new int[]{1}, List.of(tensor), bw, "logDet");
}

public static IDiffTensor[] slogDet(RereDiffTensor tensor) {
    int[] s = tensor.shape();
    if (tensor.rank() != 2 || s[0] != s[1]) {
        throw new IllegalArgumentException(
            "slogDet requires square 2D matrix, got shape " + Arrays.toString(s));
    }
    int n = s[0];
    double[] xd = tensor.value.toDoubleArray();
    IMatrix<Double> A = Linalg.fromArray(xd, n, n);
    var luDecomp = com.yishape.lab.math.linalg.decomposition.Decomps.createLU();
    luDecomp.decompose(A);
    double det = luDecomp.getDeterminant();
    double signVal = Math.signum(det);
    double logDet = Math.log(Math.abs(det));
    double[] signArr = {signVal};
    double[] logDetArr = {logDet};
    final int fN = n;
    final var fLU = luDecomp;

    // sign tensor — no gradient flow (constant)
    RereDiffTensor signTensor = new RereDiffTensor(signArr, new int[]{1});
    signTensor.setRequiresGrad(false);

    // logDet tensor — differentiable
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inp = self.inputs.get(0);
        double gradOut = self.grad[0];
        IMatrix<Double> I = Linalg.eye(fN);
        IMatrix<Double> Ainv = fLU.getSolver().solve(I);
        IMatrix<Double> AinvT = Ainv.transpose();
        double[] gradA = AinvT.flatten().toDoubleArray();
        for (int i = 0; i < gradA.length; i++) gradA[i] *= gradOut;
        inp.accGrad(gradA);
    };

    RereDiffTensor logDetTensor = new RereDiffTensor(logDetArr, new int[]{1}, List.of(tensor), bw, "logDet");
    return new IDiffTensor[]{signTensor, logDetTensor};
}

public static IDiffTensor nuclearNorm(RereDiffTensor tensor) {
    int[] s = tensor.shape();
    if (tensor.rank() != 2) {
        throw new IllegalArgumentException(
            "nuclearNorm requires 2D matrix, got rank " + tensor.rank());
    }
    int m = s[0], nDim = s[1];
    double[] xd = tensor.value.toDoubleArray();
    IMatrix<Double> A = Linalg.fromArray(xd, m, nDim);
    var svdResult = A.svd();
    IMatrix<Double> U = svdResult.getFirst();
    var Svec = svdResult.getSecond();
    IMatrix<Double> VT = svdResult.getThird();

    double[] sVals = Svec.toDoubleArray();
    double nuclear = 0;
    for (double sv : sVals) nuclear += sv;
    double[] result = {nuclear};
    final int fM = m;
    final int fN = nDim;
    final int k = sVals.length;

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inp = self.inputs.get(0);
        double gradOut = self.grad[0];
        // ∂||A||_* / ∂A = U @ V^T (using thin SVD factors)
        double[][] uData = U.toDoubleArray();    // m×k (2D)
        double[][] vtData = VT.toDoubleArray();   // n×n (2D)
        // Extract V from V^T: V_hat = first k rows of VT (shape k×n)
        double[] vFlat = new double[k * fN];
        for (int i = 0; i < k; i++) {
            System.arraycopy(vtData[i], 0, vFlat, i * fN, fN);
        }
        // Compute U(m×k) @ V_hat(k×n) via matrix multiply
        IMatrix<Double> Umat = Linalg.fromArray(
            U.flatten().toDoubleArray(), fM, k);
        IMatrix<Double> Vmat = Linalg.fromArray(vFlat, k, fN);
        IMatrix<Double> gradMat = Umat.mmul(Vmat);
        double[] gradA = gradMat.flatten().toDoubleArray();
        for (int i = 0; i < gradA.length; i++) gradA[i] *= gradOut;
        inp.accGrad(gradA);
    };

    return new RereDiffTensor(result, new int[]{1}, List.of(tensor), bw, "nuclearNorm");
}

public static IDiffTensor ctcLoss(RereDiffTensor tensor, IDiffTensor targets, IDiffTensor inputLengths, IDiffTensor targetLengths) {
    int[] s = tensor.shape();
    if (tensor.rank() != 3) {
        throw new IllegalArgumentException(
            "ctcLoss requires 3D input [T, N, C], got shape " + Arrays.toString(s));
    }
    int T = s[0], N = s[1], C = s[2];
    double[] xd = tensor.value.toDoubleArray();          // [T, N, C] row-major
    double[] tgtData = targets.toDoubleArray();    // [N, S] row-major
    double[] inLenData = inputLengths.toDoubleArray(); // [N]
    double[] tgtLenData = targetLengths.toDoubleArray(); // [N]
    int S = targets.shape()[targets.rank() - 1];

    double totalLoss = 0;
    double[][] batchGrads = new double[N][];

    for (int batch = 0; batch < N; batch++) {
        int inLen = (int) inLenData[batch];
        int tgtLen = (int) tgtLenData[batch];
        if (inLen <= 0 || inLen > T) inLen = T;

        // Extract logProbs for this batch: [T, C] row-major → flat [T*C]
        double[] batchLP = new double[T * C];
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                batchLP[t * C + c] = xd[t * (N * C) + batch * C + c];
            }
        }

        // Extract labels for this batch
        int[] labels = new int[tgtLen];
        for (int i = 0; i < tgtLen; i++) {
            labels[i] = (int) tgtData[batch * S + i];
        }

        double[] lossOut = new double[1];
        double[] gradOut = new double[T * C];
        boolean ok = HpcLoss.tryCtcForwardBackward(batchLP, labels, tgtLen, inLen, C, lossOut, gradOut);
        if (!ok) {
            throw new UnsupportedOperationException(
                "CTC HPC native runtime unavailable. Java CTC fallback not yet implemented.");
        }
        batchGrads[batch] = gradOut;
        totalLoss += lossOut[0];
    }

    double avgLoss = totalLoss / N;
    double[] lossArr = {avgLoss};
    final int fT = T, fN = N, fC = C;
    final double[][] fBatchGrads = batchGrads;
    final double[] fInLenData = inLenData;

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inp = self.inputs.get(0);
        double gradOut = self.grad[0];
        double[] dx = new double[(int) inp.value.totalSize()];

        for (int batch = 0; batch < fN; batch++) {
            int inLen = (int) fInLenData[batch];
            if (inLen <= 0 || inLen > fT) inLen = fT;
            double[] g = fBatchGrads[batch];
            if (g != null) {
                for (int t = 0; t < inLen; t++) {
                    for (int c = 0; c < fC; c++) {
                        dx[t * (fN * fC) + batch * fC + c] += g[t * fC + c] * gradOut / fN;
                    }
                }
            }
        }
        inp.accGrad(dx);
    };

    List<RereDiffTensor> inputs = new ArrayList<>();
    inputs.add(tensor);
    inputs.add((RereDiffTensor) targets);
    inputs.add((RereDiffTensor) inputLengths);
    inputs.add((RereDiffTensor) targetLengths);

    return new RereDiffTensor(lossArr, new int[]{1}, inputs, bw, "ctcLoss");
}

}
