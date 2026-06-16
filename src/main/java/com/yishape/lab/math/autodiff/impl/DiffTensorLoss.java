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
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
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
public final class DiffTensorLoss {
    private DiffTensorLoss() { /* utility class */ }

    private static final DoubleVectorComputer COMPUTER = new DoubleVectorComputer();

// ==================== Phase 3: Loss Functions ====================

public static IDiffTensor smoothL1Loss(RereDiffTensor tensor, IDiffTensor target, double beta) {
    RereDiffTensor tgt = (RereDiffTensor) target;
    double[] xd = tensor.value.toDoubleArray();
    double[] td = tgt.value.toDoubleArray();
    long n = tensor.value.totalSize();
    double[] loss = new double[1];
    double[] diff = new double[(int) n];
    double totalLoss = 0;
    double halfBeta = 0.5 * beta;

    for (int i = 0; i < n; i++) {
        double d = xd[i] - td[i];
        diff[i] = d;
        double absD = Math.abs(d);
        if (absD <= beta) {
            totalLoss += 0.5 * d * d / beta;
        } else {
            totalLoss += absD - halfBeta;
        }
    }
    loss[0] = totalLoss / n;

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpX = self.inputs.get(0);
        RereDiffTensor inpT = self.inputs.get(1);
        double[] g = self.grad;
        double scale = g[0] / n;
        double[] dx = new double[(int) n];
        double[] dt = new double[(int) n];
        for (int i = 0; i < n; i++) {
            double d = diff[i];
            double absD = Math.abs(d);
            double gradVal;
            if (absD <= beta) {
                gradVal = d / beta;
            } else {
                gradVal = Math.signum(d);
            }
            dx[i] = scale * gradVal;
            dt[i] = -scale * gradVal;
        }
        inpX.accGrad(dx);
        inpT.accGrad(dt);
    };
    return new RereDiffTensor(loss, new int[]{1}, List.of(tensor, tgt), bw, "smoothL1Loss");
}
public static IDiffTensor bceLoss(RereDiffTensor tensor, IDiffTensor target) {
    RereDiffTensor tgt = (RereDiffTensor) target;
    double[] xd = tensor.value.toDoubleArray();
    double[] td = tgt.value.toDoubleArray();
    long n = tensor.value.totalSize();
    double[] loss = new double[1];
    double[] clamped = new double[(int) n];
    double totalLoss = 0;
    final double eps = 1e-7;
    for (int i = 0; i < n; i++) {
        double p = Math.max(eps, Math.min(1.0 - eps, xd[i]));
        clamped[i] = p;
        double y = td[i];
        totalLoss += -y * Math.log(p) - (1.0 - y) * Math.log(1.0 - p);
    }
    loss[0] = totalLoss / n;
    if (!tensor.requiresGrad) return new ConstantDiffTensor(new RereDoubleTensor(loss, new int[]{1}));
	    Consumer<RereDiffTensor> bw = self -> {
	        RereDiffTensor inpX = self.inputs.get(0);
	        RereDiffTensor inpT = self.inputs.get(1);
	        double[] g = self.grad;
	        double scale = g[0] / n;
	        DoubleVectorComputer vc = COMPUTER;

	        // dx = scale * (p - y) / (p * (1 - p)) = scale * (p - y) / (p - p^2)
	        double[] p2 = vc.binaryOperate(clamped, clamped, BinaryOperation.MULTIPLY);
	        double[] denom = vc.binaryOperate(clamped, p2, BinaryOperation.SUBTRACT);
	        double[] diff = vc.binaryOperate(clamped, td, BinaryOperation.SUBTRACT);
	        double[] ratio = vc.binaryOperate(diff, denom, BinaryOperation.DIVIDE);
	        double[] dx = vc.binaryOperate(ratio, scale, BinaryOperation.MULTIPLY);

	        // dt = scale * (log(1-p) - log(p))
	        double[] logP = GpuActivation.tryLog(clamped);
	        if (logP == null) logP = vc.universalOperate(clamped, UniversalOperation.LOG, 0);
	        double[] pMinus1 = vc.binaryOperate(clamped, 1.0, BinaryOperation.SUBTRACT);
	        double[] oneMinusP = vc.binaryOperate(pMinus1, -1.0, BinaryOperation.MULTIPLY);
	        double[] logOneMinusP = GpuActivation.tryLog(oneMinusP);
	        if (logOneMinusP == null) logOneMinusP = vc.universalOperate(oneMinusP, UniversalOperation.LOG, 0);
	        double[] logDiff = vc.binaryOperate(logOneMinusP, logP, BinaryOperation.SUBTRACT);
	        double[] dt = vc.binaryOperate(logDiff, scale, BinaryOperation.MULTIPLY);

	        inpX.accGrad(dx);
	        inpT.accGrad(dt);
	    };
	    return new RereDiffTensor(loss, new int[]{1}, List.of(tensor, tgt), bw, "bceLoss");
}

    /**
     * Numerically stable BCE with logits (fused custom op).
     * Uses log-sum-exp trick: max(x,0) - x*t + log(1+exp(-|x|)).
     * Backward: (sigmoid(x) - target) / n — no division by p*(1-p).
     * Prefer this over {@link #bceLoss} when inputs are raw logits.
     */
    public static IDiffTensor bceWithLogitsLoss(RereDiffTensor tensor, IDiffTensor target) {
        RereDiffTensor tgt = (RereDiffTensor) target;
        double[] xd = tensor.value.toDoubleArray();
        double[] td = tgt.value.toDoubleArray();
        long n = tensor.value.totalSize();
        int ni = Math.toIntExact(n);
        double[] loss = new double[1];
        // Save logits for backward sigmoid recompute
        double[] logits = new double[ni];
        System.arraycopy(xd, 0, logits, 0, ni);
        DoubleVectorComputer vc = COMPUTER;
        // Forward: posPart = relu(x)
        double[] posPart = GpuActivation.tryRelu(xd);
        if (posPart == null) posPart = vc.universalOperate(xd, UniversalOperation.RELU, 0);
        // xTarget = x * target
        double[] xTarget = vc.binaryOperate(xd, td, BinaryOperation.MULTIPLY);
        // absX = |x|; negAbsX = -absX; expNegAbsX = exp(-|x|)
        double[] absX = vc.universalOperate(xd, UniversalOperation.ABS, 0);
        double[] negAbsX = vc.binaryOperate(absX, -1.0, BinaryOperation.MULTIPLY);
        double[] expNegAbsX = GpuActivation.tryExp(negAbsX);
        if (expNegAbsX == null) expNegAbsX = vc.universalOperate(negAbsX, UniversalOperation.EXP, 0);
        // log1p = log(1 + exp(-|x|)) = log1p(exp(-|x|)) for better precision
        double[] log1pArg = vc.binaryOperate(expNegAbsX, 1.0, BinaryOperation.ADD);
        double[] logTerm = GpuActivation.tryLog(log1pArg);
        if (logTerm == null) logTerm = vc.universalOperate(log1pArg, UniversalOperation.LOG, 0);
        // lossPerElem = posPart - xTarget + logTerm
        double[] lossElem = vc.binaryOperate(posPart, xTarget, BinaryOperation.SUBTRACT);
        lossElem = vc.binaryOperate(lossElem, logTerm, BinaryOperation.ADD);
        loss[0] = vc.reduceOperate(lossElem, ReduceOperation.SUM) / n;
        if (!tensor.requiresGrad) return new ConstantDiffTensor(new RereDoubleTensor(loss, new int[]{1}));
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpT = self.inputs.get(1);
            double[] g = self.grad;
            double scale = g[0] / n;
            // probs = sigmoid(logits) via acceleration chain
            double[] probs = GpuActivation.trySigmoid(logits);
            if (probs == null) probs = vc.universalOperate(logits, UniversalOperation.SIGMOID, 0);
            // dx = scale * (probs - target)
            double[] diff = vc.binaryOperate(probs, td, BinaryOperation.SUBTRACT);
            double[] dx = vc.binaryOperate(diff, scale, BinaryOperation.MULTIPLY);
            inpX.accGrad(dx);
            // dt: dL/dtarget = -logits / n
            double[] dt = vc.binaryOperate(logits, -scale, BinaryOperation.MULTIPLY);
            inpT.accGrad(dt);
        };
        return new RereDiffTensor(loss, new int[]{1}, List.of(tensor, tgt), bw, "bceWithLogitsLoss");
    }

public static IDiffTensor focalLoss(RereDiffTensor tensor, IDiffTensor target, double alpha, double gamma) {
    RereDiffTensor tgt = (RereDiffTensor) target;
    double[] xd = tensor.value.toDoubleArray();
    double[] td = tgt.value.toDoubleArray();
    long n = tensor.value.totalSize();
    double[] loss = new double[1];
    double[] clamped = new double[(int) n];
    double totalLoss = 0;
    final double eps = 1e-7;
    double oneMinusAlpha = 1.0 - alpha;
    for (int i = 0; i < n; i++) {
        double p = Math.max(eps, Math.min(1.0 - eps, xd[i]));
        clamped[i] = p;
        double y = td[i];
        double pT = (y > 0.5) ? p : 1.0 - p;
        double aT = (y > 0.5) ? alpha : oneMinusAlpha;
        double focalWeight = Math.pow(1.0 - pT, gamma);
        totalLoss += aT * focalWeight * (-Math.log(pT));
    }
    loss[0] = totalLoss / n;
    if (!tensor.requiresGrad) return new ConstantDiffTensor(new RereDoubleTensor(loss, new int[]{1}));
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpX = self.inputs.get(0);
        RereDiffTensor inpT = self.inputs.get(1);
        double[] g = self.grad;
        double scale = g[0] / n;
        double[] dx = new double[(int) n];
        double[] dt = new double[(int) n];
        for (int i = 0; i < n; i++) {
            double p = clamped[i];
            double y = td[i];
            if (y > 0.5) {
                double oneMinusP = 1.0 - p;
                double term1 = gamma * Math.pow(oneMinusP, gamma - 1.0) * (-Math.log(p));
                double term2 = Math.pow(oneMinusP, gamma) / p;
                dx[i] = scale * alpha * (term1 - term2);
                dt[i] = scale * alpha * Math.pow(oneMinusP, gamma) * Math.log(p);
            } else {
                double term1 = gamma * Math.pow(p, gamma - 1.0) * (-Math.log(1.0 - p));
                double term2 = Math.pow(p, gamma) / (1.0 - p);
                dx[i] = scale * oneMinusAlpha * (term1 + term2);
                dt[i] = scale * oneMinusAlpha * Math.pow(p, gamma) * Math.log(1.0 - p);
            }
        }
        inpX.accGrad(dx);
        inpT.accGrad(dt);
    };
    return new RereDiffTensor(loss, new int[]{1}, List.of(tensor, tgt), bw, "focalLoss");
}

public static IDiffTensor diceLoss(RereDiffTensor tensor, IDiffTensor target, double smooth) {
    RereDiffTensor tgt = (RereDiffTensor) target;
    double[] xd = tensor.value.toDoubleArray();
    double[] td = tgt.value.toDoubleArray();
    long n = tensor.value.totalSize();
    double[] loss = new double[1];
    double I = 0, Sp = 0, St = 0;
    for (int i = 0; i < n; i++) {
        I += xd[i] * td[i];
        Sp += xd[i];
        St += td[i];
    }
    double denom = Sp + St + smooth;
    double dice = (2.0 * I + smooth) / denom;
    final double If = I;
    final double denomf = denom;
    loss[0] = 1.0 - dice;
    if (!tensor.requiresGrad) return new ConstantDiffTensor(new RereDoubleTensor(loss, new int[]{1}));
	    Consumer<RereDiffTensor> bw = self -> {
	        RereDiffTensor inpX = self.inputs.get(0);
	        RereDiffTensor inpT = self.inputs.get(1);
	        double g = self.grad[0];
	        double invDenom2 = 1.0 / (denomf * denomf);
	        double twoIplusSmooth = 2.0 * If + smooth;
	        DoubleVectorComputer vc = COMPUTER;

	        // dx = g * invDenom2 * (twoIplusSmooth - 2 * denomf * td)
	        //    = A * twoIplusSmooth + coeff * td   where A = g*invDenom2, coeff = -A*2*denomf
	        double A = g * invDenom2;
	        double constTerm = A * twoIplusSmooth;
	        double coeff = -A * 2.0 * denomf;
	        double[] dx = vc.binaryOperate(td, coeff, BinaryOperation.MULTIPLY);
	        dx = vc.binaryOperate(dx, constTerm, BinaryOperation.ADD);

	        // dt = g * invDenom2 * (twoIplusSmooth - 2 * denomf * xd)
	        double[] dt = vc.binaryOperate(xd, coeff, BinaryOperation.MULTIPLY);
	        dt = vc.binaryOperate(dt, constTerm, BinaryOperation.ADD);

	        inpX.accGrad(dx);
	        inpT.accGrad(dt);
	    };
    return new RereDiffTensor(loss, new int[]{1}, List.of(tensor, tgt), bw, "diceLoss");
}

public static IDiffTensor nllLoss(RereDiffTensor tensor, IDiffTensor target, int classDim) {
    // Input is log-probabilities. Compute -mean(gather(logProbs, classDim, target))
    int d = (classDim < 0 ? classDim + tensor.rank() : classDim);
    RereDiffTensor tgt = (RereDiffTensor) target;
    // gather along classDim using target as indices
    IDiffTensor gathered = tensor.gather(d, tgt);
    IDiffTensor loss = gathered.sum().div(gathered.totalSize()).neg();
    if (loss instanceof RereDiffTensor rt) {
        rt.setOpTag("nllLoss");
    }
    return loss;
}

}
