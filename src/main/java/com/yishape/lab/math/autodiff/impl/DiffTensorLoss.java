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
public final class DiffTensorLoss {
    private DiffTensorLoss() { /* utility class */ }

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
        double[] dx = new double[(int) n];
        double[] dt = new double[(int) n];
        for (int i = 0; i < n; i++) {
            double p = clamped[i];
            double y = td[i];
            dx[i] = scale * (p - y) / (p * (1.0 - p));
            dt[i] = scale * (Math.log(1.0 - p) - Math.log(p));
        }
        inpX.accGrad(dx);
        inpT.accGrad(dt);
    };
    return new RereDiffTensor(loss, new int[]{1}, List.of(tensor, tgt), bw, "bceLoss");
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
        double[] dx = new double[(int) n];
        double[] dt = new double[(int) n];
        for (int i = 0; i < n; i++) {
            dx[i] = g * (twoIplusSmooth - 2.0 * td[i] * denomf) * invDenom2;
            dt[i] = g * (twoIplusSmooth - 2.0 * xd[i] * denomf) * invDenom2;
        }
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
