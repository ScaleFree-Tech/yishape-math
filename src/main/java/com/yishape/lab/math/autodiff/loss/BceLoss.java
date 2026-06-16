package com.yishape.lab.math.autodiff.loss;

import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.math.compute.gpu.GpuActivation;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.ConstantDiffTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import java.util.List;
import java.util.function.Consumer;

/**
 * Binary Cross Entropy Loss.
 *
 * <p>Forward: loss = mean( -y*log(p) - (1-y)*log(1-p) ) where p = clamp(x, eps, 1-eps)</p>
 *
 * <p>Acceleration strategy:
 * <ul>
 *   <li>Clamp uses hand-written loop &mdash; DoubleVectorComputer has no MIN/MAX</li>
 *   <li>All other element-wise ops use DoubleVectorComputer (GPU&rarr;SIMD&rarr;SISD)</li>
 *   <li>Forward reduce via DoubleVectorComputer.reduceOperate(SUM)</li>
 *   <li>Backward fully vectorized (same as original DiffTensorLoss)</li>
 * </ul>
 */
public final class BceLoss {
    private BceLoss() { /* utility class */ }

    public static IDiffTensor apply(RereDiffTensor tensor, IDiffTensor target) {
        RereDiffTensor tgt = (RereDiffTensor) target;
        double[] xd = tensor.value.toDoubleArray();
        double[] td = tgt.value.toDoubleArray();
        long n = tensor.value.totalSize();
        int ni = Math.toIntExact(n);
        double[] loss = new double[1];
        final double eps = 1e-7;

        DoubleVectorComputer vc = new DoubleVectorComputer();

        // clamp(x, eps, 1-eps) — hand-written: BinaryOperation has no MIN/MAX
        double[] clamped = new double[ni];
        for (int i = 0; i < ni; i++) {
            clamped[i] = Math.max(eps, Math.min(1.0 - eps, xd[i]));
        }

        // Vectorized forward: lossPerElem = -y*log(p) - (1-y)*log(1-p)
        final double[] logP;
        double[] logPTemp = GpuActivation.tryLog(clamped);
        if (logPTemp == null) logPTemp = vc.universalOperate(clamped, UniversalOperation.LOG, 0);
        logP = logPTemp;

        double[] pMinus1 = vc.binaryOperate(clamped, 1.0, BinaryOperation.SUBTRACT);
        double[] oneMinusP = vc.binaryOperate(pMinus1, -1.0, BinaryOperation.MULTIPLY);
        final double[] logOneMinusP;
        double[] logOneMinusPTemp = GpuActivation.tryLog(oneMinusP);
        if (logOneMinusPTemp == null) logOneMinusPTemp = vc.universalOperate(oneMinusP, UniversalOperation.LOG, 0);
        logOneMinusP = logOneMinusPTemp;

        double[] yTimesLogP = vc.binaryOperate(td, logP, BinaryOperation.MULTIPLY);
        double[] negTerm1 = vc.binaryOperate(yTimesLogP, -1.0, BinaryOperation.MULTIPLY);

        double[] yMinus1 = vc.binaryOperate(td, 1.0, BinaryOperation.SUBTRACT);
        double[] oneMinusY = vc.binaryOperate(yMinus1, -1.0, BinaryOperation.MULTIPLY);
        double[] oneMinusYTimesLogOneMinusP = vc.binaryOperate(oneMinusY, logOneMinusP, BinaryOperation.MULTIPLY);

        double[] lossPerElem = vc.binaryOperate(negTerm1, oneMinusYTimesLogOneMinusP, BinaryOperation.ADD);
        loss[0] = vc.reduceOperate(lossPerElem, ReduceOperation.SUM) / n;

        if (!tensor.requiresGrad) {
            return new ConstantDiffTensor(new RereDoubleTensor(loss, new int[]{1}));
        }

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpT = self.inputs.get(1);
            double[] g = self.grad;
            double scale = g[0] / n;

            // dx = scale * (p - y) / (p * (1 - p))
            double[] p2 = vc.binaryOperate(clamped, clamped, BinaryOperation.MULTIPLY);
            double[] denom = vc.binaryOperate(clamped, p2, BinaryOperation.SUBTRACT);
            double[] diff = vc.binaryOperate(clamped, td, BinaryOperation.SUBTRACT);
            double[] ratio = vc.binaryOperate(diff, denom, BinaryOperation.DIVIDE);
            double[] dx = vc.binaryOperate(ratio, scale, BinaryOperation.MULTIPLY);

            // dt = scale * (log(1-p) - log(p))
            double[] logDiff = vc.binaryOperate(logOneMinusP, logP, BinaryOperation.SUBTRACT);
            double[] dt = vc.binaryOperate(logDiff, scale, BinaryOperation.MULTIPLY);

            inpX.accGrad(dx);
            inpT.accGrad(dt);
        };

        return new RereDiffTensor(loss, new int[]{1}, List.of(tensor, tgt), bw, "bceLoss");
    }
}
