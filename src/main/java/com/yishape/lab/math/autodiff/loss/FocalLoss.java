package com.yishape.lab.math.autodiff.loss;

import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.ConstantDiffTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import java.util.List;
import java.util.function.Consumer;

/**
 * Focal Loss (Lin et al., 2017) for class-imbalanced classification.
 *
 * <p>FL(p_t) = alpha_t * (1 - p_t)^gamma * (-log(p_t))</p>
 * <p>p_t = p if y=1, else 1-p; alpha_t = alpha if y=1, else 1-alpha</p>
 *
 * <p>Forward and backward use hand-written conditional per-element loops.
 * DoubleVectorComputer has no where/select operation, so per-element conditionals
 * must remain hand-written. Raw gradients are scaled via DoubleVectorComputer.</p>
 */
public final class FocalLoss {
    private FocalLoss() { /* utility class */ }

    public static IDiffTensor apply(RereDiffTensor tensor, IDiffTensor target, double alpha, double gamma) {
        RereDiffTensor tgt = (RereDiffTensor) target;
        double[] xd = tensor.value.toDoubleArray();
        double[] td = tgt.value.toDoubleArray();
        long n = tensor.value.totalSize();
        int ni = Math.toIntExact(n);
        double[] loss = new double[1];
        final double eps = 1e-7;
        double oneMinusAlpha = 1.0 - alpha;

        // clamp(x, eps, 1-eps) — hand-written: BinaryOperation has no MIN/MAX
        double[] clamped = new double[ni];
        for (int i = 0; i < ni; i++) {
            clamped[i] = Math.max(eps, Math.min(1.0 - eps, xd[i]));
        }

        // Forward: conditional per-element reduce — hand-written
        double totalLoss = 0;
        for (int i = 0; i < ni; i++) {
            double p = clamped[i];
            double y = td[i];
            double pT = (y > 0.5) ? p : 1.0 - p;
            double aT = (y > 0.5) ? alpha : oneMinusAlpha;
            double focalWeight = Math.pow(1.0 - pT, gamma);
            totalLoss += aT * focalWeight * (-Math.log(pT));
        }
        loss[0] = totalLoss / n;

        if (!tensor.requiresGrad && !tgt.requiresGrad) {
            return new ConstantDiffTensor(new RereDoubleTensor(loss, new int[]{1}));
        }

        DoubleVectorComputer vc = new DoubleVectorComputer();

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpT = self.inputs.get(1);
            double[] g = self.grad;
            double scale = g[0] / n;

            // Backward: conditional per-element raw gradient — hand-written
            double[] rawDx = new double[ni];
            double[] rawDt = new double[ni];
            for (int i = 0; i < ni; i++) {
                double p = clamped[i];
                double y = td[i];
                if (y > 0.5) {
                    double oneMinusP = 1.0 - p;
                    double term1 = gamma * Math.pow(oneMinusP, gamma - 1.0) * (-Math.log(p));
                    double term2 = Math.pow(oneMinusP, gamma) / p;
                    rawDx[i] = alpha * (term1 - term2);
                    rawDt[i] = alpha * Math.pow(oneMinusP, gamma) * Math.log(p);
                } else {
                    double term1 = gamma * Math.pow(p, gamma - 1.0) * (-Math.log(1.0 - p));
                    double term2 = Math.pow(p, gamma) / (1.0 - p);
                    rawDx[i] = oneMinusAlpha * (term1 + term2);
                    rawDt[i] = oneMinusAlpha * Math.pow(p, gamma) * Math.log(1.0 - p);
                }
            }
            // Scale via DoubleVectorComputer
            double[] dx = vc.binaryOperate(rawDx, scale, BinaryOperation.MULTIPLY);
            double[] dt = vc.binaryOperate(rawDt, scale, BinaryOperation.MULTIPLY);

            inpX.accGrad(dx);
            inpT.accGrad(dt);
        };

        return new RereDiffTensor(loss, new int[]{1}, List.of(tensor, tgt), bw, "focalLoss");
    }
}
