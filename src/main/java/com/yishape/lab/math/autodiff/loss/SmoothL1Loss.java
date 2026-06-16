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
 * Smooth L1 Loss (Huber loss with beta threshold).
 *
 * <p>Forward: loss = mean( |x-y| &lt;= beta ? 0.5*(x-y)&sup2;/beta : |x-y|-0.5*beta )</p>
 *
 * <p>Acceleration strategy:
 * <ul>
 *   <li>{@code diff = x - y} and {@code |diff|} via DoubleVectorComputer (GPU&rarr;SIMD&rarr;SISD)</li>
 *   <li>Forward/backward raw gradients are conditional per-element &mdash;
 *       no DoubleVectorComputer where/select exists, hand-written loop required</li>
 *   <li>Scaling of raw gradients via DoubleVectorComputer.binaryOperate(MULTIPLY)</li>
 * </ul>
 */
public final class SmoothL1Loss {
    private SmoothL1Loss() { /* utility class */ }

    public static IDiffTensor apply(RereDiffTensor tensor, IDiffTensor target, double beta) {
        RereDiffTensor tgt = (RereDiffTensor) target;
        double[] xd = tensor.value.toDoubleArray();
        double[] td = tgt.value.toDoubleArray();
        long n = tensor.value.totalSize();
        int ni = Math.toIntExact(n);
        double[] loss = new double[1];
        double halfBeta = 0.5 * beta;

        DoubleVectorComputer vc = new DoubleVectorComputer();

        // diff = x - y (vectorized)
        double[] diff = vc.binaryOperate(xd, td, BinaryOperation.SUBTRACT);
        // absDiff = |diff| (vectorized)
        double[] absDiff = vc.universalOperate(diff, UniversalOperation.ABS, 0);

        // Forward: conditional per-element — no DoubleVectorComputer where/select, hand-written loop
        double totalLoss = 0;
        for (int i = 0; i < ni; i++) {
            double ad = absDiff[i];
            if (ad <= beta) {
                totalLoss += 0.5 * diff[i] * diff[i] / beta;
            } else {
                totalLoss += ad - halfBeta;
            }
        }
        loss[0] = totalLoss / n;

        if (!tensor.requiresGrad && !tgt.requiresGrad) {
            return new ConstantDiffTensor(new RereDoubleTensor(loss, new int[]{1}));
        }

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpT = self.inputs.get(1);
            double[] g = self.grad;
            double scale = g[0] / n;

            // Backward: conditional per-element raw gradient — hand-written loop
            double[] rawGrad = new double[ni];
            for (int i = 0; i < ni; i++) {
                double d = diff[i];
                double ad = absDiff[i];
                rawGrad[i] = (ad <= beta) ? d / beta : Math.signum(d);
            }
            // Scale via DoubleVectorComputer
            double[] dx = vc.binaryOperate(rawGrad, scale, BinaryOperation.MULTIPLY);
            double[] dt = vc.binaryOperate(rawGrad, -scale, BinaryOperation.MULTIPLY);

            inpX.accGrad(dx);
            inpT.accGrad(dt);
        };

        return new RereDiffTensor(loss, new int[]{1}, List.of(tensor, tgt), bw, "smoothL1Loss");
    }
}
