package com.yishape.lab.math.autodiff.loss;

import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.ConstantDiffTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import java.util.List;
import java.util.function.Consumer;

/**
 * Dice Loss for segmentation tasks.
 *
 * <p>Forward: loss = 1 - (2*I + smooth) / (Sp + St + smooth)</p>
 * <p>I = sum(x*y), Sp = sum(x), St = sum(y)</p>
 *
 * <p>Acceleration strategy:
 * <ul>
 *   <li>All element-wise ops use DoubleVectorComputer (GPU&rarr;SIMD&rarr;SISD)</li>
 *   <li>All reduces via DoubleVectorComputer.reduceOperate(SUM)</li>
 *   <li>No hand-written loops in forward or backward</li>
 * </ul>
 */
public final class DiceLoss {
    private DiceLoss() { /* utility class */ }

    public static IDiffTensor apply(RereDiffTensor tensor, IDiffTensor target, double smooth) {
        RereDiffTensor tgt = (RereDiffTensor) target;
        double[] xd = tensor.value.toDoubleArray();
        double[] td = tgt.value.toDoubleArray();
        long n = tensor.value.totalSize();
        double[] loss = new double[1];

        DoubleVectorComputer vc = new DoubleVectorComputer();

        // Vectorized forward: all reduces via DoubleVectorComputer
        double[] prod = vc.binaryOperate(xd, td, BinaryOperation.MULTIPLY);
        double I = vc.reduceOperate(prod, ReduceOperation.SUM);
        double Sp = vc.reduceOperate(xd, ReduceOperation.SUM);
        double St = vc.reduceOperate(td, ReduceOperation.SUM);

        double denom = Sp + St + smooth;
        double dice = (2.0 * I + smooth) / denom;
        final double If = I;
        final double denomf = denom;
        loss[0] = 1.0 - dice;

        if (!tensor.requiresGrad && !tgt.requiresGrad) {
            return new ConstantDiffTensor(new RereDoubleTensor(loss, new int[]{1}));
        }

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpT = self.inputs.get(1);
            double g = self.grad[0];
            double invDenom2 = 1.0 / (denomf * denomf);
            double twoIplusSmooth = 2.0 * If + smooth;

            // dx = g * invDenom2 * (twoIplusSmooth - 2 * denomf * td)
            //    = A * twoIplusSmooth + coeff * td
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
}
