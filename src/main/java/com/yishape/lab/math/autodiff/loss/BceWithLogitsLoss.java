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
 * Numerically stable BCE with logits (fused custom op).
 *
 * <p>Uses log-sum-exp trick: max(x,0) - x*t + log(1+exp(-|x|)).</p>
 * <p>Backward: (sigmoid(x) - target) / n &mdash; no division by p*(1-p).</p>
 * <p>Prefer this over {@link BceLoss} when inputs are raw logits.</p>
 */
public final class BceWithLogitsLoss {
    private BceWithLogitsLoss() { /* utility class */ }

    public static IDiffTensor apply(RereDiffTensor tensor, IDiffTensor target) {
        RereDiffTensor tgt = (RereDiffTensor) target;
        double[] xd = tensor.value.toDoubleArray();
        double[] td = tgt.value.toDoubleArray();
        long n = tensor.value.totalSize();
        int ni = Math.toIntExact(n);
        double[] loss = new double[1];
        double[] logits = new double[ni];
        System.arraycopy(xd, 0, logits, 0, ni);

        DoubleVectorComputer vc = new DoubleVectorComputer();

        // posPart = relu(x)
        double[] posPart = GpuActivation.tryRelu(xd);
        if (posPart == null) posPart = vc.universalOperate(xd, UniversalOperation.RELU, 0);

        // xTarget = x * t
        double[] xTarget = vc.binaryOperate(xd, td, BinaryOperation.MULTIPLY);

        // absX = |x|
        double[] absX = vc.universalOperate(xd, UniversalOperation.ABS, 0);

        // negAbsX = -absX
        double[] negAbsX = vc.binaryOperate(absX, -1.0, BinaryOperation.MULTIPLY);

        // expNegAbsX = exp(-|x|)
        double[] expNegAbsX = GpuActivation.tryExp(negAbsX);
        if (expNegAbsX == null) expNegAbsX = vc.universalOperate(negAbsX, UniversalOperation.EXP, 0);

        // log1pArg = 1 + exp(-|x|)
        double[] log1pArg = vc.binaryOperate(expNegAbsX, 1.0, BinaryOperation.ADD);

        // logTerm = log(1 + exp(-|x|))
        double[] logTerm = GpuActivation.tryLog(log1pArg);
        if (logTerm == null) logTerm = vc.universalOperate(log1pArg, UniversalOperation.LOG, 0);

        // lossPerElem = posPart - xTarget + logTerm
        double[] lossElem = vc.binaryOperate(posPart, xTarget, BinaryOperation.SUBTRACT);
        lossElem = vc.binaryOperate(lossElem, logTerm, BinaryOperation.ADD);
        loss[0] = vc.reduceOperate(lossElem, ReduceOperation.SUM) / n;

        if (!tensor.requiresGrad) {
            return new ConstantDiffTensor(new RereDoubleTensor(loss, new int[]{1}));
        }

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpT = self.inputs.get(1);
            double[] g = self.grad;
            double scale = g[0] / n;

            // probs = sigmoid(logits)
            double[] probs = GpuActivation.trySigmoid(logits);
            if (probs == null) probs = vc.universalOperate(logits, UniversalOperation.SIGMOID, 0);

            // dx = scale * (probs - target)
            double[] diff = vc.binaryOperate(probs, td, BinaryOperation.SUBTRACT);
            double[] dx = vc.binaryOperate(diff, scale, BinaryOperation.MULTIPLY);

            // dt = -scale * logits
            double[] dt = vc.binaryOperate(logits, -scale, BinaryOperation.MULTIPLY);

            inpX.accGrad(dx);
            inpT.accGrad(dt);
        };

        return new RereDiffTensor(loss, new int[]{1}, List.of(tensor, tgt), bw, "bceWithLogitsLoss");
    }
}
