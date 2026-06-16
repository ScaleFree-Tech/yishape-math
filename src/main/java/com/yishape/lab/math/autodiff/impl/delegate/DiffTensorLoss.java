package com.yishape.lab.math.autodiff.impl.delegate;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.loss.SmoothL1Loss;
import com.yishape.lab.math.autodiff.loss.BceLoss;
import com.yishape.lab.math.autodiff.loss.BceWithLogitsLoss;
import com.yishape.lab.math.autodiff.loss.FocalLoss;
import com.yishape.lab.math.autodiff.loss.DiceLoss;
import com.yishape.lab.math.autodiff.loss.NllLoss;

/**
 * Thin proxy to {@code com.yishape.lab.math.autodiff.loss} package.
 *
 * <p>All implementations moved to dedicated classes in {@code autodiff.loss}.
 * This class preserves the original static method signatures so that
 * {@link RereDiffTensor} instance methods ({@code smoothL1Loss}, {@code bceLoss}, etc.)
 * continue to work without changes.</p>
 */
public final class DiffTensorLoss {
    private DiffTensorLoss() { /* utility class */ }

    public static IDiffTensor smoothL1Loss(RereDiffTensor tensor, IDiffTensor target, double beta) {
        return SmoothL1Loss.apply(tensor, target, beta);
    }

    public static IDiffTensor bceLoss(RereDiffTensor tensor, IDiffTensor target) {
        return BceLoss.apply(tensor, target);
    }

    public static IDiffTensor bceWithLogitsLoss(RereDiffTensor tensor, IDiffTensor target) {
        return BceWithLogitsLoss.apply(tensor, target);
    }

    public static IDiffTensor focalLoss(RereDiffTensor tensor, IDiffTensor target,
                                         double alpha, double gamma) {
        return FocalLoss.apply(tensor, target, alpha, gamma);
    }

    public static IDiffTensor diceLoss(RereDiffTensor tensor, IDiffTensor target, double smooth) {
        return DiceLoss.apply(tensor, target, smooth);
    }

    public static IDiffTensor nllLoss(RereDiffTensor tensor, IDiffTensor target, int classDim) {
        return NllLoss.apply(tensor, target, classDim);
    }
}
