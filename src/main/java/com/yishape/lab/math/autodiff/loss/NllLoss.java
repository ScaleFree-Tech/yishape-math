package com.yishape.lab.math.autodiff.loss;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;

/**
 * Negative Log-Likelihood Loss.
 *
 * <p>Input is log-probabilities. Computes -mean(gather(logProbs, classDim, target)).</p>
 * <p>Delegates to IDiffTensor gather/sum/div/neg composition.</p>
 */
public final class NllLoss {
    private NllLoss() { /* utility class */ }

    public static IDiffTensor apply(RereDiffTensor tensor, IDiffTensor target, int classDim) {
        int d = (classDim < 0 ? classDim + tensor.rank() : classDim);
        RereDiffTensor tgt = (RereDiffTensor) target;
        IDiffTensor gathered = tensor.gather(d, tgt);
        IDiffTensor loss = gathered.sum().div(gathered.totalSize()).neg();
        if (loss instanceof RereDiffTensor rt) {
            rt.setOpTag("nllLoss");
        }
        return loss;
    }
}
