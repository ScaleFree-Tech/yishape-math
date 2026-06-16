package com.yishape.lab.math.autodiff.loss;

import com.yishape.lab.math.autodiff.IDiffTensor;

/**
 * Static facade for loss functions, providing a unified entry point for ML code.
 *
 * <p>Usage: {@code IDiffTensor loss = LossFunctions.bce(pred, target);}</p>
 *
 * <p>All methods delegate to the corresponding instance methods on {@link IDiffTensor},
 * ensuring consistent behavior whether called through the facade or directly.</p>
 */
public final class LossFunctions {
    private LossFunctions() { /* utility class */ }

    // --- Standard losses (on IDiffTensor) ---

    /** Mean Squared Error: mean((pred - target)&sup2;) */
    public static IDiffTensor mse(IDiffTensor pred, IDiffTensor target) {
        return pred.mseLoss(target);
    }

    /** Mean Absolute Error: mean(|pred - target|) */
    public static IDiffTensor l1(IDiffTensor pred, IDiffTensor target) {
        return pred.l1Loss(target);
    }

    /** KL Divergence: target * (log(target) - log(pred)) */
    public static IDiffTensor klDiv(IDiffTensor pred, IDiffTensor target) {
        return pred.klDivLoss(target);
    }

    // --- Classification losses (in autodiff.loss package) ---

    /** Smooth L1 Loss with given beta threshold */
    public static IDiffTensor smoothL1(IDiffTensor pred, IDiffTensor target, double beta) {
        return pred.smoothL1Loss(target, beta);
    }

    /** Binary Cross Entropy (input should be probabilities after sigmoid) */
    public static IDiffTensor bce(IDiffTensor pred, IDiffTensor target) {
        return pred.bceLoss(target);
    }

    /** BCE with Logits (numerically stable, input is raw logits) */
    public static IDiffTensor bceWithLogits(IDiffTensor pred, IDiffTensor target) {
        return pred.bceWithLogitsLoss(target);
    }

    /** Focal Loss for class-imbalanced classification */
    public static IDiffTensor focal(IDiffTensor pred, IDiffTensor target, double alpha, double gamma) {
        return pred.focalLoss(target, alpha, gamma);
    }

    /** Dice Loss for segmentation tasks */
    public static IDiffTensor dice(IDiffTensor pred, IDiffTensor target, double smooth) {
        return pred.diceLoss(target, smooth);
    }

    /** Negative Log-Likelihood Loss (input should be log-probabilities) */
    public static IDiffTensor nll(IDiffTensor pred, IDiffTensor target, int classDim) {
        return pred.nllLoss(target, classDim);
    }

    // --- CTC Loss ---

    /**
     * CTC Loss for sequence labeling (speech recognition, OCR, etc.).
     *
     * @param pred         log-probabilities [T, C]
     * @param targets      padded label sequences [B, L]
     * @param inputLengths actual sequence lengths [B]
     * @param targetLengths label lengths [B]
     */
    public static IDiffTensor ctc(IDiffTensor pred, IDiffTensor targets,
                                  IDiffTensor inputLengths, IDiffTensor targetLengths) {
        return pred.ctcLoss(targets, inputLengths, targetLengths);
    }
}
