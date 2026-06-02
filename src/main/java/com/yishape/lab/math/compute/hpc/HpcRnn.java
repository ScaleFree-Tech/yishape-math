package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated fused LSTM gate computation.
 * Follows the same four-gate optional-native pattern as {@link HpcIm2col}.
 */
public final class HpcRnn {

    private HpcRnn() {
    }

    /**
     * Attempt HPC fused LSTM single timestep.
     *
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryLstmFusedStep(double[] input, double[] hidden, double[] cell,
                                           double[] weightI, double[] weightH,
                                           double[] biasI, double[] biasH,
                                           int inputSize, int hiddenSize,
                                           double[] hiddenOut, double[] cellOut) {
        if (input == null || hidden == null || cell == null
                || weightI == null || weightH == null
                || hiddenOut == null || cellOut == null) {
            return false;
        }
        if (inputSize < HpcConfig.rnnMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.lstmFusedStep(
                input, hidden, cell, weightI, weightH, biasI, biasH,
                inputSize, hiddenSize, hiddenOut, cellOut);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }
}
