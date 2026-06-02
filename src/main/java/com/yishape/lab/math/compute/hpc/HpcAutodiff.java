package com.yishape.lab.math.compute.hpc;

/**
 * AD HPC bridge: attempts to offload computation graph execution to native runtime.
 * Returns {@code null} on any failure to trigger automatic Java fallback.
 *
 * <p>Follows the same reflection-based pattern as {@link HpcGemm} and {@link HpcLapackDecomps}.
 * Gradient application to leaf nodes is handled by {@code HpcGraphExecutor} in the
 * {@code autodiff.impl} package (which has package-private access to {@code accGrad}).
 */
public final class HpcAutodiff {

    private HpcAutodiff() {
    }

    /**
     * Attempt to execute a computation graph via native runtime.
     *
     * @param json pre-serialized JSON graph from {@code GraphExporter.toJson()}
     * @return array of [loss_array, grad0, grad1, ...] or {@code null} on failure
     */
    public static double[][] tryExecute(String json) {
        if (json == null || !HpcConfig.allowAttempts()) {
            return null;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return null;
        }
        try {
            return HpcOptionalRuntime.tryExecuteGraph(json);
        } catch (Throwable t) {
            return null;
        }
    }
}
