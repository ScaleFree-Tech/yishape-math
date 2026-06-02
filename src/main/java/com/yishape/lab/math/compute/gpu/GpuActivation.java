package com.yishape.lab.math.compute.gpu;

/**
 * GPU activation dispatch. Supports ops 0-16 via activation.wgsl shader.
 */
public final class GpuActivation {
    private GpuActivation() {}

    public static double[] tryRelu(double[] input) { return tryActivation(0, input); }
    public static double[] tryGelu(double[] input) { return tryActivation(1, input); }
    public static double[] trySigmoid(double[] input) { return tryActivation(2, input); }
    public static double[] tryTanh(double[] input) { return tryActivation(3, input); }
    public static double[] tryElu(double[] input) { return tryActivation(5, input); }
    public static double[] tryLeakyRelu(double[] input) { return tryActivation(6, input); }
    public static double[] trySilu(double[] input) { return tryActivation(7, input); }
    public static double[] trySoftplus(double[] input) { return tryActivation(8, input); }
    public static double[] trySelu(double[] input) { return tryActivation(9, input); }
    public static double[] tryHardtanh(double[] input) { return tryActivation(10, input); }
    public static double[] tryExp(double[] input) { return tryActivation(11, input); }
    public static double[] tryLog(double[] input) { return tryActivation(12, input); }
    public static double[] tryAbs(double[] input) { return tryActivation(13, input); }
    public static double[] trySqrt(double[] input) { return tryActivation(14, input); }
    public static double[] trySin(double[] input) { return tryActivation(15, input); }
    public static double[] tryCos(double[] input) { return tryActivation(16, input); }

    /** Float32 variants. */
    public static float[] tryFloatRelu(float[] input) { return tryFloatActivation(0, input); }
    public static float[] tryFloatGelu(float[] input) { return tryFloatActivation(1, input); }
    public static float[] tryFloatSigmoid(float[] input) { return tryFloatActivation(2, input); }
    public static float[] tryFloatTanh(float[] input) { return tryFloatActivation(3, input); }
    public static float[] tryFloatElu(float[] input) { return tryFloatActivation(5, input); }
    public static float[] tryFloatLeakyRelu(float[] input) { return tryFloatActivation(6, input); }
    public static float[] tryFloatSilu(float[] input) { return tryFloatActivation(7, input); }
    public static float[] tryFloatSoftplus(float[] input) { return tryFloatActivation(8, input); }
    public static float[] tryFloatSelu(float[] input) { return tryFloatActivation(9, input); }
    public static float[] tryFloatHardtanh(float[] input) { return tryFloatActivation(10, input); }
    public static float[] tryFloatExp(float[] input) { return tryFloatActivation(11, input); }
    public static float[] tryFloatLog(float[] input) { return tryFloatActivation(12, input); }
    public static float[] tryFloatAbs(float[] input) { return tryFloatActivation(13, input); }
    public static float[] tryFloatSqrt(float[] input) { return tryFloatActivation(14, input); }
    public static float[] tryFloatSin(float[] input) { return tryFloatActivation(15, input); }
    public static float[] tryFloatCos(float[] input) { return tryFloatActivation(16, input); }

    private static double[] tryActivation(int op, double[] input) {
        if (input == null || input.length == 0) return null;
        if ((long) input.length < GpuConfig.activationMinElements()) return null;
        if (!GpuConfig.allowAttempts()) return null;
        if (!GpuOptionalRuntime.isGpuAvailable()) return null;
        try {
            return switch (op) {
                case 0 -> GpuOptionalRuntime.tryRelu(input);
                case 1 -> GpuOptionalRuntime.tryGelu(input);
                case 2 -> GpuOptionalRuntime.trySigmoid(input);
                case 3 -> GpuOptionalRuntime.tryTanh(input);
                case 5 -> GpuOptionalRuntime.tryElu(input);
                case 6 -> GpuOptionalRuntime.tryLeakyRelu(input);
                case 7 -> GpuOptionalRuntime.trySilu(input);
                case 8 -> GpuOptionalRuntime.trySoftplus(input);
                case 9 -> GpuOptionalRuntime.trySelu(input);
                case 10 -> GpuOptionalRuntime.tryHardtanh(input);
                case 11 -> GpuOptionalRuntime.tryExp(input);
                case 12 -> GpuOptionalRuntime.tryLog(input);
                case 13 -> GpuOptionalRuntime.tryAbs(input);
                case 14 -> GpuOptionalRuntime.trySqrt(input);
                case 15 -> GpuOptionalRuntime.trySin(input);
                case 16 -> GpuOptionalRuntime.tryCos(input);
                default -> null;
            };
        } catch (Throwable t) {
            return null;
        }
    }

    private static float[] tryFloatActivation(int op, float[] input) {
        if (input == null || input.length == 0) return null;
        if ((long) input.length < GpuConfig.activationMinElements()) return null;
        if (!GpuConfig.allowAttempts()) return null;
        if (!GpuOptionalRuntime.isGpuAvailable()) return null;
        try {
            return switch (op) {
                case 0 -> GpuOptionalRuntime.tryFloatRelu(input);
                case 1 -> GpuOptionalRuntime.tryFloatGelu(input);
                case 2 -> GpuOptionalRuntime.tryFloatSigmoid(input);
                case 3 -> GpuOptionalRuntime.tryFloatTanh(input);
                case 5 -> GpuOptionalRuntime.tryFloatElu(input);
                case 6 -> GpuOptionalRuntime.tryFloatLeakyRelu(input);
                case 7 -> GpuOptionalRuntime.tryFloatSilu(input);
                case 8 -> GpuOptionalRuntime.tryFloatSoftplus(input);
                case 9 -> GpuOptionalRuntime.tryFloatSelu(input);
                case 10 -> GpuOptionalRuntime.tryFloatHardtanh(input);
                case 11 -> GpuOptionalRuntime.tryFloatExp(input);
                case 12 -> GpuOptionalRuntime.tryFloatLog(input);
                case 13 -> GpuOptionalRuntime.tryFloatAbs(input);
                case 14 -> GpuOptionalRuntime.tryFloatSqrt(input);
                case 15 -> GpuOptionalRuntime.tryFloatSin(input);
                case 16 -> GpuOptionalRuntime.tryFloatCos(input);
                default -> null;
            };
        } catch (Throwable t) {
            return null;
        }
    }
}
