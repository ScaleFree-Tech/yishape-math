package com.yishape.lab.math.compute;

/**
 *
 * @author lteb2
 */
public class ComputerConfig {
    public static final boolean USE_SIMD = false;
    public static final boolean USE_GPU = false;
    //矩阵或者向量元素多于此数就使用GPU
    public static final int GPU_THRESHOLD = 10000000; // 10M elements - scalar operations need much higher threshold due to overhead
    
    // Operation-specific thresholds based on performance analysis
    public static final int GPU_MATRIX_SCALAR_THRESHOLD = 50000000; // 50M elements - scalar ops have high overhead
    public static final int GPU_MATRIX_MULTIPLY_THRESHOLD = 1000000;  // 1M elements - complex ops benefit more
    public static final int GPU_VECTOR_THRESHOLD = 10000000;          // 10M elements - vector ops overhead
    
    // Minimum thresholds for any GPU operation
    public static final int MIN_GPU_ELEMENTS = 1000000; // Never use GPU below 1M elements 
    
    
    /**
     * 检测SIMD是否被支持
     * @return 
     */
    public static boolean checkIfSIMDSupported(){
        try {
            // 使用Class.forName延迟加载SIMDDoubleComputer类，避免在类初始化时抛出NoClassDefFoundError
            Class<?> simdClass = Class.forName("com.yishape.lab.math.compute.SIMDDoubleComputer");
            // 通过反射调用checkIfSupport方法
            return (Boolean) simdClass.getMethod("checkIfSupport").invoke(null);
        } catch (Exception e) {
            return false;
        }
    
    }
    
    /**
     * 检测GPU是否被支持
     * @return 
     */
    public static boolean checkIfGPUSupported(){
        try {
            // 使用Class.forName延迟加载GPUDoubleComputer类，避免在类初始化时抛出NoClassDefFoundError
            Class<?> gpuClass = Class.forName("com.yishape.lab.math.compute.GPUDoubleComputer");
            // 通过反射调用isGPUAvailable方法
            return (Boolean) gpuClass.getMethod("isGPUAvailable").invoke(null);
        } catch (Exception e) {
            return false;
        }
    
    }
}