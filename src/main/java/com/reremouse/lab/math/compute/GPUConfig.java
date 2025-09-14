package com.reremouse.lab.math.compute;

/**
 *
 * @author lteb2
 */
public class GPUConfig {
    //矩阵或者向量元素多于此数就使用GPU
    public static final int GPU_THRESHOLD = 10000000; // 10M elements - scalar operations need much higher threshold due to overhead
    
    // Operation-specific thresholds based on performance analysis
    public static final int GPU_MATRIX_SCALAR_THRESHOLD = 50000000; // 50M elements - scalar ops have high overhead
    public static final int GPU_MATRIX_MULTIPLY_THRESHOLD = 1000000;  // 1M elements - complex ops benefit more
    public static final int GPU_VECTOR_THRESHOLD = 10000000;          // 10M elements - vector ops overhead
    
    // Minimum thresholds for any GPU operation
    public static final int MIN_GPU_ELEMENTS = 1000000; // Never use GPU below 1M elements 
}
