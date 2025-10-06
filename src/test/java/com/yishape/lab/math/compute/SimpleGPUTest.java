package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SimpleGPUTest {
    
    @Test
    public void testBasicGPUOperation() {
        GPUDoubleComputer gpuComputer = new GPUDoubleComputer();
        
        // Test vector addition
        double[] x1 = {1.0, 2.0, 3.0};
        double[] x2 = {4.0, 5.0, 6.0};
        
        double[] result = gpuComputer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.ADD);
        
        assertArrayEquals(new double[]{5.0, 7.0, 9.0}, result, 0.001);
        
        System.out.println("Vector addition test passed");
    }
    
    @Test
    public void testUniversalOperation() {
        GPUDoubleComputer gpuComputer = new GPUDoubleComputer();
        
        // Test universal operation (sqrt)
        double[] x = {4.0, 9.0, 16.0};
        
        double[] result = gpuComputer.universalOperate(x, IDoubleVectorComputer.UniversalOperation.SQRT, 0.0);
        
        assertArrayEquals(new double[]{2.0, 3.0, 4.0}, result, 0.001);
        
        System.out.println("Universal operation test passed");
    }
}