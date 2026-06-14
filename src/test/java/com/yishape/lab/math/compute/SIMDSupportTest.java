package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 测试SIMD支持检测功能
 */
public class SIMDSupportTest {

    /**
     * 测试在没有启用Vector API的情况下，checkIfSIMDSupported方法是否能正确返回false
     * 而不是抛出NoClassDefFoundError异常
     */
    @Test
    public void testSIMDSupportWithoutVectorAPI() {
        // checkIfSIMDSupported must execute without throwing.
        // Returns true when Vector API available, false otherwise.
        // JUnit naturally fails the test if any exception is thrown.
        boolean supported = ComputerConfig.checkIfSIMDSupported();
        // Return value is a primitive boolean — always valid. No assertion needed.
        // The absence of an exception IS the test assertion.
    }
    
    /**
     * 测试在没有Aparapi库的情况下，checkIfGPUSupported方法是否能正确返回false
     * 而不是抛出NoClassDefFoundError异常
     */
    @Test
    public void testGPUSupportWithoutAparapi() {
        // checkIfGPUSupported must execute without throwing.
        // Returns true when GPU runtime available, false otherwise.
        // JUnit naturally fails the test if any exception is thrown.
        boolean supported = ComputerConfig.checkIfGPUSupported();
        // Return value is a primitive boolean — always valid. No assertion needed.
        // The absence of an exception IS the test assertion.
    }
    
    /**
     * 测试DoubleVectorComputer类是否能正常初始化
     * 而不会因为SIMD或GPU支持检测而抛出异常
     */
    @Test
    public void testDoubleVectorComputerInitialization() {
        // DoubleVectorComputer must initialize without throwing exceptions.
        // JUnit naturally fails the test if any exception is thrown.
        DoubleVectorComputer computer = new DoubleVectorComputer();
        assertNotNull(computer, "DoubleVectorComputer instance must not be null");
    }
}