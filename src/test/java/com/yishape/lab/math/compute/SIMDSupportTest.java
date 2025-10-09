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
        // 这个测试验证即使在没有启用Vector API的情况下，
        // checkIfSIMDSupported方法也能正常工作并返回false
        try {
            boolean supported = ComputerConfig.checkIfSIMDSupported();
            // 无论是否支持SIMD，方法都应该正常返回，不会抛出异常
            assertTrue(true); // 确保测试方法执行完成
        } catch (Throwable t) {
            // 如果捕获到任何异常或错误，测试失败
            fail("checkIfSIMDSupported should not throw exceptions, but threw: " + t.getClass().getName() + ": " + t.getMessage());
        }
    }
    
    /**
     * 测试在没有Aparapi库的情况下，checkIfGPUSupported方法是否能正确返回false
     * 而不是抛出NoClassDefFoundError异常
     */
    @Test
    public void testGPUSupportWithoutAparapi() {
        // 这个测试验证即使在没有Aparapi库的情况下，
        // checkIfGPUSupported方法也能正常工作并返回false
        try {
            boolean supported = ComputerConfig.checkIfGPUSupported();
            // 无论是否支持GPU，方法都应该正常返回，不会抛出异常
            assertTrue(true); // 确保测试方法执行完成
        } catch (Throwable t) {
            // 如果捕获到任何异常或错误，测试失败
            fail("checkIfGPUSupported should not throw exceptions, but threw: " + t.getClass().getName() + ": " + t.getMessage());
        }
    }
    
    /**
     * 测试DoubleVectorComputer类是否能正常初始化
     * 而不会因为SIMD或GPU支持检测而抛出异常
     */
    @Test
    public void testDoubleVectorComputerInitialization() {
        try {
            // 尝试创建DoubleVectorComputer实例
            DoubleVectorComputer computer = new DoubleVectorComputer();
            // 如果能正常创建实例，说明类初始化成功
            assertNotNull(computer);
        } catch (Throwable t) {
            // 如果捕获到任何异常或错误，测试失败
            fail("DoubleVectorComputer should initialize without exceptions, but threw: " + t.getClass().getName() + ": " + t.getMessage());
        }
    }
}