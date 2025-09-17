package com.reremouse.lab.math.timeseries;
import com.reremouse.lab.math.signal.SignalAnalysis;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.signal.SignalFiltering;

/**
 * 测试SignalAnalysis.movingAverage方法
 * Test SignalAnalysis.movingAverage method
 */
public class test_moving_average {
    public static void main(String[] args) {
        // 创建测试信号：包含噪声的正弦波
        // Create test signal: sine wave with noise
        int length = 20;
        IVector<Double> signal = Linalg.zeros(length);
        
        // 生成测试信号：sin(2πt) + 噪声
        // Generate test signal: sin(2πt) + noise
        for (int i = 0; i < length; i++) {
            double t = (double) i / length;
            double sineValue = Math.sin(2 * Math.PI * t);
            double noise = 0.3 * (Math.random() - 0.5); // 添加随机噪声
            signal.set(i, sineValue + noise);
        }
        
        System.out.println("原始信号 / Original Signal:");
        for (int i = 0; i < length; i++) {
            System.out.printf("signal[%d] = %.3f\n", i, signal.get(i));
        }
        
        // 测试不同的窗口大小
        // Test different window sizes
        int[] windowSizes = {3, 5, 7};
        
        for (int windowSize : windowSizes) {
            System.out.printf("\n窗口大小 / Window Size: %d\n", windowSize);
            System.out.println("移动平均结果 / Moving Average Result:");
            
            try {
                IVector<Double> smoothed = SignalFiltering.movingAverage(signal, windowSize);
                
                for (int i = 0; i < smoothed.length(); i++) {
                    System.out.printf("smoothed[%d] = %.3f\n", i, smoothed.get(i));
                }
            } catch (Exception e) {
                System.out.println("错误 / Error: " + e.getMessage());
            }
        }
        
        // 测试边界情况
        // Test edge cases
        System.out.println("\n测试边界情况 / Testing Edge Cases:");
        
        // 测试窗口大小等于信号长度
        // Test window size equals signal length
        try {
            IVector<Double> result1 = SignalFiltering.movingAverage(signal, length);
            System.out.println("窗口大小等于信号长度测试通过 / Window size equals signal length test passed");
        } catch (Exception e) {
            System.out.println("窗口大小等于信号长度测试失败 / Window size equals signal length test failed: " + e.getMessage());
        }
        
        // 测试无效窗口大小
        // Test invalid window sizes
        try {
            SignalFiltering.movingAverage(signal, 0);
            System.out.println("无效窗口大小测试失败 / Invalid window size test failed");
        } catch (Exception e) {
            System.out.println("无效窗口大小测试通过 / Invalid window size test passed: " + e.getMessage());
        }
        
        try {
            SignalFiltering.movingAverage(signal, length + 1);
            System.out.println("窗口大小大于信号长度测试失败 / Window size greater than signal length test failed");
        } catch (Exception e) {
            System.out.println("窗口大小大于信号长度测试通过 / Window size greater than signal length test passed: " + e.getMessage());
        }
    }
}
