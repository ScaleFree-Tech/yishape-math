package com.reremouse.lab.math.signal;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.signal.generation.ISignalGenerator;

/**
 * 信号生成测试类 / Signal Generation Test Class
 * <p>
 * 测试新添加的信号生成方法的正确性。
 * Test the correctness of newly added signal generation methods.
 * </p>
 */
public class SignalGenerationTest {
    
    public static void main(String[] args) {
        SignalGenerationTest test = new SignalGenerationTest();
        
        try {
            test.testStepSignal();
            test.testImpulseSignal();
            test.testChirpSignal();
            test.testPulseSignal();
            test.testCompositeSignal();
            test.testAddNoise();
            System.out.println("All signal generation tests passed!");
        } catch (Exception e) {
            System.err.println("Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public void testStepSignal() {
        System.out.println("Testing step signal generation...");
        
        // 测试单位阶跃信号 / Test unit step signal
        IVector<Double> unitStep = Signals.unitStep(1000, 0.5, 1000.0);
        
        if (unitStep == null) {
            throw new RuntimeException("Unit step signal should not be null");
        }
        
        if (unitStep.length() != 1000) {
            throw new RuntimeException("Unit step signal length should be 1000, but was " + unitStep.length());
        }
        
        // 检查阶跃特性 / Check step characteristics
        // 在阶跃时间之前应该是0 / Should be 0 before step time
        // Step time is 0.5 seconds, with 1000 Hz sampling rate, that's index 500
        if (Math.abs(unitStep.get(0)) > 1e-10) {
            throw new RuntimeException("Value before step time should be 0, but was " + unitStep.get(0));
        }
        
        // 在阶跃时间之后应该是1 / Should be 1 after step time
        // Check a point well after the step time (e.g., index 750 which is 0.75 seconds)
        if (Math.abs(unitStep.get(750) - 1.0) > 1e-10) {
            throw new RuntimeException("Value after step time should be 1, but was " + unitStep.get(750));
        }
        
        System.out.println("Step signal test passed");
    }
    
    public void testImpulseSignal() {
        System.out.println("Testing impulse signal generation...");
        
        // 测试单位脉冲信号 / Test unit impulse signal
        IVector<Double> unitImpulse = Signals.unitImpulse(100, 50);
        
        if (unitImpulse == null) {
            throw new RuntimeException("Unit impulse signal should not be null");
        }
        
        if (unitImpulse.length() != 100) {
            throw new RuntimeException("Unit impulse signal length should be 100, but was " + unitImpulse.length());
        }
        
        // 检查脉冲特性 / Check impulse characteristics
        // 在脉冲位置应该是1 / Should be 1 at impulse position
        if (Math.abs(unitImpulse.get(50) - 1.0) > 1e-10) {
            throw new RuntimeException("Value at impulse position should be 1, but was " + unitImpulse.get(50));
        }
        
        // 其他位置应该是0 / Should be 0 at other positions
        if (Math.abs(unitImpulse.get(0)) > 1e-10) {
            throw new RuntimeException("Value before impulse should be 0, but was " + unitImpulse.get(0));
        }
        
        if (Math.abs(unitImpulse.get(99)) > 1e-10) {
            throw new RuntimeException("Value after impulse should be 0, but was " + unitImpulse.get(99));
        }
        
        System.out.println("Impulse signal test passed");
    }
    
    public void testChirpSignal() {
        System.out.println("Testing chirp signal generation...");
        
        // 测试线性调频信号 / Test linear chirp signal
        IVector<Double> chirpSignal = Signals.chirpSignal(1000, 10.0, 100.0, 1000.0, 1.0);
        
        if (chirpSignal == null) {
            throw new RuntimeException("Chirp signal should not be null");
        }
        
        if (chirpSignal.length() != 1000) {
            throw new RuntimeException("Chirp signal length should be 1000, but was " + chirpSignal.length());
        }
        
        // 检查信号的基本特性 / Check basic signal characteristics
        // 信号应该是有界的 / Signal should be bounded
        if (Math.abs(chirpSignal.max()) > 1.0 + 1e-10) {
            throw new RuntimeException("Signal should be bounded, but max value was " + chirpSignal.max());
        }
        
        if (Math.abs(chirpSignal.min()) < -1.0 - 1e-10) {
            throw new RuntimeException("Signal should be bounded, but min value was " + chirpSignal.min());
        }
        
        System.out.println("Chirp signal test passed");
    }
    
    public void testPulseSignal() {
        System.out.println("Testing pulse signal generation...");
        
        // 测试脉冲信号 / Test pulse signal
        IVector<Double> pulseSignal = Signals.pulseSignal(1000, 1.0, 10, 10.0, 1000.0);
        
        if (pulseSignal == null) {
            throw new RuntimeException("Pulse signal should not be null");
        }
        
        if (pulseSignal.length() != 1000) {
            throw new RuntimeException("Pulse signal length should be 1000, but was " + pulseSignal.length());
        }
        
        // 检查信号的基本特性 / Check basic signal characteristics
        // 信号应该是有界的 / Signal should be bounded
        if (Math.abs(pulseSignal.max()) > 1.0 + 1e-10) {
            throw new RuntimeException("Signal should be bounded, but max value was " + pulseSignal.max());
        }
        
        if (pulseSignal.min() < -1e-10) {
            throw new RuntimeException("Signal should be non-negative, but min value was " + pulseSignal.min());
        }
        
        System.out.println("Pulse signal test passed");
    }
    
    public void testCompositeSignal() {
        System.out.println("Testing composite signal generation...");
        
        // 测试复合信号生成 / Test composite signal generation
        ISignalGenerator.SignalType[] signalTypes = {
            ISignalGenerator.SignalType.SINE,
            ISignalGenerator.SignalType.COSINE
        };
        
        ISignalGenerator.SignalParameters[] parameters = {
            new ISignalGenerator.SignalParameters().frequency(10.0).amplitude(1.0).samplingRate(1000.0),
            new ISignalGenerator.SignalParameters().frequency(20.0).amplitude(0.5).samplingRate(1000.0)
        };
        
        IVector<Double> compositeSignal = Signals.compositeSignal(signalTypes, 1000, parameters);
        
        if (compositeSignal == null) {
            throw new RuntimeException("Composite signal should not be null");
        }
        
        if (compositeSignal.length() != 1000) {
            throw new RuntimeException("Composite signal length should be 1000, but was " + compositeSignal.length());
        }
        
        // 生成单独的信号进行比较 / Generate individual signals for comparison
        IVector<Double> sineSignal = Signals.sineWave(1000, 10.0, 1000.0, 1.0, 0.0);
        IVector<Double> cosineSignal = Signals.cosineWave(1000, 20.0, 1000.0, 0.5, 0.0);
        IVector<Double> expectedSignal = sineSignal.add(cosineSignal);
        
        // 检查复合信号是否等于单独信号的和 / Check if composite signal equals sum of individual signals
        boolean signalsMatch = true;
        for (int i = 0; i < 10; i++) { // 检查前10个点 / Check first 10 points
            if (Math.abs(compositeSignal.get(i) - expectedSignal.get(i)) > 1e-10) {
                signalsMatch = false;
                break;
            }
        }
        
        if (!signalsMatch) {
            throw new RuntimeException("Composite signal should equal sum of individual signals");
        }
        
        System.out.println("Composite signal test passed");
    }
    
    public void testAddNoise() {
        System.out.println("Testing add noise functionality...");
        
        // 生成基础信号 / Generate base signal
        IVector<Double> baseSignal = Signals.sineWave(1000, 10.0, 1000.0, 1.0, 0.0);
        
        // 添加噪声 / Add noise
        IVector<Double> noisySignal = Signals.addNoise(
            baseSignal, 
            ISignalGenerator.SignalType.WHITE_NOISE, 
            new ISignalGenerator.SignalParameters().noiseVariance(0.1).samplingRate(1000.0)
        );
        
        if (noisySignal == null) {
            throw new RuntimeException("Noisy signal should not be null");
        }
        
        if (noisySignal.length() != 1000) {
            throw new RuntimeException("Noisy signal length should be 1000, but was " + noisySignal.length());
        }
        
        // 检查噪声是否被添加 / Check if noise was added
        // 噪声信号应该与原信号不同 / Noisy signal should be different from original
        boolean signalsAreDifferent = false;
        for (int i = 0; i < baseSignal.length(); i++) {
            if (Math.abs(noisySignal.get(i) - baseSignal.get(i)) > 1e-10) {
                signalsAreDifferent = true;
                break;
            }
        }
        
        if (!signalsAreDifferent) {
            throw new RuntimeException("Noisy signal should be different from original signal");
        }
        
        System.out.println("Add noise test passed");
    }
}