package com.reremouse.lab.audio.core;

import com.reremouse.lab.audio.core.AudioData;
import com.reremouse.lab.audio.core.AudioUtil;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.signal.core.Complex;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.HashMap;
import java.util.Map;

/**
 * MFCC功能测试类
 * MFCC functionality test class
 */
public class AudioUtilityMFCCtest {
    
    /**
     * 测试梅尔滤波器组计算
     * Test Mel filter bank calculation
     */
    @Test
    public void testCalculateMelFilters() {
        // 创建一个简单的测试信号 / Create a simple test signal
        int sampleRate = 44100;
        int fftSize = 1024;
        Complex[] spectrum = new Complex[fftSize];
        
        // 创建一个包含几个频率分量的测试频谱 / Create a test spectrum with several frequency components
        for (int i = 0; i < fftSize; i++) {
            spectrum[i] = new Complex(0, 0);
        }
        
        // 在几个频段添加能量 / Add energy at several frequency bands
        spectrum[10] = new Complex(100, 0);  // 低频 / Low frequency
        spectrum[50] = new Complex(200, 0);  // 中频 / Mid frequency
        spectrum[100] = new Complex(150, 0); // 中高频 / Mid-high frequency
        spectrum[200] = new Complex(80, 0);  // 高频 / High frequency
        
        // 计算梅尔滤波器 / Calculate Mel filters
        int filterCount = 13;
        double[] melFilters = AudioUtil.calculateMelFilters(spectrum, sampleRate, filterCount);
        
        // 验证输出 / Verify output
        assertNotNull(melFilters, "Mel filters should not be null");
        assertEquals(filterCount, melFilters.length, "Filter count should match");
        
        // 验证所有值都是有限的 / Verify all values are finite
        for (int i = 0; i < melFilters.length; i++) {
            assertTrue(Double.isFinite(melFilters[i]), "Mel filter " + i + " should be finite");
        }
        
        System.out.println("Mel filter outputs:");
        for (int i = 0; i < melFilters.length; i++) {
            System.out.println("Filter " + i + ": " + melFilters[i]);
        }
    }
    
    /**
     * 测试MFCC帧计算
     * Test MFCC frame calculation
     */
    @Test
    public void testCalculateMFCCFrames() {
        try {
            // 创建测试音频数据 / Create test audio data
            int sampleRate = 44100;
            int duration = 1; // 1秒 / 1 second
            int sampleCount = sampleRate * duration;
            
            // 创建一个简单的正弦波信号 / Create a simple sine wave signal
            Double[] samples = new Double[sampleCount];
            double frequency = 440.0; // A4音符 / A4 note
            for (int i = 0; i < sampleCount; i++) {
                samples[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate);
            }
            
            AudioData audioData = new AudioData(
                com.reremouse.lab.math.linalg.Linalg.vector(samples),
                sampleRate,
                1,  // 单声道 / Mono
                16, // 16位 / 16 bit
                com.reremouse.lab.audio.core.AudioFormat.WAV
            );
            
            // 设置参数 / Set parameters
            Map<String, Object> parameters = new HashMap<>();
            parameters.put("windowSize", 1024);
            parameters.put("hopSize", 512);
            
            int mfccCount = 13;
            
            // 计算MFCC帧 / Calculate MFCC frames
            double[][] mfccFrames = AudioUtil.calculateMFCCFrames(audioData, parameters, mfccCount);
            
            // 验证输出 / Verify output
            assertNotNull(mfccFrames, "MFCC frames should not be null");
            assertTrue(mfccFrames.length > 0, "Should have at least one frame");
            assertEquals(mfccCount, mfccFrames[0].length, "MFCC count should match");
            
            // 验证所有值都是有限的 / Verify all values are finite
            for (int i = 0; i < mfccFrames.length; i++) {
                for (int j = 0; j < mfccFrames[i].length; j++) {
                    assertTrue(Double.isFinite(mfccFrames[i][j]), 
                              "MFCC[" + i + "][" + j + "] should be finite");
                }
            }
            
            System.out.println("MFCC frames calculated: " + mfccFrames.length);
            System.out.println("MFCC coefficients per frame: " + mfccFrames[0].length);
            System.out.println("First frame MFCC values:");
            for (int i = 0; i < Math.min(mfccCount, 10); i++) {
                System.out.println("MFCC " + i + ": " + mfccFrames[0][i]);
            }
            
        } catch (Exception e) {
            fail("Exception occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试MFCC矩阵计算
     * Test MFCC matrix calculation
     */
    @Test
    public void testCalculateMFCCMatrix() {
        try {
            // 创建测试音频数据 / Create test audio data
            int sampleRate = 22050;
            int duration = 2; // 2秒 / 2 seconds
            int sampleCount = sampleRate * duration;
            
            // 创建一个扫频信号 / Create a sweep signal
            Double[] samples = new Double[sampleCount];
            for (int i = 0; i < sampleCount; i++) {
                double t = (double) i / sampleRate;
                double frequency = 100 + (1000 - 100) * t / duration; // 100Hz到1000Hz的扫频 / Sweep from 100Hz to 1000Hz
                samples[i] = Math.sin(2 * Math.PI * frequency * t);
            }
            
            AudioData audioData = new AudioData(
                com.reremouse.lab.math.linalg.Linalg.vector(samples),
                sampleRate,
                1,  // 单声道 / Mono
                16, // 16位 / 16 bit
                com.reremouse.lab.audio.core.AudioFormat.WAV
            );
            
            int windowSize = 512;
            int hopSize = 256;
            
            int mfccCount = 13;
            
            // 计算MFCC矩阵 / Calculate MFCC matrix
            IMatrix<Double> mfccMatrix = AudioUtil.calculateMFCCMatrix(audioData, mfccCount,windowSize,hopSize);
            
            // 验证输出 / Verify output
            assertNotNull(mfccMatrix, "MFCC matrix should not be null");
            assertTrue(mfccMatrix.rows() > 0, "Should have at least one row");
            assertEquals(mfccCount, mfccMatrix.cols(), "MFCC count should match");
            
            // 验证所有值都是有限的 / Verify all values are finite
            for (int i = 0; i < mfccMatrix.rows(); i++) {
                for (int j = 0; j < mfccMatrix.cols(); j++) {
                    assertTrue(Double.isFinite(mfccMatrix.get(i, j)), 
                              "MFCC[" + i + "][" + j + "] should be finite");
                }
            }
            
            System.out.println("MFCC matrix rows: " + mfccMatrix.rows());
            System.out.println("MFCC matrix columns: " + mfccMatrix.cols());
            System.out.println("First row MFCC values:");
            for (int i = 0; i < Math.min(mfccCount, 10); i++) {
                System.out.println("MFCC " + i + ": " + mfccMatrix.get(0, i));
            }
            
        } catch (Exception e) {
            fail("Exception occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }
}