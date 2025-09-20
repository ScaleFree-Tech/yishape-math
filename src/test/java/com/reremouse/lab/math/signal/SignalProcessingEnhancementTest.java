package com.reremouse.lab.math.signal;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.signal.core.SignalProcessingException;
import com.reremouse.lab.math.signal.transform.ZTransform;
import com.reremouse.lab.math.signal.transform.ChirpZTransform;
import com.reremouse.lab.math.signal.transform.WalshHadamardTransform;
import com.reremouse.lab.math.signal.filter.EllipticFilter;
import com.reremouse.lab.math.signal.filter.ChebyshevFilter;
import com.reremouse.lab.math.signal.filter.ISignalFilter;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 信号处理增强功能测试类 / Signal Processing Enhancement Test Class
 * <p>
 * 测试新增的信号处理功能，包括高级变换和滤波算法。
 * Test new signal processing features including advanced transforms and filtering algorithms.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class SignalProcessingEnhancementTest {
    
    private IVector<Double> testSignal;
    private static final double TOLERANCE = 1e-10;
    
    @BeforeEach
    void setUp() {
        // 创建测试信号：正弦波 + 噪声 / Create test signal: sine wave + noise
        int N = 64;
        double[] signal = new double[N];
        double fs = 1000.0; // 采样率 / Sampling rate
        double f1 = 50.0;   // 信号频率 / Signal frequency
        
        for (int i = 0; i < N; i++) {
            double t = i / fs;
            signal[i] = Math.sin(2 * Math.PI * f1 * t) + 0.1 * Math.sin(2 * Math.PI * 200 * t);
        }
        
        testSignal = Linalg.vector(signal);
    }
    
    @Test
    void testZTransform() throws SignalProcessingException {
        System.out.println("=== Z变换测试 / Z-Transform Test ===");
        
        ZTransform zt = new ZTransform();
        
        // 测试正向变换 / Test forward transform
        Complex[] zResult = zt.forward(testSignal);
        assertNotNull(zResult);
        assertEquals(512, zResult.length); // 默认评估点数
        
        // 测试逆变换 / Test inverse transform
        IVector<Double> reconstructed = zt.inverse(zResult);
        assertNotNull(reconstructed);
        
        System.out.println("Z变换完成，结果长度: " + zResult.length);
        System.out.println("重建信号长度: " + reconstructed.length());
        
        // 验证变换的基本性质 / Verify basic properties of transform
        for (Complex c : zResult) {
            assertFalse(Double.isNaN(c.real));
            assertFalse(Double.isNaN(c.imag));
        }
    }
    
    @Test
    void testChirpZTransform() throws SignalProcessingException {
        System.out.println("=== Chirp-Z变换测试 / Chirp-Z Transform Test ===");
        
        // 创建Chirp-Z变换用于特定频率范围分析
        ChirpZTransform czt = ChirpZTransform.forFrequencyRange(0.01, 0.4, 128);
        
        // 测试正向变换 / Test forward transform
        Complex[] cztResult = czt.forward(testSignal);
        assertNotNull(cztResult);
        assertEquals(128, cztResult.length);
        
        // 测试逆变换 / Test inverse transform
        IVector<Double> reconstructed = czt.inverse(cztResult);
        assertNotNull(reconstructed);
        
        System.out.println("Chirp-Z变换完成，结果长度: " + cztResult.length);
        System.out.println("重建信号长度: " + reconstructed.length());
        
        // 验证频谱的正确性 / Verify spectrum correctness
        double maxMagnitude = 0;
        int maxIndex = 0;
        for (int i = 0; i < cztResult.length; i++) {
            double magnitude = cztResult[i].magnitude();
            if (magnitude > maxMagnitude) {
                maxMagnitude = magnitude;
                maxIndex = i;
            }
        }
        
        System.out.println("最大幅度索引: " + maxIndex + ", 幅度: " + maxMagnitude);
        assertTrue(maxMagnitude > 0);
    }
    
    @Test
    void testWalshHadamardTransform() throws SignalProcessingException {
        System.out.println("=== Walsh-Hadamard变换测试 / Walsh-Hadamard Transform Test ===");
        
        WalshHadamardTransform wht = new WalshHadamardTransform(
            WalshHadamardTransform.WalshOrdering.NATURAL, true);
        
        // 测试正向变换 / Test forward transform
        IVector<Double> whtResult = wht.forward(testSignal);
        assertNotNull(whtResult);
        assertEquals(testSignal.length(), whtResult.length());
        
        // 测试逆变换 / Test inverse transform
        IVector<Double> reconstructed = wht.inverse(whtResult);
        assertNotNull(reconstructed);
        assertEquals(testSignal.length(), reconstructed.length());
        
        System.out.println("Walsh-Hadamard变换完成");
        System.out.println("原始信号长度: " + testSignal.length());
        System.out.println("变换结果长度: " + whtResult.length());
        System.out.println("重建信号长度: " + reconstructed.length());
        
        // 验证变换的自逆性质（考虑数值误差）/ Verify self-inverse property (considering numerical errors)
        double maxError = 0;
        for (int i = 0; i < testSignal.length(); i++) {
            double error = Math.abs(testSignal.get(i) - reconstructed.get(i));
            maxError = Math.max(maxError, error);
        }
        System.out.println("重建最大误差: " + maxError);
        assertTrue(maxError < 1e-10); // 验证重建精度
    }
    
    @Test
    void testEllipticFilter() throws SignalProcessingException {
        System.out.println("=== 椭圆滤波器测试 / Elliptic Filter Test ===");
        
        // 创建椭圆低通滤波器 / Create elliptic low-pass filter
        EllipticFilter filter = new EllipticFilter(4, 100.0, 1000.0, 0.5, 40.0);
        
        // 验证滤波器属性 / Verify filter properties
        assertEquals(ISignalFilter.FilterType.LOW_PASS, filter.getFilterType());
        assertEquals(ISignalFilter.FilterImplementation.ELLIPTIC, filter.getImplementationType());
        assertEquals(4, filter.getOrder());
        assertEquals(1000.0, filter.getSamplingRate());
        
        // 测试滤波功能 / Test filtering function
        IVector<Double> filtered = filter.filter(testSignal);
        assertNotNull(filtered);
        assertEquals(testSignal.length(), filtered.length());
        
        System.out.println("椭圆滤波器测试完成");
        System.out.println("滤波器阶数: " + filter.getOrder());
        System.out.println("截止频率: " + filter.getCutoffFrequencies()[0] + " Hz");
        System.out.println("通带波纹: " + filter.getPassbandRipple() + " dB");
        System.out.println("阻带波纹: " + filter.getStopbandRipple() + " dB");
        
        // 验证滤波效果 / Verify filtering effect
        double originalEnergy = 0, filteredEnergy = 0;
        for (int i = 0; i < testSignal.length(); i++) {
            originalEnergy += testSignal.get(i) * testSignal.get(i);
            filteredEnergy += filtered.get(i) * filtered.get(i);
        }
        
        System.out.println("原始信号能量: " + originalEnergy);
        System.out.println("滤波后能量: " + filteredEnergy);
        assertTrue(filteredEnergy > 0); // 滤波后仍有信号
        assertTrue(filteredEnergy <= originalEnergy); // 能量不增加
    }
    
    @Test
    void testChebyshevFilter() throws SignalProcessingException {
        System.out.println("=== 切比雪夫滤波器测试 / Chebyshev Filter Test ===");
        
        // 创建切比雪夫I型滤波器 / Create Chebyshev Type I filter
        ChebyshevFilter filter = new ChebyshevFilter(
            ChebyshevFilter.ChebyshevType.TYPE_I, 3, 80.0, 1000.0, 1.0);
        
        // 验证滤波器属性 / Verify filter properties
        assertEquals(ISignalFilter.FilterType.LOW_PASS, filter.getFilterType());
        assertEquals(ISignalFilter.FilterImplementation.CHEBYSHEV_I, filter.getImplementationType());
        assertEquals(3, filter.getOrder());
        assertEquals(ChebyshevFilter.ChebyshevType.TYPE_I, filter.getChebyshevType());
        
        // 测试滤波功能 / Test filtering function
        IVector<Double> filtered = filter.filter(testSignal);
        assertNotNull(filtered);
        assertEquals(testSignal.length(), filtered.length());
        
        System.out.println("切比雪夫滤波器测试完成");
        System.out.println("滤波器类型: " + filter.getChebyshevType().getEnglishName());
        System.out.println("滤波器阶数: " + filter.getOrder());
        System.out.println("截止频率: " + filter.getCutoffFrequencies()[0] + " Hz");
        System.out.println("波纹: " + filter.getRipple() + " dB");
        
        // 计算滤波前后的统计特性 / Calculate statistics before and after filtering
        double originalMean = testSignal.mean();
        double filteredMean = filtered.mean();
        double originalStd = testSignal.std();
        double filteredStd = filtered.std();
        
        System.out.println("原始信号均值: " + originalMean + ", 标准差: " + originalStd);
        System.out.println("滤波后均值: " + filteredMean + ", 标准差: " + filteredStd);
    }
    
    @Test
    void testFrequencyResponse() throws SignalProcessingException {
        System.out.println("=== 频率响应测试 / Frequency Response Test ===");
        
        EllipticFilter filter = new EllipticFilter(4, 100.0, 1000.0, 0.5, 40.0);
        
        // 测试频率点 / Test frequency points
        double[] frequencies = {1, 10, 50, 100, 150, 200, 300, 400, 500};
        
        ISignalFilter.FrequencyResponse response = filter.getFrequencyResponse(frequencies);
        
        assertNotNull(response);
        assertEquals(frequencies.length, response.getFrequencies().length);
        assertEquals(frequencies.length, response.getMagnitude().length);
        assertEquals(frequencies.length, response.getPhase().length);
        
        System.out.println("频率响应计算完成");
        System.out.println("频率(Hz)\t幅度\t\t相位(rad)\t幅度(dB)");
        
        double[] magnitude = response.getMagnitude();
        double[] phase = response.getPhase();
        double[] magnitudeDB = response.getMagnitudeDB();
        
        for (int i = 0; i < frequencies.length; i++) {
            System.out.printf("%.1f\t\t%.6f\t%.6f\t%.2f%n", 
                frequencies[i], magnitude[i], phase[i], magnitudeDB[i]);
        }
        
        // 验证低通特性：低频幅度 > 高频幅度 / Verify low-pass characteristics
        assertTrue(magnitude[0] > magnitude[magnitude.length - 1]);
    }
    
    @Test
    void testSignalProcessorInterface() throws SignalProcessingException {
        System.out.println("=== 信号处理器接口测试 / Signal Processor Interface Test ===");
        
        // 测试接口的统一性 / Test interface uniformity
        ZTransform zt = new ZTransform();
        EllipticFilter filter = new EllipticFilter(4, 100.0, 1000.0, 0.5, 40.0);
        
        // 通过接口调用 / Call through interface
        IVector<Double> ztResult = zt.process(testSignal);
        IVector<Double> filterResult = filter.process(testSignal);
        
        assertNotNull(ztResult);
        assertNotNull(filterResult);
        
        System.out.println("Z变换处理器名称: " + zt.getName());
        System.out.println("椭圆滤波器名称: " + filter.getName());
        System.out.println("Z变换版本: " + zt.getVersion());
        System.out.println("椭圆滤波器版本: " + filter.getVersion());
        
        // 测试克隆功能 / Test clone functionality
        ZTransform ztClone = zt.clone();
        EllipticFilter filterClone = filter.clone();
        
        assertNotNull(ztClone);
        assertNotNull(filterClone);
        assertNotSame(zt, ztClone);
        assertNotSame(filter, filterClone);
        
        System.out.println("接口测试完成");
    }
    
    @Test
    void testParameterValidation() {
        System.out.println("=== 参数验证测试 / Parameter Validation Test ===");
        
        // 测试无效参数的异常处理 / Test exception handling for invalid parameters
        
        // 无效滤波器阶数 / Invalid filter order
        assertThrows(SignalProcessingException.class, () -> {
            new EllipticFilter(0, 100.0, 1000.0, 0.5, 40.0);
        });
        
        // 无效截止频率 / Invalid cutoff frequency
        assertThrows(SignalProcessingException.class, () -> {
            new EllipticFilter(4, 600.0, 1000.0, 0.5, 40.0); // 超过Nyquist频率
        });
        
        // 无效波纹参数 / Invalid ripple parameter
        assertThrows(SignalProcessingException.class, () -> {
            new EllipticFilter(4, 100.0, 1000.0, -0.5, 40.0);
        });
        
        System.out.println("参数验证测试完成 - 所有无效参数都被正确捕获");
    }
}