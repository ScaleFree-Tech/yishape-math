package com.reremouse.lab.math.stats.bayes.inference;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;
import java.util.HashMap;

/**
 * 全面测试默认推理配置的功能
 */
public class DefaultInferenceConfigTest {
    
    private DefaultInferenceConfig config;
    
    @BeforeEach
    void setUp() {
        config = new DefaultInferenceConfig();
    }
    
    @Test
    void testDefaultValues() {
        assertEquals(1000, config.getNumSamples());
        assertEquals(500, config.getNumWarmup());
        assertEquals(4, config.getNumChains());
        assertNotNull(config.getAlgorithmParams());
        assertTrue(config.getAlgorithmParams().isEmpty());
        assertTrue(config.isParallel());
        assertEquals(1e-6, config.getConvergenceTolerance(), 1e-10);
        assertFalse(config.isSaveIntermediateResults());
    }
    
    @Test
    void testSettersAndGetters() {
        config.setNumSamples(2000);
        assertEquals(2000, config.getNumSamples());
        
        config.setNumWarmup(1000);
        assertEquals(1000, config.getNumWarmup());
        
        config.setNumChains(1);
        assertEquals(1, config.getNumChains());
        
        config.setSeed(12345L);
        assertEquals(12345L, config.getSeed());
        
        config.setParallel(false);
        assertFalse(config.isParallel());
        
        config.setConvergenceTolerance(1e-8);
        assertEquals(1e-8, config.getConvergenceTolerance(), 1e-12);
        
        config.setSaveIntermediateResults(true);
        assertTrue(config.isSaveIntermediateResults());
    }
    
    @Test
    void testFluentInterface() {
        DefaultInferenceConfig result = config
            .setNumSamples(2000)
            .setNumWarmup(1000)
            .setNumChains(1)
            .setSeed(12345L)
            .setParallel(false)
            .setConvergenceTolerance(1e-8)
            .setSaveIntermediateResults(true);
        
        assertSame(config, result);
        assertEquals(2000, config.getNumSamples());
        assertEquals(1000, config.getNumWarmup());
        assertEquals(1, config.getNumChains());
        assertEquals(12345L, config.getSeed());
        assertFalse(config.isParallel());
        assertEquals(1e-8, config.getConvergenceTolerance(), 1e-12);
        assertTrue(config.isSaveIntermediateResults());
    }
    
    @Test
    void testAlgorithmParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("stepSize", 0.1);
        params.put("maxDepth", 10);
        
        config.setAlgorithmParams(params);
        
        Map<String, Object> retrieved = config.getAlgorithmParams();
        assertEquals(2, retrieved.size());
        assertEquals(0.1, retrieved.get("stepSize"));
        assertEquals(10, retrieved.get("maxDepth"));
        
        // 测试修改返回的map不影响原始配置
        retrieved.put("newParam", "value");
        assertFalse(config.getAlgorithmParams().containsKey("newParam"));
    }
    
    @Test
    void testAddAlgorithmParam() {
        config.addAlgorithmParam("stepSize", 0.1);
        config.addAlgorithmParam("maxDepth", 10);
        
        Map<String, Object> params = config.getAlgorithmParams();
        assertEquals(2, params.size());
        assertEquals(0.1, params.get("stepSize"));
        assertEquals(10, params.get("maxDepth"));
        
        // 测试覆盖现有参数
        config.addAlgorithmParam("stepSize", 0.2);
        assertEquals(0.2, config.getAlgorithmParams().get("stepSize"));
    }
    
    @Test
    void testValidation() {
        // 测试有效配置 - 不应该抛出异常
        assertDoesNotThrow(() -> config.validate());
        
        // 测试无效的样本数
        config.setNumSamples(0);
        assertThrows(IllegalStateException.class, () -> config.validate());
        
        config.setNumSamples(-100);
        assertThrows(IllegalStateException.class, () -> config.validate());
        
        config.setNumSamples(1000); // 重置为有效值
        assertDoesNotThrow(() -> config.validate());
        
        // 测试无效的预热样本数
        config.setNumWarmup(-1);
        assertThrows(IllegalStateException.class, () -> config.validate());
        
        config.setNumWarmup(500); // 重置为有效值
        assertDoesNotThrow(() -> config.validate());
        
        // 测试无效的链数
        config.setNumChains(0);
        assertThrows(IllegalStateException.class, () -> config.validate());
        
        config.setNumChains(-1);
        assertThrows(IllegalStateException.class, () -> config.validate());
        
        config.setNumChains(1); // 重置为有效值
        assertDoesNotThrow(() -> config.validate());
        
        // 测试无效的收敛容差
        config.setConvergenceTolerance(0.0);
        assertThrows(IllegalStateException.class, () -> config.validate());
        
        config.setConvergenceTolerance(-1e-6);
        assertThrows(IllegalStateException.class, () -> config.validate());
        
        config.setConvergenceTolerance(1e-6); // 重置为有效值
        assertDoesNotThrow(() -> config.validate());
    }
    
    @Test
    void testBuilder() {
        DefaultInferenceConfig built = DefaultInferenceConfig.builder()
            .numSamples(2000)
            .numWarmup(1000)
            .numChains(1)
            .seed(12345L)
            .parallel(false)
            .convergenceTolerance(1e-8)
            .saveIntermediateResults(true)
            .algorithmParam("stepSize", 0.1)
            .algorithmParam("maxDepth", 10)
            .build();
        
        assertEquals(2000, built.getNumSamples());
        assertEquals(1000, built.getNumWarmup());
        assertEquals(1, built.getNumChains());
        assertEquals(12345L, built.getSeed());
        assertFalse(built.isParallel());
        assertEquals(1e-8, built.getConvergenceTolerance(), 1e-12);
        assertTrue(built.isSaveIntermediateResults());
        
        Map<String, Object> params = built.getAlgorithmParams();
        assertEquals(2, params.size());
        assertEquals(0.1, params.get("stepSize"));
        assertEquals(10, params.get("maxDepth"));
    }
    
    @Test
    void testBuilderValidation() {
        // 测试无效配置构建
        assertThrows(IllegalStateException.class, () -> {
            DefaultInferenceConfig.builder().numSamples(-1).build();
        });
        
        assertThrows(IllegalStateException.class, () -> {
            DefaultInferenceConfig.builder().numSamples(1000).numWarmup(-1).build();
        });
        
        assertThrows(IllegalStateException.class, () -> {
            DefaultInferenceConfig.builder().numSamples(1000).numWarmup(500).numChains(0).build();
        });
        
        assertThrows(IllegalStateException.class, () -> {
            DefaultInferenceConfig.builder().numSamples(1000).numWarmup(500).numChains(1)
                   .convergenceTolerance(0.0).build();
        });
    }
    
    @Test
    void testQuickConfigs() {
        // 测试快速原型配置
        DefaultInferenceConfig prototype = DefaultInferenceConfig.QuickConfigs.quickPrototype();
        assertEquals(100, prototype.getNumSamples());
        assertEquals(50, prototype.getNumWarmup());
        assertEquals(1, prototype.getNumChains());
        assertFalse(prototype.isParallel());
        
        // 测试标准配置
        DefaultInferenceConfig standard = DefaultInferenceConfig.QuickConfigs.standard();
        assertEquals(1000, standard.getNumSamples());
        assertEquals(500, standard.getNumWarmup());
        assertEquals(4, standard.getNumChains());
        assertTrue(standard.isParallel());
        
        // 测试高精度配置
        DefaultInferenceConfig highPrecision = DefaultInferenceConfig.QuickConfigs.highPrecision();
        assertEquals(10000, highPrecision.getNumSamples());
        assertEquals(5000, highPrecision.getNumWarmup());
        assertEquals(8, highPrecision.getNumChains());
        assertTrue(highPrecision.isParallel());
        assertEquals(1e-8, highPrecision.getConvergenceTolerance(), 1e-12);
        
        // 测试快速配置
        DefaultInferenceConfig fast = DefaultInferenceConfig.QuickConfigs.fast();
        assertEquals(500, fast.getNumSamples());
        assertEquals(100, fast.getNumWarmup());
        assertEquals(2, fast.getNumChains());
        assertTrue(fast.isParallel());
        
        // 测试调试配置
        DefaultInferenceConfig debug = DefaultInferenceConfig.QuickConfigs.debug();
        assertEquals(50, debug.getNumSamples());
        assertEquals(25, debug.getNumWarmup());
        assertEquals(1, debug.getNumChains());
        assertFalse(debug.isParallel());
        assertTrue(debug.isSaveIntermediateResults());
    }
    
    @Test
    void testCopy() {
        // 设置原始配置
        config.setNumSamples(2000)
              .setNumWarmup(1000)
              .setNumChains(1)
              .setSeed(12345L)
              .setParallel(false)
              .setConvergenceTolerance(1e-8)
              .setSaveIntermediateResults(true)
              .addAlgorithmParam("stepSize", 0.1)
              .addAlgorithmParam("maxDepth", 10);
        
        // 创建副本
        DefaultInferenceConfig copy = new DefaultInferenceConfig(config);
        
        // 验证副本
        assertEquals(config.getNumSamples(), copy.getNumSamples());
        assertEquals(config.getNumWarmup(), copy.getNumWarmup());
        assertEquals(config.getNumChains(), copy.getNumChains());
        assertEquals(config.getSeed(), copy.getSeed());
        assertEquals(config.isParallel(), copy.isParallel());
        assertEquals(config.getConvergenceTolerance(), copy.getConvergenceTolerance(), 1e-12);
        assertEquals(config.isSaveIntermediateResults(), copy.isSaveIntermediateResults());
        
        Map<String, Object> originalParams = config.getAlgorithmParams();
        Map<String, Object> copyParams = copy.getAlgorithmParams();
        assertEquals(originalParams.size(), copyParams.size());
        assertEquals(originalParams.get("stepSize"), copyParams.get("stepSize"));
        assertEquals(originalParams.get("maxDepth"), copyParams.get("maxDepth"));
        
        // 验证独立性
        copy.setNumSamples(3000);
        assertNotEquals(config.getNumSamples(), copy.getNumSamples());
        
        copy.addAlgorithmParam("newParam", "value");
        assertFalse(config.getAlgorithmParams().containsKey("newParam"));
    }
    
    @Test
    void testToString() {
        config.setNumSamples(2000)
              .setNumWarmup(1000)
              .setNumChains(1)
              .setSeed(12345L)
              .setParallel(false);
        
        String str = config.toString();
        
        assertTrue(str.contains("numSamples=2000"));
        assertTrue(str.contains("numWarmup=1000"));
        assertTrue(str.contains("numChains=1"));
        assertTrue(str.contains("seed=12345"));
        assertTrue(str.contains("parallel=false"));
    }
    
    @Test
    void testEdgeCases() {
        // 测试null算法参数
        config.setAlgorithmParams(null);
        assertNotNull(config.getAlgorithmParams());
        assertTrue(config.getAlgorithmParams().isEmpty());
        
        // 测试空算法参数
        config.setAlgorithmParams(new HashMap<>());
        assertTrue(config.getAlgorithmParams().isEmpty());
        
        // 测试极大值
        config.setNumSamples(Integer.MAX_VALUE);
        assertEquals(Integer.MAX_VALUE, config.getNumSamples());
        assertDoesNotThrow(() -> config.validate());
        
        // 测试极小收敛容差
        config.setConvergenceTolerance(Double.MIN_VALUE);
        assertEquals(Double.MIN_VALUE, config.getConvergenceTolerance(), 0.0);
        assertDoesNotThrow(() -> config.validate());
    }
    
    @Test
    void testEqualsAndHashCode() {
        DefaultInferenceConfig config1 = new DefaultInferenceConfig()
            .setNumSamples(2000)
            .setNumWarmup(1000)
            .setNumChains(1)
            .setSeed(12345L)
            .setParallel(false)
            .addAlgorithmParam("stepSize", 0.1);
        
        DefaultInferenceConfig config2 = new DefaultInferenceConfig()
            .setNumSamples(2000)
            .setNumWarmup(1000)
            .setNumChains(1)
            .setSeed(12345L)
            .setParallel(false)
            .addAlgorithmParam("stepSize", 0.1);
        
        DefaultInferenceConfig config3 = new DefaultInferenceConfig()
            .setNumSamples(1000); // 不同的值
        
        // 测试equals
        assertEquals(config1, config2);
        assertNotEquals(config1, config3);
        assertNotEquals(config1, null);
        assertNotEquals(config1, "not a config");
        
        // 测试hashCode
        assertEquals(config1.hashCode(), config2.hashCode());
        assertNotEquals(config1.hashCode(), config3.hashCode());
    }
}