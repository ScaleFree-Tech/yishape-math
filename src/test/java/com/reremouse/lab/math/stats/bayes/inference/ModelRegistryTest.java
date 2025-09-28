package com.reremouse.lab.math.stats.bayes.inference;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 全面测试模型注册表的功能
 */
public class ModelRegistryTest {
    
    private ModelRegistry registry;
    
    @BeforeEach
    void setUp() {
        registry = ModelRegistry.getInstance();
    }
    
    @Test
    void testRegistryBasics() {
        // 测试获取注册表实例
        assertNotNull(registry);
        
        // 测试获取注册的模型和引擎
        assertNotNull(registry.getRegisteredModels());
        assertNotNull(registry.getRegisteredInferenceEngines());
    }
}