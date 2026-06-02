package com.yishape.lab.math.signal.factory;

import com.yishape.lab.math.signal.factory.SignalProcessorFactory.ProcessorCategory;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link SignalProcessorFactory}.
 */
class SignalProcessorFactoryTest {

    @Test
    void getInstance_returnsSingleton() {
        SignalProcessorFactory f1 = SignalProcessorFactory.getInstance();
        SignalProcessorFactory f2 = SignalProcessorFactory.getInstance();
        assertSame(f1, f2);
    }

    @Test
    void getVersion() {
        assertEquals("1.0.0", SignalProcessorFactory.getInstance().getVersion());
    }

    @Test
    void getDescription() {
        assertNotNull(SignalProcessorFactory.getInstance().getDescription());
    }

    // ==================== Default Registrations ====================

    @Test
    void defaultProcessors_registered() {
        Set<String> names = SignalProcessorFactory.getInstance().getRegisteredProcessorNames();
        assertTrue(names.contains("butterworth"));
        assertTrue(names.contains("gaussian"));
        assertTrue(names.contains("wavelet"));
        assertTrue(names.contains("spectrum"));
        assertTrue(names.contains("sine"));
        assertTrue(names.size() > 10);
    }

    @Test
    void isProcessorRegistered() {
        assertTrue(SignalProcessorFactory.getInstance().isProcessorRegistered("butterworth"));
        assertFalse(SignalProcessorFactory.getInstance().isProcessorRegistered("nonexistent"));
    }

    // ==================== Creation ====================

    @Test
    void createFilter() throws Exception {
        var filter = SignalProcessorFactory.getInstance().createFilter("butterworth");
        assertNotNull(filter);
    }

    @Test
    void createGenerator() throws Exception {
        var gen = SignalProcessorFactory.getInstance().createGenerator("sine");
        assertNotNull(gen);
    }

    @Test
    void createAnalyzer() throws Exception {
        var analyzer = SignalProcessorFactory.getInstance().createAnalyzer("spectrum");
        assertNotNull(analyzer);
    }

    @Test
    void createTransform() throws Exception {
        var transform = SignalProcessorFactory.getInstance().createTransform("wavelet");
        assertNotNull(transform);
    }

    @Test
    void createProcessor_unknown_throws() {
        assertThrows(Exception.class,
            () -> SignalProcessorFactory.getInstance().createProcessor("nonexistent_processor_xyz"));
    }

    // ==================== Categories ====================

    @Test
    void getProcessorsByCategory_filter() {
        Map<String, ?> filters = SignalProcessorFactory.getInstance()
            .getProcessorsByCategory(ProcessorCategory.FILTER);
        assertNotNull(filters);
        assertTrue(filters.containsKey("butterworth"));
    }

    @Test
    void getProcessorsByCategory_transform() {
        Map<String, ?> transforms = SignalProcessorFactory.getInstance()
            .getProcessorsByCategory(ProcessorCategory.TRANSFORM);
        assertNotNull(transforms);
    }

    // ==================== Register & Unregister ====================

    @Test
    void register_andUnregister() {
        SignalProcessorFactory factory = SignalProcessorFactory.getInstance();
        String testName = "test_custom_processor_xyz";
        try {
            factory.registerProcessor(testName,
                com.yishape.lab.math.signal.generation.SignalGenerator.class,
                ProcessorCategory.GENERATOR, "test", "1.0");
            assertTrue(factory.isProcessorRegistered(testName));
            factory.unregisterProcessor(testName);
            assertFalse(factory.isProcessorRegistered(testName));
        } catch (Exception e) {
            // Cleanup in case of error
            factory.unregisterProcessor(testName);
        }
    }

    @Test
    void getProcessorInfo() {
        var info = SignalProcessorFactory.getInstance().getProcessorInfo("butterworth");
        assertNotNull(info);
    }

    // ==================== ProcessorCategory ====================

    @Test
    void processorCategory_names() {
        assertNotNull(ProcessorCategory.FILTER.getChineseName());
        assertNotNull(ProcessorCategory.FILTER.getEnglishName());
        assertNotNull(ProcessorCategory.TRANSFORM.getChineseName());
        assertNotNull(ProcessorCategory.GENERATOR.getEnglishName());
    }

    @Test
    void allCategories() {
        assertEquals(8, ProcessorCategory.values().length);
    }
}
