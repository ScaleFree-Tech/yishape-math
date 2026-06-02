package com.yishape.lab.math.compute.hpc;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link HpcSwitch} 运行时总开关测试。
 */
class HpcSwitchTest {

    @Test
    void testDefaultEnabled() {
        // 默认启用
        assertTrue(HpcSwitch.isEnabled(), "HpcSwitch 默认应启用");
    }

    @Test
    void testDisableAndEnable() {
        boolean original = HpcSwitch.isEnabled();
        try {
            HpcSwitch.disable();
            assertFalse(HpcSwitch.isEnabled(), "disable() 后应返回 false");

            HpcSwitch.enable();
            assertTrue(HpcSwitch.isEnabled(), "enable() 后应返回 true");
        } finally {
            if (original) HpcSwitch.enable();
            else HpcSwitch.disable();
        }
    }

    @Test
    void testToggle() {
        boolean original = HpcSwitch.isEnabled();
        try {
            boolean afterToggle = HpcSwitch.toggle();
            assertEquals(!original, afterToggle, "toggle() 应翻转状态");
            assertEquals(afterToggle, HpcSwitch.isEnabled(), "toggle() 返回值应与 isEnabled() 一致");

            // 再 toggle 一次应恢复
            boolean restored = HpcSwitch.toggle();
            assertEquals(original, restored, "两次 toggle 应恢复原始状态");
        } finally {
            if (original) HpcSwitch.enable();
            else HpcSwitch.disable();
        }
    }

    @Test
    void testRunWithRestoresState() {
        boolean original = HpcSwitch.isEnabled();
        try {
            // 先确保初始状态为 true
            HpcSwitch.enable();
            assertTrue(HpcSwitch.isEnabled());

            // runWith(false) 临时禁用，结束后自动恢复
            HpcSwitch.runWith(false, () -> {
                assertFalse(HpcSwitch.isEnabled(), "runWith 内部应禁用");
            });
            assertTrue(HpcSwitch.isEnabled(), "runWith 结束后应恢复为 true");

            // runWith(true) 临时启用，结束后自动恢复
            HpcSwitch.disable();
            assertFalse(HpcSwitch.isEnabled());

            HpcSwitch.runWith(true, () -> {
                assertTrue(HpcSwitch.isEnabled(), "runWith 内部应启用");
            });
            assertFalse(HpcSwitch.isEnabled(), "runWith 结束后应恢复为 false");
        } finally {
            if (original) HpcSwitch.enable();
            else HpcSwitch.disable();
        }
    }

    @Test
    void testRunWithExceptionStillRestores() {
        boolean original = HpcSwitch.isEnabled();
        try {
            HpcSwitch.enable();

            assertThrows(RuntimeException.class, () ->
                HpcSwitch.runWith(false, () -> {
                    assertFalse(HpcSwitch.isEnabled());
                    throw new RuntimeException("test exception");
                })
            );

            // 即使抛异常，状态也应恢复
            assertTrue(HpcSwitch.isEnabled(), "抛异常后仍应恢复原始状态");
        } finally {
            if (original) HpcSwitch.enable();
            else HpcSwitch.disable();
        }
    }

    @Test
    void testHpcConfigRespectsSwitch() {
        boolean original = HpcSwitch.isEnabled();
        try {
            // 系统属性默认 true，但 HpcSwitch 禁用时 allowAttempts 应返回 false
            HpcSwitch.disable();
            assertFalse(HpcConfig.allowAttempts(), "HpcSwitch 禁用时 allowAttempts 应返回 false");

            HpcSwitch.enable();
            assertTrue(HpcConfig.allowAttempts(), "HpcSwitch 启用时 allowAttempts 应返回 true");
        } finally {
            if (original) HpcSwitch.enable();
            else HpcSwitch.disable();
        }
    }
}
