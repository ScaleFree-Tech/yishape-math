package com.yishape.lab.math.autodiff;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleVector;

/**
 * Global registry for user-defined custom backward functions.
 * 用户自定义反向函数的全局注册表。
 *
 * @deprecated Use {@link CustomOp} instead. CustomOp embeds backward functions directly in
 *             graph nodes, eliminating the need for a global registry and preventing memory leaks
 *             from forgotten {@link #unregister} calls.
 *
 * <p>Used by {@link AD#custom(String, java.util.function.Function, IDiffVector...)}.
 * 供 {@link AD#custom(String, java.util.function.Function, IDiffVector...)} 使用。</p>
 */
@Deprecated
public class CustomGradientRegistry {

    private static final Map<String, Function<IDoubleVector, IDoubleVector[]>> registry = new ConcurrentHashMap<>();

    private CustomGradientRegistry() {}

    /**
     * Registers a backward function: given upstream gradient {@code gradOut}, returns per-input gradients.
     * 注册反向函数：给定上游梯度 {@code gradOut}，返回各输入的梯度数组。
     */
    public static void register(String name, Function<IDoubleVector, IDoubleVector[]> backwardFn) {
        registry.put(name, backwardFn);
    }

    public static Function<IDoubleVector, IDoubleVector[]> get(String name) {
        return registry.get(name);
    }

    public static boolean contains(String name) {
        return registry.containsKey(name);
    }

    public static void unregister(String name) {
        registry.remove(name);
    }

    /**
     * Clears all registered backward functions. Useful for test cleanup to prevent memory leaks.
     * 清除所有已注册的反向函数。用于测试清理以防止内存泄漏。
     */
    public static void clear() {
        registry.clear();
    }
}
