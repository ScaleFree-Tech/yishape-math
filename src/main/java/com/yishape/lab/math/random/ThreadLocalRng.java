package com.yishape.lab.math.random;

/**
 * 线程局部 RngProvider 包装，确保线程安全。
 *
 * <p>每个线程持有独立分叉的生成器，避免锁竞争。
 *
 * <pre>{@code
 * RngProvider rng = ThreadLocalRng.current();
 * double x = rng.nextDouble();
 * }</pre>
 *
 * @author lteb2
 * @since 0.6.0
 */
public final class ThreadLocalRng {

    private static final ThreadLocal<RngProvider> threadLocal = ThreadLocal.withInitial(
        () -> RngFactory.createSeeded(RngFactory.generateSeed())
    );

    /** 获取当前线程的 RngProvider */
    public static RngProvider current() {
        return threadLocal.get();
    }

    /** 用指定生成器替换当前线程的实例 */
    public static void set(RngProvider rng) {
        threadLocal.set(rng);
    }

    /** 重置当前线程的生成器为新实例 */
    public static void reset() {
        threadLocal.remove();
    }

    private ThreadLocalRng() {}
}
