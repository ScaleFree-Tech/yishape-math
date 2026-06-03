package com.yishape.lab.math.random;

import com.yishape.lab.math.random.impl.Pcg64Rng;
import com.yishape.lab.math.random.impl.Xoroshiro128PlusPlusRng;

/**
 * RNG 工厂类。
 *
 * <p>默认使用 Xoroshiro128++，速度快，统计质量好。PCG64 适合需要更强统计保证的场景。
 *
 * @author lteb2
 * @since 0.6.0
 */
public final class RngFactory {

    public enum RngType {
        XOROSHIRO128PP,
        PCG64
    }

    /** 使用当前时间作为种子创建默认生成器 */
    public static RngProvider createDefault() {
        return new Xoroshiro128PlusPlusRng(System.nanoTime() ^ System.currentTimeMillis());
    }

    /** 使用指定种子创建默认类型的生成器 */
    public static RngProvider createSeeded(long seed) {
        return new Xoroshiro128PlusPlusRng(seed);
    }

    /** 使用指定类型和种子创建生成器 */
    public static RngProvider create(RngType type, long seed) {
        switch (type) {
            case PCG64:
                return new Pcg64Rng(seed);
            case XOROSHIRO128PP:
            default:
                return new Xoroshiro128PlusPlusRng(seed);
        }
    }

    /** 从系统熵源生成种子（使用 nanoTime + 线程 ID + 静态计数器） */
    public static long generateSeed() {
        return System.nanoTime()
            ^ Thread.currentThread().threadId()
            ^ System.currentTimeMillis()
            ^ counter.getAndIncrement();
    }

    private static final java.util.concurrent.atomic.AtomicLong counter = new java.util.concurrent.atomic.AtomicLong(0x9e3779b97f4a7c15L);

    private RngFactory() {}
}
