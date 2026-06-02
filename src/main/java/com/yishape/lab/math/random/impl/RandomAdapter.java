package com.yishape.lab.math.random.impl;

import com.yishape.lab.math.random.RngProvider;

import java.util.Random;

/**
 * java.util.Random 到 RngProvider 的适配器，用于向后兼容。
 *
 * @deprecated 请使用 {@link Xoroshiro128PlusPlusRng} 或 {@link Pcg64Rng}
 * @author lteb2
 * @since 0.6.0
 */
@Deprecated
public class RandomAdapter implements RngProvider {

    private final Random random;

    public RandomAdapter(Random random) {
        this.random = random;
    }

    public RandomAdapter(long seed) {
        this.random = new Random(seed);
    }

    @Override
    public long nextLong() {
        return random.nextLong();
    }

    @Override
    public double nextDouble() {
        return random.nextDouble();
    }

    @Override
    public double nextGaussian() {
        return random.nextGaussian();
    }

    @Override
    public void nextBytes(byte[] bytes) {
        random.nextBytes(bytes);
    }

    @Override
    public int nextInt(int bound) {
        return random.nextInt(bound);
    }

    @Override
    public RngProvider split() {
        return new RandomAdapter(random.nextLong());
    }

    @Override
    public RngProvider copy() {
        throw new UnsupportedOperationException("java.util.Random 不支持深拷贝");
    }

    @Override
    public long[] getState() {
        throw new UnsupportedOperationException("java.util.Random 状态不可序列化");
    }

    @Override
    public void setState(long[] state) {
        throw new UnsupportedOperationException("java.util.Random 状态不可序列化");
    }

    /** 获取底层的 java.util.Random */
    public Random getRandom() {
        return random;
    }
}
