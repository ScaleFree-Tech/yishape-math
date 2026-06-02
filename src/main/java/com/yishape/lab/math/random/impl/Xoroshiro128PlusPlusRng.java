package com.yishape.lab.math.random.impl;

import com.yishape.lab.math.random.RngProvider;

/**
 * Xoroshiro128++ 随机数生成器 (Blackman &amp; Vigna 2019).
 *
 * <p>128 位状态，速度快，统计质量好，适合数值计算。核心操作：rotate + xor + shift + add。
 *
 * <p>参考实现: <a href="https://prng.di.unimi.it/">https://prng.di.unimi.it/</a>
 *
 * @author lteb2
 * @since 0.6.0
 */
public class Xoroshiro128PlusPlusRng implements RngProvider {

    private long s0, s1;
    private double nextGaussian;
    private boolean hasGaussian;

    public Xoroshiro128PlusPlusRng(long seed) {
        // 使用 SplitMix64 初始化状态，避免 seed 为 0 导致的全零状态
        long z = seed + 0x9e3779b97f4a7c15L;
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        z = z ^ (z >>> 31);
        this.s0 = z;
        z = z + 0x9e3779b97f4a7c15L;
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        z = z ^ (z >>> 31);
        this.s1 = z;
        this.hasGaussian = false;
    }

    public Xoroshiro128PlusPlusRng(long s0, long s1) {
        if ((s0 | s1) == 0) {
            s0 = 1;
        }
        this.s0 = s0;
        this.s1 = s1;
        this.hasGaussian = false;
    }

    @Override
    public long nextLong() {
        long s0 = this.s0;
        long s1 = this.s1;
        long result = Long.rotateLeft(s0 + s1, 17) + s0;
        s1 ^= s0;
        this.s0 = Long.rotateLeft(s0, 49) ^ s1 ^ (s1 << 21);
        this.s1 = Long.rotateLeft(s1, 28);
        return result;
    }

    @Override
    public double nextDouble() {
        return (nextLong() >>> 11) * 0x1.0p-53;
    }

    @Override
    public double nextGaussian() {
        if (hasGaussian) {
            hasGaussian = false;
            return nextGaussian;
        }
        // Box-Muller transform
        double u1, u2, s;
        do {
            u1 = 2.0 * nextDouble() - 1.0;
            u2 = 2.0 * nextDouble() - 1.0;
            s = u1 * u1 + u2 * u2;
        } while (s >= 1.0 || s == 0.0);
        double factor = Math.sqrt(-2.0 * Math.log(s) / s);
        nextGaussian = u2 * factor;
        hasGaussian = true;
        return u1 * factor;
    }

    @Override
    public Xoroshiro128PlusPlusRng split() {
        long seed1 = nextLong();
        long seed2 = nextLong();
        return new Xoroshiro128PlusPlusRng(seed1, seed2);
    }

    @Override
    public Xoroshiro128PlusPlusRng copy() {
        Xoroshiro128PlusPlusRng copy = new Xoroshiro128PlusPlusRng(s0, s1);
        copy.nextGaussian = this.nextGaussian;
        copy.hasGaussian = this.hasGaussian;
        return copy;
    }

    @Override
    public long[] getState() {
        return new long[]{s0, s1, hasGaussian ? 1L : 0L, Double.doubleToLongBits(nextGaussian)};
    }

    @Override
    public void setState(long[] state) {
        if (state == null || state.length < 2) {
            throw new IllegalArgumentException("state must have at least 2 elements");
        }
        this.s0 = state[0];
        this.s1 = state[1];
        this.hasGaussian = state.length > 2 && state[2] != 0;
        if (state.length > 3 && hasGaussian) {
            this.nextGaussian = Double.longBitsToDouble(state[3]);
        }
    }
}
