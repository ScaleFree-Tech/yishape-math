package com.yishape.lab.math.random.impl;

import com.yishape.lab.math.random.RngProvider;

/**
 * PCG XSL RR 128/64 随机数生成器 (O'Neill 2014).
 *
 * <p>128 位 LCG 状态 + 64 位输出。核心操作：LCG multiplier + XOR + random rotation。
 * 适合需要更强统计保证的场景。
 *
 * @author lteb2
 * @since 0.6.0
 */
public class Pcg64Rng implements RngProvider {

    private long state;
    private long inc; // must be odd
    private double nextGaussian;
    private boolean hasGaussian;

    private static final long MULTIPLIER = 0x5851f42d4c957f2dL;
    private static final long DEFAULT_INCREMENT = 0x14057b7ef767814fL;

    public Pcg64Rng(long seed) {
        // 使用 SplitMix64 初始化
        long z = seed + 0x9e3779b97f4a7c15L;
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        z = z ^ (z >>> 31);
        this.state = z;
        z = z + 0x9e3779b97f4a7c15L;
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        z = z ^ (z >>> 31);
        this.inc = z | 1L; // ensure odd
        this.hasGaussian = false;
    }

    public Pcg64Rng(long state, long inc) {
        this.state = state;
        this.inc = inc | 1L;
        this.hasGaussian = false;
    }

    @Override
    public long nextLong() {
        long oldState = state;
        state = oldState * MULTIPLIER + inc;
        long xorshifted = ((oldState >>> 18) ^ oldState) >>> 27;
        int rot = (int) (oldState >>> 59);
        return Long.rotateRight(xorshifted, rot);
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
    public Pcg64Rng split() {
        long newState = nextLong();
        long newInc = nextLong() | 1L;
        return new Pcg64Rng(newState, newInc);
    }

    @Override
    public Pcg64Rng copy() {
        Pcg64Rng copy = new Pcg64Rng(state, inc);
        copy.nextGaussian = this.nextGaussian;
        copy.hasGaussian = this.hasGaussian;
        return copy;
    }

    @Override
    public long[] getState() {
        return new long[]{state, inc, hasGaussian ? 1L : 0L, Double.doubleToLongBits(nextGaussian)};
    }

    @Override
    public void setState(long[] state) {
        if (state == null || state.length < 2) {
            throw new IllegalArgumentException("state must have at least 2 elements");
        }
        this.state = state[0];
        this.inc = state[1] | 1L;
        this.hasGaussian = state.length > 2 && state[2] != 0;
        if (state.length > 3 && hasGaussian) {
            this.nextGaussian = Double.longBitsToDouble(state[3]);
        }
    }
}
