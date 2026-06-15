package com.yishape.lab.math.compute;

/**
 * Mixed-precision GEMM dispatch: fp16 inputs → fp32 accumulation.
 *
 * <p>Dispatch chain:
 * <ol>
 *   <li>HPC native {@code flatHgemm} (fp16 GEMM in Rust/faer) — PLANNED</li>
 *   <li>GPU native {@code gpu_gemm_f16} (fp16 WGSL shader) — PLANNED</li>
 *   <li>CPU fp32 GEMM via existing {@link FloatFlatGemm#flatMmul} — available now,
 *       with fp16→fp32 conversion on the fly</li>
 * </ol>
 *
 * <p>When native fp16 kernels are unavailable, this class converts fp16 inputs to fp32,
 * delegates to the existing fp32 GEMM. This provides correctness with fp16 storage
 * (halved weight memory) even without native fp16 compute kernels.
 * Full fp16 arithmetic throughput requires the Rust native kernels.</p>
 */
public final class HalfFloatGemm {

    private HalfFloatGemm() {}

    /** Check whether native fp16 GEMM is available via HPC backend. */
    public static boolean isNativeFp16Available() {
        return com.yishape.lab.math.hpc.YishapeHpc.isFp16Available();
    }

    /**
     * fp16 GEMM: C = A × B where A[n×k] and B[k×m] in fp16, C[n×m] in fp32.
     *
     * <p>Returns {@code true} if native fp16 GEMM succeeded, {@code false} if fallback used.</p>
     *
     * @param a    input matrix A in fp16 (short[])
     * @param b    input matrix B in fp16 (short[])
     * @param cOut output matrix C in fp32 (float[]) — caller-allocated
     * @param n    rows of A and C
     * @param m    columns of B and C
     * @param k    columns of A / rows of B
     * @return true if native fp16 path was used
     */
    public static boolean flatHgemm(int n, int m, int k, short[] a, short[] b, float[] cOut) {
        // Tier 1: HPC native fp16 GEMM
        if (isNativeFp16Available()) {
            int rc = com.yishape.lab.math.hpc.YishapeHpc.flatHgemm(n, m, k, a, b, cOut);
            if (rc == 0) return true;
        }
        // Tier 2: fp16→fp32 conversion → fp32 GEMM (fallback, always available)
        float[] fa = halfToFloat(a);
        float[] fb = halfToFloat(b);
        float[] tmp = FloatFlatGemm.flatMmul(fa, n, k, fb, m);
        System.arraycopy(tmp, 0, cOut, 0, n * m);
        return false;
    }

    /**
     * Convert fp16 (short[] IEEE 754 half-precision) to fp32 (float[]).
     *
     * <p>Uses software conversion (bit manipulation) since Java has no half type.
     * This is correct but ~10× slower than hardware-accelerated conversion.</p>
     */
    public static float[] halfToFloat(short[] half) {
        float[] f = new float[half.length];
        for (int i = 0; i < half.length; i++) {
            f[i] = halfToFloat(half[i]);
        }
        return f;
    }

    /**
     * Convert fp32 (float[]) to fp16 (short[] IEEE 754 half-precision).
     * Uses round-to-nearest-even tie-breaking.
     */
    public static short[] floatToHalf(float[] f) {
        short[] h = new short[f.length];
        for (int i = 0; i < f.length; i++) {
            h[i] = floatToHalf(f[i]);
        }
        return h;
    }

    /**
     * IEEE 754 half-precision (binary16) to float conversion.
     *
     * <p>Layout: 1 sign | 5 exponent (bias 15) | 10 mantissa.
     * Subnormal and special values (NaN, Inf) are handled correctly.</p>
     */
    public static float halfToFloat(short h) {
        int bits = h & 0xFFFF;
        int sign = (bits & 0x8000) << 16;
        int exp  = (bits & 0x7C00) >> 10;
        int mant = (bits & 0x03FF);

        if (exp == 0) {
            // Zero or subnormal
            if (mant == 0) return Float.intBitsToFloat(sign);
            // Subnormal: normalize
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FF;
            exp++;
        } else if (exp == 31) {
            // Infinity or NaN
            return Float.intBitsToFloat(sign | 0x7F800000 | (mant << 13));
        }

        exp = exp + (127 - 15); // rebias
        return Float.intBitsToFloat(sign | (exp << 23) | (mant << 13));
    }

    /**
     * Float to IEEE 754 half-precision (binary16) with round-to-nearest-even.
     */
    public static short floatToHalf(float f) {
        int bits = Float.floatToRawIntBits(f);
        int sign = (bits >>> 16) & 0x8000;
        int exp  = ((bits >>> 23) & 0xFF);
        int mant = (bits & 0x007FFFFF); // 23-bit fractional part

        if (exp == 0) {
            // Zero or subnormal float → zero in half
            return (short) sign;
        } else if (exp == 0xFF) {
            // Infinity or NaN
            if (mant == 0) return (short) (sign | 0x7C00); // Inf
            return (short) (sign | 0x7C00 | (mant >>> 13) | 0x0200); // NaN (ensure quiet)
        }

        int halfExp = exp - 127 + 15; // rebias: float(bias=127) → half(bias=15)

        if (halfExp >= 31) {
            // Overflow → Inf
            return (short) (sign | 0x7C00);
        }

        if (halfExp <= 0) {
            // Subnormal: 2^(exp-127) * 1.mant = 2^(-14) * subMant/2^10
            // subMant = round((1.mant) >> (14 - halfExp))
            if (halfExp < -10) return (short) sign; // too small → zero
            int fullMant = 0x800000 | mant; // 24 bits: implicit 1 + 23-bit fraction
            int shift = 14 - halfExp;
            int subMant = fullMant >>> shift;
            // Round to nearest even: guard at (shift-1), sticky = OR of bits below guard
            int guard = (fullMant >>> (shift - 1)) & 1;
            int sticky = (fullMant & ((1 << (shift - 1)) - 1)) != 0 ? 1 : 0;
            if (guard == 1 && (sticky == 1 || (subMant & 1) == 1)) {
                subMant++;
            }
            return (short) (sign | subMant);
        }

        // Normal half: mantissa = top 10 bits of 23-bit fraction
        int halfMant = mant >>> 13;
        int guard = (mant >>> 12) & 1;    // first bit shifted out
        int sticky = (mant & 0x0FFF) != 0 ? 1 : 0; // remaining 12 bits shifted out

        // Round to nearest even
        if (guard == 1 && (sticky == 1 || (halfMant & 1) == 1)) {
            halfMant++;
            if (halfMant > 0x3FF) {
                // Mantissa overflow → carry to exponent
                halfExp++;
                halfMant = 0;
                if (halfExp >= 31) return (short) (sign | 0x7C00);
            }
        }

        return (short) (sign | (halfExp << 10) | halfMant);
    }
}
