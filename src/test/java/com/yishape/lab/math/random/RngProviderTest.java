package com.yishape.lab.math.random;

import com.yishape.lab.math.random.impl.Pcg64Rng;
import com.yishape.lab.math.random.impl.RandomAdapter;
import com.yishape.lab.math.random.impl.Xoroshiro128PlusPlusRng;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Random;

public class RngProviderTest {

    // ==================== Xoroshiro128++ ====================

    @Test
    public void xoroshiro_seededProducesDeterministicSequence() {
        Xoroshiro128PlusPlusRng r1 = new Xoroshiro128PlusPlusRng(42L);
        Xoroshiro128PlusPlusRng r2 = new Xoroshiro128PlusPlusRng(42L);
        for (int i = 0; i < 100; i++) {
            assertEquals(r1.nextLong(), r2.nextLong());
        }
    }

    @Test
    public void xoroshiro_differentSeedsProduceDifferentSequences() {
        Xoroshiro128PlusPlusRng r1 = new Xoroshiro128PlusPlusRng(42L);
        Xoroshiro128PlusPlusRng r2 = new Xoroshiro128PlusPlusRng(99L);
        boolean same = true;
        for (int i = 0; i < 10; i++) {
            if (r1.nextLong() != r2.nextLong()) { same = false; break; }
        }
        assertFalse(same);
    }

    @Test
    public void xoroshiro_nextDouble_inRange() {
        Xoroshiro128PlusPlusRng rng = new Xoroshiro128PlusPlusRng(42L);
        for (int i = 0; i < 1000; i++) {
            double v = rng.nextDouble();
            assertTrue(v >= 0.0 && v < 1.0, "nextDouble out of [0,1): " + v);
        }
    }

    @Test
    public void xoroshiro_nextInt_inRange() {
        Xoroshiro128PlusPlusRng rng = new Xoroshiro128PlusPlusRng(42L);
        for (int i = 0; i < 1000; i++) {
            int v = rng.nextInt(10);
            assertTrue(v >= 0 && v < 10, "nextInt(10) out of range: " + v);
        }
    }

    @Test
    public void xoroshiro_split_producesIndependentRng() {
        Xoroshiro128PlusPlusRng rng = new Xoroshiro128PlusPlusRng(42L);
        RngProvider r2 = rng.split();
        assertNotEquals(rng.nextLong(), r2.nextLong());
    }

    @Test
    public void xoroshiro_copy_preservesState() {
        Xoroshiro128PlusPlusRng rng = new Xoroshiro128PlusPlusRng(42L);
        for (int i = 0; i < 5; i++) rng.nextLong();
        Xoroshiro128PlusPlusRng copy = rng.copy();
        for (int i = 0; i < 100; i++) {
            assertEquals(rng.nextLong(), copy.nextLong());
        }
    }

    @Test
    public void xoroshiro_serializeRoundTrip() {
        Xoroshiro128PlusPlusRng rng = new Xoroshiro128PlusPlusRng(42L);
        for (int i = 0; i < 10; i++) rng.nextLong();
        long[] state = rng.getState();
        Xoroshiro128PlusPlusRng restored = new Xoroshiro128PlusPlusRng(0L);
        restored.setState(state);
        for (int i = 0; i < 100; i++) {
            assertEquals(rng.nextLong(), restored.nextLong());
        }
    }

    // ==================== PCG64 ====================

    @Test
    public void pcg64_seededProducesDeterministicSequence() {
        Pcg64Rng r1 = new Pcg64Rng(42L);
        Pcg64Rng r2 = new Pcg64Rng(42L);
        for (int i = 0; i < 100; i++) {
            assertEquals(r1.nextLong(), r2.nextLong());
        }
    }

    @Test
    public void pcg64_nextDouble_inRange() {
        Pcg64Rng rng = new Pcg64Rng(42L);
        for (int i = 0; i < 1000; i++) {
            double v = rng.nextDouble();
            assertTrue(v >= 0.0 && v < 1.0, "nextDouble out of [0,1): " + v);
        }
    }

    @Test
    public void pcg64_copy_preservesState() {
        Pcg64Rng rng = new Pcg64Rng(42L);
        for (int i = 0; i < 5; i++) rng.nextLong();
        Pcg64Rng copy = rng.copy();
        for (int i = 0; i < 100; i++) {
            assertEquals(rng.nextLong(), copy.nextLong());
        }
    }

    // ==================== RngFactory ====================

    @Test
    public void factory_createSeeded_isDeterministic() {
        RngProvider r1 = RngFactory.createSeeded(42L);
        RngProvider r2 = RngFactory.createSeeded(42L);
        for (int i = 0; i < 100; i++) {
            assertEquals(r1.nextLong(), r2.nextLong());
        }
    }

    @Test
    public void factory_createXoroshiro_producesCorrectType() {
        RngProvider rng = RngFactory.create(RngFactory.RngType.XOROSHIRO128PP, 42L);
        assertTrue(rng instanceof Xoroshiro128PlusPlusRng);
    }

    @Test
    public void factory_createPcg64_producesCorrectType() {
        RngProvider rng = RngFactory.create(RngFactory.RngType.PCG64, 42L);
        assertTrue(rng instanceof Pcg64Rng);
    }

    @Test
    public void factory_createDefault_isXoroshiro() {
        RngProvider rng = RngFactory.createDefault();
        assertTrue(rng instanceof Xoroshiro128PlusPlusRng);
    }

    // ==================== ThreadLocalRng ====================

    @Test
    public void threadLocalRng_current_returnsValidRng() {
        RngProvider rng = ThreadLocalRng.current();
        assertNotNull(rng);
        double v = rng.nextDouble();
        assertTrue(v >= 0.0 && v < 1.0);
    }

    @Test
    public void threadLocalRng_set_customRng() {
        RngProvider custom = RngFactory.createSeeded(12345L);
        ThreadLocalRng.set(custom);
        assertEquals(ThreadLocalRng.current(), custom);
        // reset for other tests
        ThreadLocalRng.reset();
    }

    // ==================== RandomAdapter ====================

    @Test
    public void adapter_matchesUnderylingRandom() {
        Random jr = new Random(42L);
        RngProvider adapter = new RandomAdapter(jr);

        Random jr2 = new Random(42L);
        for (int i = 0; i < 100; i++) {
            assertEquals(jr2.nextLong(), adapter.nextLong());
        }
    }

    @Test
    public void adapter_copy_throwsUnsupported() {
        RandomAdapter adapter = new RandomAdapter(42L);
        assertThrows(UnsupportedOperationException.class, adapter::copy);
    }

    // ==================== nextInt uniformity (basic) ====================

    @Test
    public void nextInt_powerOfTwo_unbiasedSampling() {
        RngProvider rng = RngFactory.createSeeded(42L);
        int[] counts = new int[4];
        int n = 10000;
        for (int i = 0; i < n; i++) {
            counts[rng.nextInt(4)]++;
        }
        double expected = n / 4.0;
        for (int c : counts) {
            assertEquals(expected, c, expected * 0.1, "non-uniform distribution");
        }
    }

    @Test
    public void nextInt_nonPowerOfTwo_unbiasedSampling() {
        RngProvider rng = RngFactory.createSeeded(42L);
        int[] counts = new int[3];
        int n = 10000;
        for (int i = 0; i < n; i++) {
            counts[rng.nextInt(3)]++;
        }
        double expected = n / 3.0;
        for (int c : counts) {
            assertEquals(expected, c, expected * 0.1, "non-uniform distribution");
        }
    }

    // ==================== nextGaussian ====================

    @Test
    public void xoroshiro_nextGaussian_meanNearZero() {
        Xoroshiro128PlusPlusRng rng = new Xoroshiro128PlusPlusRng(42L);
        double sum = 0;
        int n = 10000;
        for (int i = 0; i < n; i++) {
            sum += rng.nextGaussian();
        }
        double mean = sum / n;
        assertEquals(0, mean, 0.05, "mean should be near 0");
    }
}
