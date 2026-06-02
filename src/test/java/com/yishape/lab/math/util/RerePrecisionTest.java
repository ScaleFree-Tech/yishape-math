package com.yishape.lab.math.util;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for {@link RerePrecision} utility class.
 */
class RerePrecisionTest {

    // ==================== Constants ====================

    @Test
    void machineEpsilon_isPositive() {
        assertTrue(RerePrecision.MACHINE_EPSILON > 0);
        assertTrue(RerePrecision.MACHINE_EPSILON < 1e-14);
    }

    @Test
    void safeMin_isPositive() {
        assertTrue(RerePrecision.SAFE_MIN > 0);
    }

    @Test
    void getDefaultEpsilon() {
        assertEquals(1e-15, RerePrecision.getDefaultEpsilon());
    }

    @Test
    void getDefaultMaxUlps() {
        assertEquals(1, RerePrecision.getDefaultMaxUlps());
    }

    // ==================== Equals ====================

    @Test
    void equals_identicalValues() {
        assertTrue(RerePrecision.equals(1.0, 1.0));
    }

    @Test
    void equals_differentValues() {
        assertFalse(RerePrecision.equals(1.0, 2.0));
    }

    @Test
    void equals_nearbyValues_withinDefaultUlps() {
        // Adjacent double values
        double a = 1.0;
        double b = Math.nextUp(a);
        assertTrue(RerePrecision.equals(a, b));
    }

    @Test
    void equals_farValues_false() {
        assertFalse(RerePrecision.equals(1.0, 1.1));
    }

    @Test
    void equals_withEpsilon_absoluteTolerance() {
        assertTrue(RerePrecision.equals(1.0, 1.0 + 1e-10, 1e-9));
        assertFalse(RerePrecision.equals(1.0, 1.0 + 1e-8, 1e-9));
    }

    @Test
    void equals_withUlps() {
        double a = 1.0;
        double b = Math.nextUp(a);
        assertTrue(RerePrecision.equals(a, b, 1));
        assertTrue(RerePrecision.equals(a, b, 2));
    }

    @Test
    void equals_withEpsAndUlps() {
        assertTrue(RerePrecision.equals(1.0, 1.0, 1e-10, 1));
        assertFalse(RerePrecision.equals(Double.NaN, 1.0, 1e-10, 1));
    }

    @Test
    void equals_nanNotEqualToAnything() {
        assertFalse(RerePrecision.equals(Double.NaN, Double.NaN));
        assertFalse(RerePrecision.equals(Double.NaN, 1.0));
        assertFalse(RerePrecision.equals(1.0, Double.NaN));
    }

    @Test
    void equals_infinityEqualsItself() {
        assertTrue(RerePrecision.equals(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY));
        assertTrue(RerePrecision.equals(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY));
    }

    @Test
    void equalsIncludingNaN_nanEqualsNan() {
        assertTrue(RerePrecision.equalsIncludingNaN(Double.NaN, Double.NaN));
    }

    @Test
    void equalsIncludingNaN_nanNotEqualToNumber() {
        assertFalse(RerePrecision.equalsIncludingNaN(Double.NaN, 1.0));
    }

    @Test
    void equalsIncludingNaN_withEpsilon() {
        assertTrue(RerePrecision.equalsIncludingNaN(1.0, 1.0 + 1e-10, 1e-9));
    }

    @Test
    void equalsIncludingNaN_withUlps() {
        assertTrue(RerePrecision.equalsIncludingNaN(Double.NaN, Double.NaN, 1));
    }

    @Test
    void equalsWithRelativeTolerance_closeValues() {
        assertTrue(RerePrecision.equalsWithRelativeTolerance(1.0, 1.0 + 1e-10, 1e-6));
    }

    @Test
    void equalsWithRelativeTolerance_farValues() {
        assertFalse(RerePrecision.equalsWithRelativeTolerance(1.0, 2.0, 1e-6));
    }

    // ==================== EqualsZero ====================

    @Test
    void equalsZero_trueForZero() {
        assertTrue(RerePrecision.equalsZero(0.0));
    }

    @Test
    void equalsZero_trueForTiny() {
        assertTrue(RerePrecision.equalsZero(1e-16));
    }

    @Test
    void equalsZero_falseForNonZero() {
        assertFalse(RerePrecision.equalsZero(1.0));
    }

    @Test
    void equalsZero_customEpsilon() {
        assertTrue(RerePrecision.equalsZero(0.5, 1.0));
        assertFalse(RerePrecision.equalsZero(2.0, 1.0));
    }

    // ==================== CompareTo ====================

    @Test
    void compareTo_lessThan() {
        assertEquals(-1, RerePrecision.compareTo(1.0, 2.0));
    }

    @Test
    void compareTo_greaterThan() {
        assertEquals(1, RerePrecision.compareTo(2.0, 1.0));
    }

    @Test
    void compareTo_equal() {
        assertEquals(0, RerePrecision.compareTo(1.0, 1.0));
    }

    @Test
    void compareTo_withEpsilon() {
        assertEquals(0, RerePrecision.compareTo(1.0, 1.0 + 1e-10, 1e-9));
    }

    @Test
    void compareTo_withUlps() {
        assertEquals(0, RerePrecision.compareTo(1.0, Math.nextUp(1.0), 2));
    }

    // ==================== Ordering ====================

    @Test
    void isGreaterThan() {
        assertTrue(RerePrecision.isGreaterThan(2.0, 1.0));
        assertFalse(RerePrecision.isGreaterThan(1.0, 1.0));
    }

    @Test
    void isLessThan() {
        assertTrue(RerePrecision.isLessThan(1.0, 2.0));
        assertFalse(RerePrecision.isLessThan(1.0, 1.0));
    }

    @Test
    void isGreaterThanOrEqual() {
        assertTrue(RerePrecision.isGreaterThanOrEqual(2.0, 1.0));
        assertTrue(RerePrecision.isGreaterThanOrEqual(1.0, 1.0));
        assertFalse(RerePrecision.isGreaterThanOrEqual(0.5, 1.0));
    }

    @Test
    void isLessThanOrEqual() {
        assertTrue(RerePrecision.isLessThanOrEqual(1.0, 2.0));
        assertTrue(RerePrecision.isLessThanOrEqual(1.0, 1.0));
        assertFalse(RerePrecision.isLessThanOrEqual(1.5, 1.0));
    }

    @Test
    void isGreaterThan_withEpsilon() {
        assertTrue(RerePrecision.isGreaterThan(1.0 + 1e-8, 1.0, 1e-9));
        assertFalse(RerePrecision.isGreaterThan(1.0 + 1e-10, 1.0, 1e-9));
    }

    @Test
    void isLessThan_withEpsilon() {
        assertTrue(RerePrecision.isLessThan(1.0 - 1e-8, 1.0, 1e-9));
        assertFalse(RerePrecision.isLessThan(1.0 - 1e-10, 1.0, 1e-9));
    }

    // ==================== Rounding ====================

    @Test
    void roundToDecimalPlaces() {
        assertEquals(3.14, RerePrecision.roundToDecimalPlaces(3.14159, 2), 1e-10);
        assertEquals(3.142, RerePrecision.roundToDecimalPlaces(3.14159, 3), 1e-10);
    }

    @Test
    void roundToDecimalPlaces_zero() {
        assertEquals(0, RerePrecision.roundToDecimalPlaces(0.001, 2), 1e-10);
    }

    @Test
    void roundToDecimalPlaces_negative() {
        assertEquals(-3.14, RerePrecision.roundToDecimalPlaces(-3.14159, 2), 1e-10);
    }
}
