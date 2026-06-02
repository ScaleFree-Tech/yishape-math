package com.yishape.lab.math.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for {@link Complex} number class.
 */
class ComplexTest {

    private static final double EPS = 1e-9;

    // ==================== Constants ====================

    @Test
    void zero_isZero() {
        assertEquals(0, Complex.ZERO.real);
        assertEquals(0, Complex.ZERO.imag);
        assertTrue(Complex.ZERO.isZero());
    }

    @Test
    void one_isOne() {
        assertEquals(1, Complex.ONE.real);
        assertEquals(0, Complex.ONE.imag);
    }

    @Test
    void i_isUnitImaginary() {
        assertEquals(0, Complex.I.real);
        assertEquals(1, Complex.I.imag);
    }

    // ==================== Constructors & Factories ====================

    @Test
    void constructor_setsFields() {
        Complex z = new Complex(3, 4);
        assertEquals(3, z.real);
        assertEquals(4, z.imag);
    }

    @Test
    void of_createsSameAsConstructor() {
        Complex z = Complex.of(2, 5);
        assertEquals(2, z.real);
        assertEquals(5, z.imag);
    }

    @Test
    void fromPolar_createsCorrectComplex() {
        // magnitude=1, phase=pi/2 => (0, 1)
        Complex z = Complex.fromPolar(1, Math.PI / 2);
        assertEquals(0, z.real, EPS);
        assertEquals(1, z.imag, EPS);
    }

    @Test
    void fromPolar_degrees() {
        Complex z = Complex.fromPolarDegrees(1, 90);
        assertEquals(0, z.real, EPS);
        assertEquals(1, z.imag, EPS);
    }

    @Test
    void unit_createsUnitCircle() {
        Complex z = Complex.unit(Math.PI);
        assertEquals(-1, z.real, EPS);
        assertEquals(0, z.imag, EPS);
    }

    @Test
    void unitDegrees() {
        Complex z = Complex.unitDegrees(0);
        assertEquals(1, z.real, EPS);
        assertEquals(0, z.imag, EPS);
    }

    @Test
    void arrayOf_createsArray() {
        double[] reals = {1, 2, 3};
        double[] imags = {4, 5, 6};
        Complex[] arr = Complex.arrayOf(reals, imags);
        assertEquals(3, arr.length);
        assertEquals(1, arr[0].real);
        assertEquals(4, arr[0].imag);
        assertEquals(3, arr[2].real);
        assertEquals(6, arr[2].imag);
    }

    @Test
    void arrayOf_differentLengths_throws() {
        assertThrows(IllegalArgumentException.class,
            () -> Complex.arrayOf(new double[]{1}, new double[]{1, 2}));
    }

    // ==================== Arithmetic ====================

    @Test
    void add_twoComplexNumbers() {
        Complex a = new Complex(1, 2);
        Complex b = new Complex(3, 4);
        Complex result = a.add(b);
        assertEquals(4, result.real);
        assertEquals(6, result.imag);
    }

    @Test
    void subtract_twoComplexNumbers() {
        Complex a = new Complex(5, 7);
        Complex b = new Complex(3, 4);
        Complex result = a.subtract(b);
        assertEquals(2, result.real);
        assertEquals(3, result.imag);
    }

    @Test
    void multiply_twoComplexNumbers() {
        // (1+2i)*(3+4i) = 3+4i+6i+8i^2 = -5+10i
        Complex a = new Complex(1, 2);
        Complex b = new Complex(3, 4);
        Complex result = a.multiply(b);
        assertEquals(-5, result.real, EPS);
        assertEquals(10, result.imag, EPS);
    }

    @Test
    void scale_byScalar() {
        Complex z = new Complex(3, 4);
        Complex result = z.scale(2);
        assertEquals(6, result.real);
        assertEquals(8, result.imag);
    }

    @Test
    void divide_twoComplexNumbers() {
        // (1+2i)/(3+4i) = (1+2i)(3-4i)/25 = (3-4i+6i-8i^2)/25 = (11+2i)/25
        Complex a = new Complex(1, 2);
        Complex b = new Complex(3, 4);
        Complex result = a.divide(b);
        assertEquals(11.0 / 25, result.real, EPS);
        assertEquals(2.0 / 25, result.imag, EPS);
    }

    @Test
    void divide_byZero_throws() {
        Complex a = new Complex(1, 2);
        Complex b = new Complex(0, 0);
        assertThrows(ArithmeticException.class, () -> a.divide(b));
    }

    @Test
    void reciprocal() {
        // 1/(1+i) = (1-i)/2
        Complex z = new Complex(1, 1);
        Complex r = z.reciprocal();
        assertEquals(0.5, r.real, EPS);
        assertEquals(-0.5, r.imag, EPS);
    }

    @Test
    void reciprocal_ofZero_throws() {
        assertThrows(ArithmeticException.class, () -> Complex.ZERO.reciprocal());
    }

    @Test
    void conjugate() {
        Complex z = new Complex(3, -4);
        Complex c = z.conjugate();
        assertEquals(3, c.real);
        assertEquals(4, c.imag);
    }

    // ==================== Magnitude & Phase ====================

    @Test
    void magnitude_3_4_is5() {
        assertEquals(5, new Complex(3, 4).magnitude(), EPS);
    }

    @Test
    void abs_aliasForMagnitude() {
        assertEquals(new Complex(3, 4).abs(), new Complex(3, 4).magnitude(), EPS);
    }

    @Test
    void phase_of1i_isPiOver2() {
        assertEquals(Math.PI / 2, Complex.I.phase(), EPS);
    }

    @Test
    void phaseDegrees_of1i_is90() {
        assertEquals(90, Complex.I.phaseDegrees(), EPS);
    }

    // ==================== Power & Root ====================

    @Test
    void square() {
        // (3+4i)^2 = 9+24i-16 = -7+24i
        Complex z = new Complex(3, 4);
        Complex s = z.square();
        assertEquals(-7, s.real, EPS);
        assertEquals(24, s.imag, EPS);
    }

    @Test
    void power_zero() {
        Complex z = new Complex(3, 4);
        Complex p = z.power(0);
        assertEquals(1, p.real, EPS);
        assertEquals(0, p.imag, EPS);
    }

    @Test
    void power_positive() {
        // (1+i)^2 = 1+2i-1 = 2i
        Complex z = new Complex(1, 1);
        Complex p = z.power(2);
        assertEquals(0, p.real, EPS);
        assertEquals(2, p.imag, EPS);
    }

    @Test
    void power_negative() {
        // (1+i)^(-1) = 1/(1+i) = (1-i)/2
        Complex z = new Complex(1, 1);
        Complex p = z.power(-1);
        assertEquals(0.5, p.real, EPS);
        assertEquals(-0.5, p.imag, EPS);
    }

    @Test
    void sqrt_of1_is1() {
        Complex s = Complex.ONE.sqrt();
        assertEquals(1, s.real, EPS);
        assertEquals(0, s.imag, EPS);
    }

    @Test
    void sqrt_ofNegativeReal_givesPositiveImaginary() {
        // sqrt(-1) = i
        Complex z = new Complex(-1, 0);
        Complex s = z.sqrt();
        assertEquals(0, s.real, EPS);
        assertEquals(1, s.imag, EPS);
    }

    @Test
    void sqrt_roundtrip() {
        Complex z = new Complex(3, 4);
        Complex s = z.sqrt();
        Complex sq = s.multiply(s);
        assertEquals(z.real, sq.real, EPS);
        assertEquals(z.imag, sq.imag, EPS);
    }

    // ==================== Transcendental Functions ====================

    @Test
    void exp_of0_is1() {
        Complex e = Complex.ZERO.exp();
        assertEquals(1, e.real, EPS);
        assertEquals(0, e.imag, EPS);
    }

    @Test
    void exp_ofIpi_isMinus1() {
        // e^(i*pi) = -1
        Complex z = new Complex(0, Math.PI);
        Complex e = z.exp();
        assertEquals(-1, e.real, EPS);
        assertEquals(0, e.imag, EPS);
    }

    @Test
    void log_of1_is0() {
        Complex l = Complex.ONE.log();
        assertEquals(0, l.real, EPS);
        assertEquals(0, l.imag, EPS);
    }

    @Test
    void log_roundtrip() {
        Complex z = new Complex(1, 1);
        Complex l = z.exp().log();
        assertEquals(z.real, l.real, 1e-6);
        assertEquals(z.imag, l.imag, 1e-6);
    }

    @Test
    void sin_of0_is0() {
        assertEquals(0, Complex.ZERO.sin().real, EPS);
    }

    @Test
    void cos_of0_is1() {
        assertEquals(1, Complex.ZERO.cos().real, EPS);
    }

    @Test
    void sinI_isIsinh1() {
        // sin(i) = i*sinh(1)
        Complex s = Complex.I.sin();
        assertEquals(0, s.real, EPS);
        assertEquals(Math.sinh(1), s.imag, EPS);
    }

    // ==================== Predicates ====================

    @Test
    void isZero_trueForZero() {
        assertTrue(Complex.ZERO.isZero());
        assertTrue(new Complex(1e-15, 1e-15).isZero());
    }

    @Test
    void isZero_falseForNonZero() {
        assertFalse(new Complex(1, 0).isZero());
    }

    @Test
    void isReal_trueForRealNumbers() {
        assertTrue(new Complex(5, 0).isReal());
        assertFalse(new Complex(5, 1).isReal());
    }

    @Test
    void isPurelyImaginary() {
        assertTrue(new Complex(0, 5).isPurelyImaginary());
        assertFalse(new Complex(1, 5).isPurelyImaginary());
    }

    // ==================== Equality & Hash ====================

    @Test
    void equals_sameValues() {
        Complex a = new Complex(3, 4);
        Complex b = new Complex(3, 4);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
    }

    @Test
    void equals_differentValues() {
        assertNotEquals(new Complex(3, 4), new Complex(3, 5));
    }

    @Test
    void equals_withinEpsilon() {
        Complex a = new Complex(1, 2);
        Complex b = new Complex(1 + 1e-14, 2);
        assertEquals(a, b);
    }

    // ==================== String ====================

    @Test
    void toString_format() {
        String s = new Complex(1, 2).toString();
        assertTrue(s.contains("1.000"));
        assertTrue(s.contains("2.000"));
    }

    @Test
    void toStringFull_format() {
        String s = new Complex(1, 2).toStringFull();
        assertTrue(s.contains("1.000000"));
        assertTrue(s.contains("2.000000"));
    }
}
