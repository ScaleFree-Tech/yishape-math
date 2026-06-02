package com.yishape.lab.math.core;

public class Complex {
    public final double real;
    public final double imag;

    private static final double EPSILON = 1e-12;

    public static final Complex ZERO = new Complex(0, 0);
    public static final Complex ONE = new Complex(1, 0);
    public static final Complex I = new Complex(0, 1);

    public Complex(double real, double imag) {
        this.real = real;
        this.imag = imag;
    }

    public static Complex of(double real, double imag) {
        return new Complex(real, imag);
    }

    public static Complex fromPolar(double magnitude, double phase) {
        return new Complex(magnitude * Math.cos(phase), magnitude * Math.sin(phase));
    }

    public static Complex[] arrayOf(double[] reals, double[] imaginaries) {
        if (reals.length != imaginaries.length) {
            throw new IllegalArgumentException("Real and imaginary arrays must have same length");
        }
        Complex[] result = new Complex[reals.length];
        for (int i = 0; i < reals.length; i++) {
            result[i] = new Complex(reals[i], imaginaries[i]);
        }
        return result;
    }

    public Complex add(Complex other) {
        return new Complex(this.real + other.real, this.imag + other.imag);
    }

    public Complex subtract(Complex other) {
        return new Complex(this.real - other.real, this.imag - other.imag);
    }

    public Complex multiply(Complex other) {
        return new Complex(
            this.real * other.real - this.imag * other.imag,
            this.real * other.imag + this.imag * other.real
        );
    }

    public Complex scale(double scalar) {
        return new Complex(this.real * scalar, this.imag * scalar);
    }

    public Complex conjugate() {
        return new Complex(this.real, -this.imag);
    }

    public double magnitude() {
        return Math.hypot(real, imag);
    }

    public double phase() {
        return Math.atan2(imag, real);
    }

    public double phaseDegrees() {
        return Math.toDegrees(phase());
    }

    public Complex square() {
        return multiply(this);
    }

    public Complex power(int n) {
        if (n == 0) {
            return new Complex(1, 0);
        }
        double magnitude = magnitude();
        double phase = phase();
        double newMagnitude = Math.pow(magnitude, n);
        double newPhase = n * phase;
        return new Complex(newMagnitude * Math.cos(newPhase), newMagnitude * Math.sin(newPhase));
    }

    public Complex sqrt() {
        double magnitude = magnitude();
        double phase = phase();
        double newMagnitude = Math.sqrt(magnitude);
        double newPhase = phase / 2;
        return new Complex(newMagnitude * Math.cos(newPhase), newMagnitude * Math.sin(newPhase));
    }

    public Complex exp() {
        double expReal = Math.exp(real);
        return new Complex(expReal * Math.cos(imag), expReal * Math.sin(imag));
    }

    public Complex log() {
        double magnitude = magnitude();
        double phase = phase();
        return new Complex(Math.log(magnitude), phase);
    }

    public Complex sin() {
        double expReal = Math.exp(imag);
        double expNegReal = Math.exp(-imag);
        double realPart = Math.sin(real) * (expReal + expNegReal) / 2;
        double imagPart = Math.cos(real) * (expReal - expNegReal) / 2;
        return new Complex(realPart, imagPart);
    }

    public Complex cos() {
        double expReal = Math.exp(imag);
        double expNegReal = Math.exp(-imag);
        double realPart = Math.cos(real) * (expReal + expNegReal) / 2;
        double imagPart = -Math.sin(real) * (expReal - expNegReal) / 2;
        return new Complex(realPart, imagPart);
    }

    public Complex divide(Complex other) {
        double denominator = other.real * other.real + other.imag * other.imag;
        if (denominator < EPSILON) {
            throw new ArithmeticException("Division by near-zero denominator: " + denominator);
        }
        double realPart = (real * other.real + imag * other.imag) / denominator;
        double imagPart = (imag * other.real - real * other.imag) / denominator;
        return new Complex(realPart, imagPart);
    }

    public Complex reciprocal() {
        double denominator = real * real + imag * imag;
        if (denominator < EPSILON) {
            throw new ArithmeticException("Division by near-zero denominator: " + denominator);
        }
        return new Complex(real / denominator, -imag / denominator);
    }

    public double abs() {
        return magnitude();
    }

    public double getReal() {
        return real;
    }

    public double getImaginary() {
        return imag;
    }

    public boolean isZero() {
        return Math.abs(real) < EPSILON && Math.abs(imag) < EPSILON;
    }

    public boolean isReal() {
        return Math.abs(imag) < EPSILON;
    }

    public boolean isPurelyImaginary() {
        return Math.abs(real) < EPSILON;
    }

    public static Complex unit(double phase) {
        return new Complex(Math.cos(phase), Math.sin(phase));
    }

    public static Complex unitDegrees(double phaseDegrees) {
        return unit(Math.toRadians(phaseDegrees));
    }

    public static Complex fromPolarDegrees(double magnitude, double phaseDegrees) {
        return fromPolar(magnitude, Math.toRadians(phaseDegrees));
    }

    @Override
    public String toString() {
        return String.format("(%.3f, %.3fi)", real, imag);
    }

    public String toStringFull() {
        if (imag >= 0) {
            return String.format("%.6f + %.6fi", real, imag);
        } else {
            return String.format("%.6f - %.6fi", real, -imag);
        }
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Complex complex = (Complex) obj;
        return Math.abs(complex.real - real) < EPSILON && Math.abs(complex.imag - imag) < EPSILON;
    }

    @Override
    public int hashCode() {
        return Double.hashCode(real) + 31 * Double.hashCode(imag);
    }
}