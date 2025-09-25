package com.reremouse.lab.math.signal.core;

/**
 *
 * @author lteb2
 */
public class Complex {
    public double real;
    public double imag;

    public Complex(double real, double imag) {
        this.real = real;
        this.imag = imag;
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
        return Math.sqrt(real * real + imag * imag);
    }

    /**
     * 计算复数的相位 / Calculate Phase of Complex Number
     * <p>
     * 计算复数的相位角（弧度）。
     * Calculate phase angle of complex number in radians.
     * </p>
     *
     * @return 相位角 / Phase angle
     */
    public double phase() {
        return Math.atan2(imag, real);
    }

    /**
     * 计算复数的相位（度） / Calculate Phase of Complex Number (Degrees)
     * <p>
     * 计算复数的相位角（度）。
     * Calculate phase angle of complex number in degrees.
     * </p>
     *
     * @return 相位角（度） / Phase angle in degrees
     */
    public double phaseDegrees() {
        return Math.toDegrees(phase());
    }

    /**
     * 计算复数的平方 / Calculate Square of Complex Number
     * <p>
     * 计算复数的平方。
     * Calculate square of complex number.
     * </p>
     *
     * @return 复数的平方 / Square of complex number
     */
    public Complex square() {
        return multiply(this);
    }

    /**
     * 计算复数的n次幂 / Calculate nth Power of Complex Number
     * <p>
     * 使用德摩弗公式计算复数的n次幂。
     * Calculate nth power of complex number using De Moivre's formula.
     * </p>
     *
     * @param n 幂次 / Power
     * @return 复数的n次幂 / nth power of complex number
     */
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

    /**
     * 计算复数的平方根 / Calculate Square Root of Complex Number
     * <p>
     * 计算复数的主平方根。
     * Calculate principal square root of complex number.
     * </p>
     *
     * @return 复数的主平方根 / Principal square root of complex number
     */
    public Complex sqrt() {
        double magnitude = magnitude();
        double phase = phase();
        
        double newMagnitude = Math.sqrt(magnitude);
        double newPhase = phase / 2;
        
        return new Complex(newMagnitude * Math.cos(newPhase), newMagnitude * Math.sin(newPhase));
    }

    /**
     * 计算复数的指数 / Calculate Exponential of Complex Number
     * <p>
     * 计算e的复数次幂。
     * Calculate e raised to the power of complex number.
     * </p>
     *
     * @return e的复数次幂 / e raised to the power of complex number
     */
    public Complex exp() {
        double expReal = Math.exp(real);
        return new Complex(expReal * Math.cos(imag), expReal * Math.sin(imag));
    }

    /**
     * 计算复数的自然对数 / Calculate Natural Logarithm of Complex Number
     * <p>
     * 计算复数的自然对数。
     * Calculate natural logarithm of complex number.
     * </p>
     *
     * @return 复数的自然对数 / Natural logarithm of complex number
     */
    public Complex log() {
        double magnitude = magnitude();
        double phase = phase();
        
        return new Complex(Math.log(magnitude), phase);
    }

    /**
     * 计算复数的正弦 / Calculate Sine of Complex Number
     * <p>
     * 计算复数的正弦值。
     * Calculate sine of complex number.
     * </p>
     *
     * @return 复数的正弦值 / Sine of complex number
     */
    public Complex sin() {
        double expReal = Math.exp(imag);
        double expNegReal = Math.exp(-imag);
        
        double realPart = Math.sin(real) * (expReal + expNegReal) / 2;
        double imagPart = Math.cos(real) * (expReal - expNegReal) / 2;
        
        return new Complex(realPart, imagPart);
    }

    /**
     * 计算复数的余弦 / Calculate Cosine of Complex Number
     * <p>
     * 计算复数的余弦值。
     * Calculate cosine of complex number.
     * </p>
     *
     * @return 复数的余弦值 / Cosine of complex number
     */
    public Complex cos() {
        double expReal = Math.exp(imag);
        double expNegReal = Math.exp(-imag);
        
        double realPart = Math.cos(real) * (expReal + expNegReal) / 2;
        double imagPart = -Math.sin(real) * (expReal - expNegReal) / 2;
        
        return new Complex(realPart, imagPart);
    }

    /**
     * 计算复数的除法 / Calculate Division of Complex Numbers
     * <p>
     * 计算两个复数的除法。
     * Calculate division of two complex numbers.
     * </p>
     *
     * @param other 另一个复数 / Other complex number
     * @return 除法结果 / Division result
     */
    public Complex divide(Complex other) {
        double denominator = other.real * other.real + other.imag * other.imag;
        if (denominator == 0) {
            throw new ArithmeticException("除零错误");
        }
        
        double realPart = (real * other.real + imag * other.imag) / denominator;
        double imagPart = (imag * other.real - real * other.imag) / denominator;
        
        return new Complex(realPart, imagPart);
    }

    /**
     * 计算复数的倒数 / Calculate Reciprocal of Complex Number
     * <p>
     * 计算复数的倒数。
     * Calculate reciprocal of complex number.
     * </p>
     *
     * @return 复数的倒数 / Reciprocal of complex number
     */
    public Complex reciprocal() {
        double denominator = real * real + imag * imag;
        if (denominator == 0) {
            throw new ArithmeticException("除零错误");
        }
        
        return new Complex(real / denominator, -imag / denominator);
    }

    /**
     * 计算复数的绝对值 / Calculate Absolute Value of Complex Number
     * <p>
     * 计算复数的绝对值（模长）。
     * Calculate absolute value (magnitude) of complex number.
     * </p>
     *
     * @return 复数的绝对值 / Absolute value of complex number
     */
    public double abs() {
        return magnitude();
    }

    /**
     * 计算复数的实部 / Get Real Part of Complex Number
     * <p>
     * 获取复数的实部。
     * Get real part of complex number.
     * </p>
     *
     * @return 实部 / Real part
     */
    public double getReal() {
        return real;
    }

    /**
     * 计算复数的虚部 / Get Imaginary Part of Complex Number
     * <p>
     * 获取复数的虚部。
     * Get imaginary part of complex number.
     * </p>
     *
     * @return 虚部 / Imaginary part
     */
    public double getImaginary() {
        return imag;
    }

    /**
     * 检查复数是否为零 / Check if Complex Number is Zero
     * <p>
     * 检查复数是否为零。
     * Check if complex number is zero.
     * </p>
     *
     * @return 是否为零 / Whether it's zero
     */
    public boolean isZero() {
        return real == 0 && imag == 0;
    }

    /**
     * 检查复数是否为实数 / Check if Complex Number is Real
     * <p>
     * 检查复数是否为实数（虚部为零）。
     * Check if complex number is real (imaginary part is zero).
     * </p>
     *
     * @return 是否为实数 / Whether it's real
     */
    public boolean isReal() {
        return imag == 0;
    }

    /**
     * 检查复数是否为纯虚数 / Check if Complex Number is Purely Imaginary
     * <p>
     * 检查复数是否为纯虚数（实部为零）。
     * Check if complex number is purely imaginary (real part is zero).
     * </p>
     *
     * @return 是否为纯虚数 / Whether it's purely imaginary
     */
    public boolean isPurelyImaginary() {
        return real == 0;
    }

    /**
     * 创建单位复数 / Create Unit Complex Number
     * <p>
     * 创建指定相位的单位复数。
     * Create unit complex number with specified phase.
     * </p>
     *
     * @param phase 相位（弧度） / Phase in radians
     * @return 单位复数 / Unit complex number
     */
    public static Complex unit(double phase) {
        return new Complex(Math.cos(phase), Math.sin(phase));
    }

    /**
     * 创建单位复数（度） / Create Unit Complex Number (Degrees)
     * <p>
     * 创建指定相位的单位复数（度）。
     * Create unit complex number with specified phase in degrees.
     * </p>
     *
     * @param phaseDegrees 相位（度） / Phase in degrees
     * @return 单位复数 / Unit complex number
     */
    public static Complex unitDegrees(double phaseDegrees) {
        return unit(Math.toRadians(phaseDegrees));
    }

    /**
     * 从极坐标创建复数 / Create Complex Number from Polar Coordinates
     * <p>
     * 从极坐标（幅度和相位）创建复数。
     * Create complex number from polar coordinates (magnitude and phase).
     * </p>
     *
     * @param magnitude 幅度 / Magnitude
     * @param phase 相位（弧度） / Phase in radians
     * @return 复数 / Complex number
     */
    public static Complex fromPolar(double magnitude, double phase) {
        return new Complex(magnitude * Math.cos(phase), magnitude * Math.sin(phase));
    }

    /**
     * 从极坐标创建复数（度） / Create Complex Number from Polar Coordinates (Degrees)
     * <p>
     * 从极坐标（幅度和相位）创建复数（度）。
     * Create complex number from polar coordinates (magnitude and phase) in degrees.
     * </p>
     *
     * @param magnitude 幅度 / Magnitude
     * @param phaseDegrees 相位（度） / Phase in degrees
     * @return 复数 / Complex number
     */
    public static Complex fromPolarDegrees(double magnitude, double phaseDegrees) {
        return fromPolar(magnitude, Math.toRadians(phaseDegrees));
    }

    @Override
    public String toString() {
        return String.format("(%.3f, %.3fi)", real, imag);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Complex complex = (Complex) obj;
        return Double.compare(complex.real, real) == 0 && Double.compare(complex.imag, imag) == 0;
    }

    @Override
    public int hashCode() {
        return Double.hashCode(real) + 31 * Double.hashCode(imag);
    }
}