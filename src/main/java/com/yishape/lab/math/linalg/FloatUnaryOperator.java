package com.yishape.lab.math.linalg;

/**
 * A functional interface for operating on primitive {@code float} values
 * without boxing overhead. Analogous to {@link java.util.function.DoubleUnaryOperator}.
 */
@FunctionalInterface
public interface FloatUnaryOperator {
    float applyAsFloat(float operand);
}
