package com.yishape.lab.math.linalg;

/**
 * A functional interface for operating on primitive {@code float[]} arrays
 * without boxing overhead, returning a {@code float} scalar. Analogous to
 * {@link java.util.function.ToDoubleFunction} but for {@code float[] → float}.
 */
@FunctionalInterface
public interface ToFloatFunction {
    float applyAsFloat(float[] values);
}
