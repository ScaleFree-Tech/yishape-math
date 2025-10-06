package com.yishape.lab.math;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IVector;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class RereMathUtilTest {

    @Test
    public void testToPrimitiveWithNullValues() {
        // Test Double array with null values
        Double[] doubleArrayWithNulls = {1.0, null, 3.0, null, 5.0};
        double[] primitiveDoubleArray = RereMathUtil.toPrimitive(doubleArrayWithNulls);
        assertNotNull(primitiveDoubleArray);
        assertEquals(5, primitiveDoubleArray.length);
        assertEquals(1.0, primitiveDoubleArray[0], 0.001);
        assertEquals(0.0, primitiveDoubleArray[1], 0.001); // null should become 0.0
        assertEquals(3.0, primitiveDoubleArray[2], 0.001);
        assertEquals(0.0, primitiveDoubleArray[3], 0.001); // null should become 0.0
        assertEquals(5.0, primitiveDoubleArray[4], 0.001);
    }

    @Test
    public void testToPrimitiveWithAllNullValues() {
        // Test Double array with all null values
        Double[] doubleArrayAllNulls = {null, null, null};
        double[] primitiveDoubleArray = RereMathUtil.toPrimitive(doubleArrayAllNulls);
        assertNotNull(primitiveDoubleArray);
        assertEquals(3, primitiveDoubleArray.length);
        assertEquals(0.0, primitiveDoubleArray[0], 0.001);
        assertEquals(0.0, primitiveDoubleArray[1], 0.001);
        assertEquals(0.0, primitiveDoubleArray[2], 0.001);
    }

    @Test
    public void testToPrimitiveWithNullArray() {
        // Test with null array
        assertThrows(NullPointerException.class, () -> {
            RereMathUtil.toPrimitive((Double[]) null);
        });
    }

    @Test
    public void testIDoubleVectorWithNullValues() {
        // Test IDoubleVector creation with null values
        Double[] doubleArrayWithNulls = {1.0, null, 3.0};
        IDoubleVector vector = IDoubleVector.of(doubleArrayWithNulls);
        assertNotNull(vector);
        assertEquals(3, vector.length());
        assertEquals(1.0, vector.get(0), 0.001);
        assertEquals(0.0, vector.get(1), 0.001); // null should become 0.0
        assertEquals(3.0, vector.get(2), 0.001);
    }

    @Test
    public void testIVectorWithNullValues() {
        // Test IVector creation with null values
        Double[] doubleArrayWithNulls = {1.0, null, 3.0};
        IVector<Double> vector = IVector.of(doubleArrayWithNulls);
        assertNotNull(vector);
        assertEquals(3, vector.length());
        assertEquals(Double.valueOf(1.0), vector.get(0));
        assertEquals(Double.valueOf(0.0), vector.get(1)); // null should become 0.0
        assertEquals(Double.valueOf(3.0), vector.get(2));
    }
}