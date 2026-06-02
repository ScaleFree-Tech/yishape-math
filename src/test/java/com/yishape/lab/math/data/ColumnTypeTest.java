package com.yishape.lab.math.data;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link ColumnType} enum.
 */
class ColumnTypeTest {

    @Test
    void isNumeric_trueForNumeric() {
        assertTrue(ColumnType.Numeric.isNumeric());
    }

    @Test
    void isNumeric_trueForInteger() {
        assertTrue(ColumnType.Integer.isNumeric());
    }

    @Test
    void isNumeric_falseForString() {
        assertFalse(ColumnType.String.isNumeric());
    }

    @Test
    void isNumeric_falseForBoolean() {
        assertFalse(ColumnType.Boolean.isNumeric());
    }

    @Test
    void isNumeric_falseForDateTime() {
        assertFalse(ColumnType.DateTime.isNumeric());
    }

    @Test
    void canBeNumeric_trueForString() {
        assertTrue(ColumnType.String.canBeNumeric());
    }

    @Test
    void canBeNumeric_trueForNumeric() {
        assertTrue(ColumnType.Numeric.canBeNumeric());
    }

    @Test
    void canBeNumeric_trueForInteger() {
        assertTrue(ColumnType.Integer.canBeNumeric());
    }

    @Test
    void canBeNumeric_falseForBoolean() {
        assertFalse(ColumnType.Boolean.canBeNumeric());
    }

    @Test
    void canBeNumeric_falseForDateTime() {
        assertFalse(ColumnType.DateTime.canBeNumeric());
    }

    @Test
    void allValuesExist() {
        assertEquals(5, ColumnType.values().length);
    }
}
