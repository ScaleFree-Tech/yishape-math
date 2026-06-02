package com.yishape.lab.math.data;

import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link Column} class.
 */
class ColumnTest {

    // ==================== Basic Properties ====================

    @Test
    void defaultConstructor_createsEmptyNumericColumn() {
        Column col = new Column();
        assertEquals(ColumnType.Numeric, col.getColumnType());
        assertNotNull(col.getData());
        assertTrue(col.getData().isEmpty());
    }

    @Test
    void getNameAndSetName() {
        Column col = new Column();
        col.setName("age");
        assertEquals("age", col.getName());
    }

    @Test
    void getColumnTypeAndSetColumnType() {
        Column col = new Column();
        col.setColumnType(ColumnType.String);
        assertEquals(ColumnType.String, col.getColumnType());
    }

    @Test
    void getDataAndSetData() {
        Column col = new Column();
        List<Object> data = Arrays.asList(1.0, 2.0, 3.0);
        col.setData(data);
        assertEquals(3, col.getData().size());
    }

    // ==================== toVec ====================

    @Test
    void toVec_numericColumn() {
        Column col = new Column();
        col.setColumnType(ColumnType.Numeric);
        col.setData(Arrays.asList(1.0, 2.0, 3.0));
        var vec = col.toVec();
        assertEquals(3, vec.size());
        assertEquals(1.0, vec.get(0), 1e-10);
        assertEquals(2.0, vec.get(1), 1e-10);
        assertEquals(3.0, vec.get(2), 1e-10);
    }

    @Test
    void toVec_integerColumn() {
        Column col = new Column();
        col.setColumnType(ColumnType.Integer);
        col.setData(Arrays.asList(10, 20, 30));
        var vec = col.toVec();
        assertEquals(10.0, vec.get(0), 1e-10);
        assertEquals(20.0, vec.get(1), 1e-10);
    }

    @Test
    void toVec_stringColumn_throws() {
        Column col = new Column();
        col.setColumnType(ColumnType.String);
        col.setData(Arrays.asList("a", "b", "c"));
        assertThrows(IllegalStateException.class, col::toVec);
    }

    @Test
    void toVec_booleanColumn_throws() {
        Column col = new Column();
        col.setColumnType(ColumnType.Boolean);
        col.setData(Arrays.asList(true, false));
        assertThrows(IllegalStateException.class, col::toVec);
    }

    @Test
    void toDoubleArray() {
        Column col = new Column();
        col.setColumnType(ColumnType.Numeric);
        col.setData(Arrays.asList(1.5, 2.5, 3.5));
        double[] arr = col.toDoubleArray();
        assertEquals(3, arr.length);
        assertEquals(1.5, arr[0], 1e-10);
        assertEquals(3.5, arr[2], 1e-10);
    }

    // ==================== Missing Values ====================

    @Test
    void hasMissingValues_falseWhenComplete() {
        Column col = new Column();
        col.setColumnType(ColumnType.Numeric);
        col.setData(Arrays.asList(1.0, 2.0, 3.0));
        assertFalse(col.hasMissingValues());
    }

    @Test
    void hasMissingValues_trueWithNull() {
        Column col = new Column();
        col.setColumnType(ColumnType.Numeric);
        col.setData(Arrays.asList(1.0, null, 3.0));
        assertTrue(col.hasMissingValues());
    }

    @Test
    void hasMissingValues_trueWithNaN() {
        Column col = new Column();
        col.setColumnType(ColumnType.Numeric);
        col.setData(Arrays.asList(1.0, Double.NaN, 3.0));
        assertTrue(col.hasMissingValues());
    }

    @Test
    void missingCount() {
        Column col = new Column();
        col.setColumnType(ColumnType.Numeric);
        col.setData(Arrays.asList(1.0, null, Double.NaN, 4.0));
        assertEquals(2, col.missingCount());
    }

    @Test
    void validCount() {
        Column col = new Column();
        col.setColumnType(ColumnType.Numeric);
        col.setData(Arrays.asList(1.0, null, Double.NaN, 4.0));
        assertEquals(2, col.validCount());
    }

    // ==================== String Conversion ====================

    @Test
    void toStringList_nullsBecomesNaN() {
        Column col = new Column();
        col.setColumnType(ColumnType.Numeric);
        col.setData(Arrays.asList(1.0, null, 3.0));
        List<String> strings = col.toStringList();
        assertEquals(3, strings.size());
        assertEquals("1.0", strings.get(0));
        assertEquals("NaN", strings.get(1));
        assertEquals("3.0", strings.get(2));
    }

    @Test
    void toStringArray() {
        Column col = new Column();
        col.setColumnType(ColumnType.String);
        col.setData(Arrays.asList("hello", null, "world"));
        String[] arr = col.toStringArray();
        assertEquals(3, arr.length);
        assertEquals("hello", arr[0]);
        assertEquals("NaN", arr[1]);
        assertEquals("world", arr[2]);
    }

    // ==================== Statistics ====================

    @Test
    void statistics_basicMetrics() {
        Column col = new Column();
        col.setColumnType(ColumnType.Numeric);
        col.setData(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0));
        Map<String, Double> stats = col.statistics();
        assertEquals(5.0, stats.get("count"), 1e-10);
        assertEquals(3.0, stats.get("mean"), 1e-10);
        assertEquals(1.0, stats.get("min"), 1e-10);
        assertEquals(5.0, stats.get("max"), 1e-10);
        assertEquals(3.0, stats.get("median"), 1e-10);
    }
}
