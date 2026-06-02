package com.yishape.lab.math.data;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 数据列接口 / Data Column Interface
 * <p>
 * 表示DataFrame中的一列数据，支持名称、类型和数据存储。
 * 提供列数据到向量的转换功能，用于数值分析。
 * 所有数值列操作均依赖 {@link IVector} 和 {@link Linalg} 提供的能力。
 * </p>
 * <p>
 * Represents a column of data in DataFrame, supporting name, type and data storage.
 * Provides conversion from column data to vectors for numerical analysis.
 * All numeric column operations rely on capabilities provided by {@link IVector} and {@link Linalg}.
 * </p>
 *
 * @author lteb2
 * @version 1.1
 * @since 1.0
 * @see ColumnType 列类型枚举 / Column type enum
 * @see IVector 向量接口 / Vector interface
 * @see Linalg 线性代数工厂类 / Linear algebra factory class
 */
public class Column implements Serializable {

    private static final long serialVersionUID = 1L;

    /** 列名 */
    private String name;
    /** 列的类型，默认 Numeric */
    private ColumnType columnType = ColumnType.Numeric;
    /** 列数据，数值类型使用 Double 存储，缺失值使用 Double.NaN 表示 */
    private List<Object> data = new ArrayList<>();

    /**
     * 获取列名 / Get column name
     * @return 列名 / Column name
     */
    public String getName() {
        return name;
    }

    /**
     * 设置列名 / Set column name
     * @param name 列名 / Column name
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * 获取列类型 / Get column type
     * @return 列类型 / Column type
     */
    public ColumnType getColumnType() {
        return columnType;
    }

    /**
     * 设置列类型 / Set column type
     * @param columnType 列类型 / Column type
     */
    public void setColumnType(ColumnType columnType) {
        this.columnType = columnType;
    }

    /**
     * 获取列数据 / Get column data
     * @return 数据列表 / Data list
     */
    public List<Object> getData() {
        return data;
    }

    /**
     * 设置列数据 / Set column data
     * @param data 数据列表 / Data list
     */
    public void setData(List<Object> data) {
        this.data = data;
    }

    /**
     * Numeric 或 Integer 列转为 {@link IVector} {@code <Double>}；
     * 元素可为任意 {@link Number}（如 CSV 解析得到的 {@link Double}）。
     * String 列或包含非数值元素的列将抛出异常。
     * 缺失值（Double.NaN）会被保留在向量中。
     *
     * @return 双精度向量
     * @throws IllegalStateException 如果列类型不是数值类型
     */
    public IVector<Double> toVec() {
        if (!columnType.isNumeric()) {
            throw new IllegalStateException(
                    "Column \"" + name + "\" is not numeric (type=" + columnType + "); use toStringList() / toStringArray() instead.");
        }
        int n = data.size();
        double[] buf = new double[n];
        for (int i = 0; i < n; i++) {
            Object o = data.get(i);
            if (o instanceof Number number) {
                buf[i] = number.doubleValue();
            } else if (o == null || (o instanceof String s && s.trim().isEmpty())) {
                buf[i] = Double.NaN;
            } else {
                throw new IllegalStateException(
                        "Column \"" + name + "\" row " + i + " is not a Number: " + o);
            }
        }
        return Linalg.vector(buf);
    }

    /**
     * 将数值列转换为 double[] 数组。
     * 依赖 {@link IVector#toDoubleArray()} 实现。
     *
     * @return double 数组
     * @throws IllegalStateException 如果列类型不是数值类型
     */
    public double[] toDoubleArray() {
        return toVec().toDoubleArray();
    }

    /**
     * 计算数值列的统计信息，利用 {@link IVector} 的丰富统计方法。
     * 返回包含 count、mean、std、min、max、median、q1、q3 的统计摘要。
     * 缺失值（NaN）会被忽略。
     *
     * @return 统计信息映射
     */
    public java.util.Map<String, Double> statistics() {
        IVector<Double> vec = toVec();
        java.util.Map<String, Double> stats = new java.util.LinkedHashMap<>();
        stats.put("count", (double) vec.length());
        stats.put("mean", vec.meanValue());
        stats.put("std", vec.stdValue());
        stats.put("min", vec.minValue());
        stats.put("max", vec.maxValue());
        stats.put("median", vec.median());
        stats.put("q1", vec.q1());
        stats.put("q3", vec.q3());
        return stats;
    }

    /**
     * 如果列类型为 String，返回；如果是 Numeric/Integer，转换为 String 返回。
     * @return 字符串列表 / String list
     */
    public List<String> toStringList() {
        return data.stream().map(e -> e == null ? "NaN" : e.toString()).toList();
    }

    /**
     * 转换为字符串数组 / Convert to string array
     * @return 字符串数组 / String array
     */
    public String[] toStringArray() {
        return data.stream().map(e -> e == null ? "NaN" : e.toString()).toArray(String[]::new);
    }

    /**
     * 检查是否包含缺失值（null 或 Double.NaN）
     * @return 是否包含缺失值
     */
    public boolean hasMissingValues() {
        for (Object o : data) {
            if (o == null) return true;
            if (o instanceof Double d && Double.isNaN(d)) return true;
        }
        return false;
    }

    /**
     * 返回缺失值的数量
     * @return 缺失值数量
     */
    public int missingCount() {
        int count = 0;
        for (Object o : data) {
            if (o == null) {
                count++;
            } else if (o instanceof Double d && Double.isNaN(d)) {
                count++;
            }
        }
        return count;
    }

    /**
     * 返回非缺失值的数量
     * @return 非缺失值数量
     */
    public int validCount() {
        return data.size() - missingCount();
    }
}
