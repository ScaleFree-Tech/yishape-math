package com.yishape.lab.math.data;

/**
 * 列数据类型枚举 / Column Data Type Enum
 * <p>
 * 定义数据列支持的数据类型，包括字符串类型和数值类型。
 * 用于DataFrame中列的元数据管理，标识列中存储的数据类型。
 * </p>
 * <p>
 * Defines the data types supported by data columns, including string type and numeric type.
 * Used for column metadata management in DataFrame, identifying the data type stored in each column.
 * </p>
 *
 * @author lteb2
 * @version 1.1
 * @since 1.0
 * @see Column 数据列接口 / Data column interface
 */
public enum ColumnType {
    /** 字符串类型 / String type */
    String,
    /** 数值类型（Double）/ Numeric type (Double) */
    Numeric,
    /** 整数类型 / Integer type */
    Integer,
    /** 布尔类型 / Boolean type */
    Boolean,
    /** 日期时间类型 / DateTime type */
    DateTime;

    /**
     * 检查是否为数值类型（包括 Integer、Numeric）
     * @return 是否为数值类型
     */
    public boolean isNumeric() {
        return this == Numeric || this == Integer;
    }

    /**
     * 检查是否可以转换为数值类型
     * @return 是否可以转换
     */
    public boolean canBeNumeric() {
        return this == String || this == Numeric || this == Integer;
    }
}
