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
 * @version 1.0
 * @since 1.0
 * @see Column 数据列接口 / Data column interface
 */
public enum ColumnType {
    String,Numeric
}
