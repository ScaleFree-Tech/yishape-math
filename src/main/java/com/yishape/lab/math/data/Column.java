package com.yishape.lab.math.data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import com.yishape.lab.math.linalg.IVector;

/**
 * 数据列接口 / Data Column Interface
 * <p>
 * 表示DataFrame中的一列数据，支持名称、类型和数据存储。
 * 提供列数据到向量的转换功能，用于数值分析。
 * </p>
 * <p>
 * Represents a column of data in DataFrame, supporting name, type and data storage.
 * Provides conversion from column data to vectors for numerical analysis.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 * @see ColumnType 列类型枚举 / Column type enum
 * @see IVector 向量接口 / Vector interface
 */
public class Column  implements Serializable{
    
    private String name;//列名
    private ColumnType columnType = ColumnType.Numeric;//列的类型
    
    private List<Object> data = new ArrayList();

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
     * Numeric 列转为 {@link IVector}{@code <Double>}；元素可为任意 {@link Number}（如 CSV 解析得到的 {@link Double}）。
     * String 列或未识别的非数值元素将抛出异常。
     *
     * @return 双精度向量
     */
    public IVector<Double> toVec() {
        if (columnType != ColumnType.Numeric) {
            throw new IllegalStateException(
                    "Column \"" + name + "\" is not numeric; use toStringList() / toStringArray() instead.");
        }
        int n = data.size();
        double[] buf = new double[n];
        for (int i = 0; i < n; i++) {
            Object o = data.get(i);
            if (!(o instanceof Number)) {
                throw new IllegalStateException(
                        "Column \"" + name + "\" row " + i + " is not a Number: " + o);
            }
            buf[i] = ((Number) o).doubleValue();
        }
        return IVector.of(buf);
    }
    
    /**
     * 如果列类型为String，返回；如果是Float，转换为String返回 / Return if column type is String; if Float, convert to String and return
     * @return 字符串列表 / String list
     */
    public List<String> toStringList() {
        return data.stream().map(e->e.toString()).toList();
    }

    /**
     * 转换为字符串数组 / Convert to string array
     * @return 字符串数组 / String array
     */
    public String[] toStringArray() {
        return data.stream().map(e->e.toString()).toArray(String[]::new);
    }
}
