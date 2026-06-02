package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.data.Column;
import com.yishape.lab.math.data.ColumnType;
import com.yishape.lab.math.linalg.IMatrix;

import java.util.ArrayList;
import java.util.List;

/**
 * DataFrame 友好的预处理器接口
 *
 * <p>
 * 继承自 {@link ITransform}，额外提供 DataFrame 级别的操作能力。
 * 实现类应同时支持 Matrix 输入（通过父接口）和 DataFrame 输入（通过本接口）。
 * </p>
 *
 * <h3>设计原则</h3>
 * <ul>
 * <li>DataFrame 版本的 fit/transform 不替代 Matrix 版本，而是并行提供</li>
 * <li>DataFrame 操作默认实现：提取指定列 → 委托 Matrix 版本 → 结果塞回</li>
 * <li>保留 DataFrame 中未参与处理的列（数值列 + 非数值列）</li>
 * </ul>
 *
 * @param <T> 数值类型
 */
public interface IDataFrameTransform<T extends Number> extends ITransform<T> {

    // ==================== DataFrame 版本的 Fit ====================

    /**
     * 从 DataFrame 的指定列进行 fit。
     *
     * <p>
     * 默认实现：从 df 提取 columns → 转为 IMatrix → 调用父接口的 fit(IMatrix)。
     * 子类可覆盖以实现更高效的 DataFrame 直接处理。
     * </p>
     *
     * @param df 输入 DataFrame
     * @param columns 要处理的列名（必须是数值列）
     * @return 当前实例，支持链式调用
     * @throws IllegalArgumentException 如果任何指定的列不存在或非数值类型
     */
    default IDataFrameTransform<T> fit(DataFrame df, String... columns) {
        if (columns == null || columns.length == 0) {
            throw new IllegalArgumentException("必须指定至少一列 / At least one column must be specified");
        }

        // 收集所有指定列
        DataFrame subset = new DataFrame();
        for (String colName : columns) {
            Column col = df.getColumnByName(colName);
            if (col == null) {
                throw new IllegalArgumentException("列不存在 / Column not found: " + colName);
            }
            subset.addColumn(col);
        }

        // 转为 IMatrix 并 fit（父接口方法）
        return (IDataFrameTransform<T>) fit(subset.toMatrix());
    }

    /**
     * fit 的变体，自动推断所有数值列。
     * @param df 输入 DataFrame
     * @return 当前实例
     */
    default IDataFrameTransform<T> fitAllNumericColumns(DataFrame df) {
        String[] numericCols = df.selectNumeric().getColumnNames()
            .toArray(new String[0]);
        return fit(df, numericCols);
    }

    // ==================== DataFrame 版本的 Transform ====================

    /**
     * 将 DataFrame 中的指定列进行变换，返回新的 DataFrame。
     *
     * <p>
     * 默认实现：
     * <ul>
     * <li>提取 columns 对应的 IMatrix</li>
     * <li>调用父接口 transform(IMatrix)</li>
     * <li>将变换后的列替换回原 DataFrame 的对应位置</li>
     * <li>保留 df 中未参与变换的列（数值列 + 非数值列）</li>
     * </ul>
     * </p>
     *
     * @param df 输入 DataFrame
     * @param columns 要变换的列名
     * @return 变换后的新 DataFrame
     * @throws IllegalStateException 如果未 fit
     */
    default DataFrame transform(DataFrame df, String... columns) {
        if (!ifTrained()) {
            throw new IllegalStateException("Must be fitted before transform");
        }

        // 提取列子集
        DataFrame subset = new DataFrame();
        for (String colName : columns) {
            Column col = df.getColumnByName(colName);
            if (col == null) {
                throw new IllegalArgumentException("列不存在 / Column not found: " + colName);
            }
            subset.addColumn(col);
        }

        // 变换
        IMatrix<T> transformed = transform(subset.toMatrix());

        // 构建结果：保留 df 中不在 columns 里的列
        DataFrame result = new DataFrame();

        // 先加未指定的列（保留原样）
        for (Column col : df.getColumns()) {
            boolean isTransformed = false;
            for (String tCol : columns) {
                if (col.getName().equals(tCol)) {
                    isTransformed = true;
                    break;
                }
            }
            if (!isTransformed) {
                result.addColumn(col);
            }
        }

        // 再追加变换后的列
        int rows = transformed.rows();
        int cols = transformed.cols();
        for (int j = 0; j < cols; j++) {
            Column newCol = new Column();
            newCol.setName(df.getColumnByName(columns[j]).getName()); // 复用原列名
            newCol.setColumnType(ColumnType.Numeric);
            List<Object> data = new ArrayList<>();
            for (int i = 0; i < rows; i++) {
                data.add(transformed.get(i, j));
            }
            newCol.setData(data);
            result.addColumn(newCol);
        }

        return result;
    }

    /**
     * DataFrame 版本的 fit + transform。
     * @param df 输入 DataFrame
     * @param columns 要处理的列名
     * @return 变换后的新 DataFrame
     */
    default DataFrame fitTransform(DataFrame df, String... columns) {
        fit(df, columns);
        return transform(df, columns);
    }

    // ==================== 结果查询 ====================

    /**
     * 获取被处理的列名列表。
     * @return 列名数组，未 fit 返回空数组
     */
    String[] getInputColumnNames();

    /**
     * 获取变换后输出的列名列表。
     * 默认返回 getInputColumnNames()（逐列变换时），
     * 子类（如 OneHotEncoder、PolynomialFeatures）应覆盖。
     * @return 输出列名数组
     */
    default String[] getOutputColumnNames() {
        return getInputColumnNames();
    }
}
