package com.yishape.lab.math.plot;

/**
 * 配对图对角线样式 / Pairplot Diagonal Style
 * <p>
 * 指定配对图中对角线单元格的显示样式。
 * Diagonal style for {@link IPlot#pairplot} diagonal cells.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public enum PairplotDiagonal {
    /** Univariate KDE curve */
    KDE,
    /** Histogram */
    HIST,
    /** Empty cell */
    NONE
}
