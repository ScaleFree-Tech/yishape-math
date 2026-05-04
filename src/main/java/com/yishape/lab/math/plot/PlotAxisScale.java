package com.yishape.lab.math.plot;

/**
 * 坐标轴刻度类型 / Cartesian Axis Scale Type
 * <p>
 * 笛卡尔坐标轴刻度类型（线性或对数），类似于Matplotlib/Seaborn的坐标轴刻度。
 * Cartesian axis scale (linear or logarithmic), analogous to Matplotlib/Seaborn axis scale.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public enum PlotAxisScale {
    LINEAR,
    /** Base-10 log scale; values must be strictly positive for rendering. */
    LOG
}
