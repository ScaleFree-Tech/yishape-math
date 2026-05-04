package com.yishape.lab.math.timeseries;

/**
 * 滤波结果类 / Filtering Result Class
 * <p>
 * 存储时间序列滤波操作的结果，包括滤波后的数据、噪声成分和信噪比。
 * 用于TimeSeriesFiltering类返回滤波处理的结果。
 * </p>
 * <p>
 * Stores the results of time series filtering operations, including filtered data,
 * noise component, and signal-to-noise ratio. Used by TimeSeriesFiltering class
 * to return the results of filtering processing.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class FilterResult {

    /** 滤波后的时间序列 / Filtered time series */
    public final TimeSeriesData filtered;
    /** 提取的噪声成分 / Extracted noise component */
    public final TimeSeriesData noise;
    /** 信噪比 / Signal-to-noise ratio */
    public final double snr;
    /** 滤波器类型 / Filter type */
    public final String filterType;

    /**
     * 构造函数 / Constructor
     *
     * @param filtered 滤波后的时间序列 / Filtered time series
     * @param noise 提取的噪声成分 / Extracted noise component
     * @param snr 信噪比 / Signal-to-noise ratio
     * @param filterType 滤波器类型 / Filter type
     */
    public FilterResult(TimeSeriesData filtered, TimeSeriesData noise, double snr, String filterType) {
        this.filtered = filtered;
        this.noise = noise;
        this.snr = snr;
        this.filterType = filterType;
    }
}
