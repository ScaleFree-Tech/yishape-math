package com.yishape.lab.math.timeseries;

/**
 *
 * @author lteb2
 */
/**
 * 滤波结果类 / Filtering Result Class
 */
public class FilterResult {

    public final TimeSeriesData filtered;
    public final TimeSeriesData noise;
    public final double snr;
    public final String filterType;

    public FilterResult(TimeSeriesData filtered, TimeSeriesData noise, double snr, String filterType) {
        this.filtered = filtered;
        this.noise = noise;
        this.snr = snr;
        this.filterType = filterType;
    }
}
