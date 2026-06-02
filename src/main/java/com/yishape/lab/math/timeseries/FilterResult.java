package com.yishape.lab.math.timeseries;

import com.yishape.lab.math.linalg.IVector;

/**
 * 滤波结果类 / Filtering Result Class
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class FilterResult {

    public final IVector<Double> filtered;
    public final IVector<Double> noise;
    public final double snr;
    public final String filterType;

    public FilterResult(IVector<Double> filtered, IVector<Double> noise, double snr, String filterType) {
        this.filtered = filtered;
        this.noise = noise;
        this.snr = snr;
        this.filterType = filterType;
    }
}
